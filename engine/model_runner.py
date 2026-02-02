import pickle
import torch
from abc import ABC, abstractmethod
import torch.distributed as dist
from utils import Logger, Config
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory


class ModelRunner(ABC):

    shared_memory_name = "mini_llm_model_shm"
    shared_memory:SharedMemory | None = None
    graphs = {}
    graph_pool = None

    def __init__(self, config: Config, rank:int):
        self.config = config
        hf_config = config.hf_config
        self.logger = Logger()
        self.logger.info("Initializing ModelRunner")
        self.block_size = config.kvcache_block_size
        self.enforced_eager = config.enforce_eager
        self.model_path = config.model_path
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.logger.info(f"ModelRunner initialized with rank {self.rank} out of {self.world_size}")

        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)
        self.logger.info(f"Process group initialized with backend 'nccl' for rank {self.rank}")

        default_dtype = torch.get_default_dtype()
        self.logger.info(f"Default torch dtype is {default_dtype}")
        torch.set_default_dtype(hf_config.torch_dtype)
        self.logger.info(f"Torch dtype set to {hf_config.torch_dtype} for rank {self.rank}")
        torch.set_default_device('cuda')
        self.logger.info(f"Torch default device set to 'cuda' for rank {self.rank}")


    def run(self, input_data):
        # Placeholder for model inference logic
        return f"Running model at {self.model_path} with input {input_data}"
    
    @abstractmethod
    def exit(self):
        if not self.enforced_eager:
           del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()
        self.logger.info(f"ModelRunner with rank {self.rank} has exited cleanly.")

    def call(self, method_name: str, *args):
        self.logger.info(f"Rank {self.rank} calling method: {method_name} with args: {args}")
        method = getattr(self, method_name, None)
        if not method:
            self.logger.error(f"Method {method_name} not found in ModelRunner.")
            return None
        return method(*args)
    

class MasterModelRunner(ModelRunner):
    def __init__(self, config: Config, rank:int, events: list[Event]):
        super().__init__(config, rank)
        self.events = events
        if self.world_size > 1:
            self.shared_memory = SharedMemory(name=self.shared_memory_name, create=True, size=1024 * 1024) # 1MB is enough
            dist.barrier()
            self.logger.info(f"Master rank {self.rank} created shared memory: {self.shared_memory_name}")
        else:
            self.logger.info(f"Single rank mode, no shared memory created.")

    def call(self, method_name: str, *args):
        if self.world_size > 1:
            self.write_shared_memory(method_name, args)

        return super().call(method_name, *args)
    
    def write_shared_memory(self, method_name: str, args):
        assert self.shared_memory is not None and self.world_size > 1 and self.rank == 0, "Shared memory must be initialized for master rank."
        data = pickle.dumps((method_name, *args))
        bytes_to_write = len(data)
        assert bytes_to_write + 4 <= self.shared_memory.size, "Data exceeds shared memory size."
        self.shared_memory.buf[:4] = bytes_to_write.to_bytes(4, "little")
        self.shared_memory.buf[4:4+bytes_to_write] = data
        self.logger.debug(f"Master rank {self.rank} wrote method: {method_name} with args: {args} to shared memory.")
        
        # Notify slaves
        for event in self.events:
            event.set()

    def exit(self):
        if self.world_size > 1:
            self.shared_memory.close()
            dist.barrier()
            self.shared_memory.unlink()
            self.logger.info(f"Master rank {self.rank} unlinked shared memory: {self.shared_memory_name}")
        super().exit()

class SlaveModelRunner(ModelRunner):
    def __init__(self, config: Config, rank:int, event: Event):
        super().__init__(config, rank)
        self.event = event
        
        # we need wait for the master to create the shared memory
        dist.barrier()
        self.shared_memory = SharedMemory(name=self.shared_memory_name)

        self.loop()


    def loop(self):
        self.logger.info(f"Slave rank {self.rank} entering wait loop.")
        while True:
            method_name, args = self.read_shared_memory()
            self.call(method_name, *args)
            if method_name == "exit":
                break
        self.logger.info(f"Slave rank {self.rank} detected event set, exiting loop.")


    def read_shared_memory(self):
        assert self.shared_memory is not None and self.world_size > 1 and self.rank > 0, "Shared memory must be initialized for slave ranks."
        self.event.wait()
        bytes_to_read = int.from_bytes(self.shared_memory.buf[:4], "little")
        method_name, *args = pickle.loads(self.shared_memory.buf[4:4+bytes_to_read])
        self.logger.debug(f"Slave rank {self.rank} read method: {method_name} with args: {args} from shared memory.")
        self.event.clear()
        return method_name, args
    
    def exit(self):
        if self.shared_memory:
            self.shared_memory.close()
            dist.barrier()
        super().exit()