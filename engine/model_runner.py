import pickle
from typing import override
import torch
from abc import ABC, abstractmethod
import torch.distributed as dist
from engine.sequence import Sequence
from layers import Sampler
from models.llama3 import LlamaForCausalLM
from utils import Logger, Config, Context, ContextManager
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory


class ModelRunner(ABC):

    '''Slots for CUDA graph capture'''
    CUDA_GRAPH_CAPTURE_SIZE = 512

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
        torch.set_default_dtype(hf_config.dtype)
        self.logger.info(f"Torch dtype set to {hf_config.dtype} for rank {self.rank}")
        torch.set_default_device('cuda')
        self.logger.info(f"Torch default device set to 'cuda' for rank {self.rank}")

        self.model = LlamaForCausalLM(config=hf_config)
        self.load_model()
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kvcache()

        if not self.enforced_eager:
            self.capture_cuda_graphs()
        torch.set_default_device('cpu')
        torch.set_default_dtype(default_dtype)



    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        '''
        This is the default run method that both Master and Slave ModelRunners use to process sequences.
        Besides this, the MasterModelRunner will write this method to shared memory for SlaveModelRunners to read and execute.
        
        :param self: Description
        :param seqs: Description
        :type seqs: list[Sequence]
        :param is_prefill: Description
        :type is_prefill: bool
        :return: Description
        :rtype: list[int]
        '''
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else []
        return token_ids

    
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
    
    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mappings = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq) # seq.num_tokens
            # we only prefill the uncached tokens.
            input_ids.extend(seq[seq.num_cached_tokens:])

            # positions refer to the query token_id position in the kv cache.
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            
            # warming up block table is None.
            if not seq.block_table:
                continue
            
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * Sequence.block_size
                if i != seq.num_blocks - 1:
                    end = start + seq.block_table[i] * Sequence.block_size
                else:
                    end = start + seq.last_block_num_tokens
                
                slot_mappings.append(list(range(start, end)))

            if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
                block_tables = self.prepare_block_tables(seqs)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mappings_tensor = torch.tensor(slot_mappings, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context = Context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q_tensor,
            cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mappings_tensor,
            block_tables=block_tables
        )
        ContextManager().set_default_context(context)
        return input_ids_tensor, positions_tensor
    
    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        positions = []
        slot_mappings = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token_id)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # we only need to map the block where the last token resides.
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mappings_tensor = torch.tensor(slot_mappings, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context = Context(
            is_prefill=False,
            slot_mapping=slot_mappings_tensor,
            context_lens=context_lens_tensor
        )
        ContextManager().set_default_context(context)
        return input_ids_tensor, positions_tensor

    
    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        '''
        We will make all the block tables the same length by padding -1 to the end for all the sequences.
        '''
        max_block_table_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_block_table_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.enforced_eager and input_ids.size(0) > self.CUDA_GRAPH_CAPTURE_SIZE:
            self.logger.debug(f"Running model eagerly on rank {self.rank} due to enforced eager mode with input_ids shape: {input_ids.shape} and positions shape: {positions.shape}")
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            self.logger.debug(f"Running model on rank {self.rank} with input_ids shape: {input_ids.shape} and positions shape: {positions.shape} in CUDA graph mode.")
            return torch.empty()  # Placeholder for CUDA graph execution
    
    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    
    def load_model(self):
        self.logger.error("Model loading not implemented. Please implement the load_model method to load model weights.")

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_length = self.config.max_model_length
        num_seqs = min(self.config.max_num_seqs, max_num_batched_tokens // max_model_length)
        seqs = [Sequence([0]*max_model_length) for _ in range(num_seqs)]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    def allocate_kvcache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()['alloc_bytes.all.peak']
        current = torch.cuda.memory_stats()['allocated_bytes.all.current']
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads)
        block_size_in_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_size_in_bytes
        assert config.num_kvcache_blocks > 0, "Not enough GPU memory to allocate KV cache."
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, dtype=hf_config.dtype, device='cuda')
        self.logger.info(f"Allocated KV cache with {config.num_kvcache_blocks} blocks on rank {self.rank}.")
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def capture_cuda_graphs(self):
        self.logger.error("CUDA graph capture not implemented. Please implement the capture_cuda_graphs method to capture CUDA graphs.")


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


    @override
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        '''
        Master will write the 'run' command to shared memory for slaves before executing it locally.
        
        :param self: Description
        :param seqs: Description
        :type seqs: list[Sequence]
        :param is_prefill: Description
        :type is_prefill: bool
        :return: Description
        :rtype: list[int]
        '''
        if self.world_size > 1:
            self.write_shared_memory("run", (seqs, is_prefill))
            self.logger.info(f"Master rank {self.rank} wrote 'run' command to shared memory for slaves.")
        return super().run(seqs, is_prefill)

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