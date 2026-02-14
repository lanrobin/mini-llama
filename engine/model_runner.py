import bisect
import pickle
from typing import override
import os
import torch
from torch import nn
from abc import ABC, abstractmethod
import torch.distributed as dist
from engine.sequence import Sequence
from layers import Sampler
from llama import LlamaForCausalLM
from utils import Logger, Config, Context, ContextManager, load_weights_from_safetensors
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory


class ModelRunner(ABC):
    '''
    Abstract base class for model runners. This class provides the common interface and functionality for both master and slave model runners.
    
    MasterModelRunner will handle the communication with the scheduler and write commands to shared memory for SlaveModelRunners to read and execute and also execute the commands locally.
    Then it will collect all the result from SlaveModelRunners and return the final result to the scheduler.
    
    SlaveModelRunner will wait for the master to write commands to shared memory, read the commands, execute them and write the results back to shared memory for the master to read.
    '''
    #Slots for CUDA graph capture
    CUDA_GRAPH_CAPTURE_SIZE = 512

    shared_memory_name = "mini_llm_model_shm"
    shared_memory:SharedMemory | None = None
    graphs = {}
    graph_pool = None

    def __init__(self, config: Config, rank:int):
        self.config = config
        hf_config = config.hf_config
        self.logger = Logger()
        self.context_manager = ContextManager()
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
        load_weights_from_safetensors(self.model, self.model_path)
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
        
        self.run_model will return a tensor with shape [number_of_sequences, vocab_size], 
        and then the sampler will pick the next token id for each sequence based on the logits and temperatures.
        
        See Sampler.forward for more details about the sampling strategy.
        '''
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        #token_ids = self.sampler(logits, temperatures, pick_max=True).tolist() if self.rank == 0 else []
        token_ids = self.sampler(logits, temperatures, pick_max=False).tolist() if self.rank == 0 else []
        self.context_manager.clear_default_context()
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
        '''
        Prefill stage is for processing the prompt tokens. In this stage, we will prepare the input_ids
        and positions for all the uncached tokens in the sequences and also prepare the block tables if there are cached blocks.
        
        This function is a little bit complex, let take an example to explain the logic.
        Assume we have 3 sequences with block size 2 and the following token ids (after tokenization):
        seq1: [1,2,3,4,5,6,7]
        seq2: [1,2,0,4,5,6,7]
        seq3: [1,2,3,4,0,6,7,8]
        
        After prefill, all the result should be:
                
        input_ids:[1,2,3,4,5,6,7,0,4,5,6,7,0,6,7,8]
                   |     seq1   |   seq2  |   seq3|
        positions:[0,1,2,3,4,5,6,2,3,4,5,6,4,5,6,7]
                   |   seq1     | seq2    |  seq3 |
        cu_seqlens_q: [0,7, 7+5 = 12, 12 + 4 = 16] => [0,7,12,16] Without cached tokens.
        cu_seqlens_k: [0,7, 7+7 = 14, 14 + 8 = 22] => [0,7,14,22] All tokens including cached and uncached tokens.
        max_seqlen_q: 7 (seq1 has the longest uncached tokens)
        max_seqlen_k: 8 (seq3 has the longest total tokens)
        
        slot_mappings, the value is the physical address of kv_cache in GPU for each token in the input_ids.
        
        Finally, we will create a context for this prefill stage and store it in the context manager, and the model runner will use this context when running the model.
        '''
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
                    end = start + Sequence.block_size
                else:
                    end = start + seq.last_block_num_tokens
                
                slot_mappings.extend(list(range(start, end)))

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
        self.context_manager.set_default_context(context)
        return input_ids_tensor, positions_tensor
    
    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Comparing the prefill stage, the decode stage is much simpler,
        as we only need to proceed the last token just added. All the previous tokens have been processed in the prefill stage 
        and the new generated token will be added to the end of the sequence, so we just need to prepare the input_ids and positions for the last token.
        '''
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
            
        block_tables = self.prepare_block_tables(seqs)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mappings_tensor = torch.tensor(slot_mappings, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context = Context(
            is_prefill=False,
            slot_mapping=slot_mappings_tensor,
            context_lens=context_lens_tensor,
            block_tables=block_tables
        )
        self.context_manager.set_default_context(context)
        return input_ids_tensor, positions_tensor

    
    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        '''
        We will make all the block tables the same length to the longest sequence by padding -1 to the end for all the sequences.
        '''
        max_block_table_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_block_table_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        '''
        This function will run the model in either eager mode or CUDA graph mode depending on the input sequence length and the enforced_eager flag.
        
        After run the model, it will return the logits for the last token in the input_ids, which will be used for sampling the next token.
        The logits is a tensor with shape [number_of_sequences, vocab_size].
        For each sequence, vocab_size of elements mean the possibility of next token id, in the follow step, the sampler will pick one token id based on it strategy.
        '''
        if is_prefill or self.enforced_eager or input_ids.size(0) > self.CUDA_GRAPH_CAPTURE_SIZE:
            self.logger.info(f"Running model eagerly on rank {self.rank} due to enforced eager mode with input_ids shape: {input_ids.shape} and positions shape: {positions.shape}")
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            self.logger.info(f"Running model on rank {self.rank} with input_ids shape: {input_ids.shape} and positions shape: {positions.shape} in CUDA graph mode.")
            bs = input_ids.size(0)
            context = self.context_manager.get_default_context()
            graph = self.graphs[self.get_good_graph_bs(bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs].copy_(input_ids)
            graph_vars["positions"][:bs].copy_(positions)
            graph_vars["slot_mappings"].fill_(-1)
            graph_vars["slot_mappings"][:bs].copy_(context.slot_mapping)
            graph_vars["context_lens"][:bs].zero_()
            graph_vars["context_lens"][:bs].copy_(context.context_lens)
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)].copy_(context.block_tables)
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])
    
    def get_good_graph_bs(self, bs:int) -> int:
        '''
        This method is used to get the good batch size for CUDA graph replay.
        It will return the smallest batch size in self.graph_bs that is greater than or equal to the input batch size.
        '''
        # O(N)
        #return next(x for x in self.graph_bs if x >= bs)

        # O(log(N))
        # idx = bisect.bisect_left(self.graph_bs, bs)
        # target_bs = self.graph_bs[idx]
        # return target_bs

        # O(1)
        if bs <= 8:
            # Optimized for 1,2,4,8
            target_bs = 1 << (bs - 1).bit_length()
        else:
            # Optimized for 16,32,48,...,512
            target_bs = ((bs + 15) // 16) * 16
        return target_bs
    
    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        '''
        Just prepare the temperatures for sampling, which is only used in rank 0 as it is responsible for sampling and generating the token ids,
        and then broadcast to other ranks if needed.
        '''
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

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
        '''
        It will preserve the kv cache for each layer in a big chunk of memory with shape:
        [2 (key and value), num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim],
        and then assign the corresponding chunk to each attention layer's k_cache and v_cache.
        This cache is managed by the block manager, which will allocate/free blocks for each sequence
        and the attention layer will write the key/value to the corresponding block according to the slot mapping in the context.
        '''
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()['allocated_bytes.all.peak']
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

    @torch.inference_mode()
    def capture_cuda_graphs(self):
        '''
        Here we leverage cudaStream.capture to capture the decode stage of the model with different batch sizes, and store the graphs in a dictionary for later use.
        cudastream.capture always involve following steps:
        1. Create a cuda stream and initialized the variables.
        2. Warmup the stream to make sure everything is ready.
        3. Capture the graph by launching kernels on the stream.
        4. Synchronize and store the graph for later replay.
        
        In this method, all size of graphs shared the same input and output tensors,
        so it starts from the largest batch size to capture the graph, and then reuse the same graph pool for smaller batch size graphs to save memory and speed up the capture process.
        '''
        hf_config = self.config.hf_config
        # max batch size for CUDA graph capture, 512 is just an experience number for RTX3090
        max_bs = min(self.config.max_num_seqs, self.CUDA_GRAPH_CAPTURE_SIZE)
        max_num_blocks = (self.config.max_model_length + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mappings = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1,2,4,8] + list(range(16, max_bs+1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            # self.logger.info(f"Capturing CUDA graph for batch size {bs} on rank {self.rank}")
            graph = torch.cuda.CUDAGraph()

            context = Context(
                is_prefill=False, # Only decode stage uses CUDA graph
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mappings[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs]
            )
            self.context_manager.set_default_context(context)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # graph_pool is None for the largest batch size graph, so it will create a new pool.
            # and put it into graph.pool
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # If it is the first time creating the graph pool with largest batch size
            # assign it and will reuse for the following smaller batch size graphs.
            if self.graph_pool is None:
                # self.logger.info(f"Creating CUDA graph pool on rank {self.rank} as it does not exist.")
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            self.context_manager.clear_default_context()

        self.graph_vars = {"input_ids": input_ids, "positions": positions,
                           "slot_mappings": slot_mappings, "context_lens": context_lens,
                           "block_tables": block_tables, "outputs": outputs}


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
        '''
        Slave rank will wait for the master to write commands to shared memory and execute them.
        If exit command is received, it will break the loop and exit.
        '''
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