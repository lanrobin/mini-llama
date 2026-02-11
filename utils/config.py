import os
from dataclasses import dataclass
from transformers import LlamaConfig
from . import constants as CONST

@dataclass
class Config:
    model_path:str
    max_num_batched_tokens:int = 16384
    max_num_seqs:int = CONST.MAX_NUM_SEQS
    max_model_length:int = 4096
    gpu_memory_utilization:float = 0.99
    tensor_parallel_size:int = 1
    enforce_eager:bool = False
    hf_config:LlamaConfig | None = None
    eos_token_id:int = -1
    kvcache_block_size:int = CONST.DEFAULT_KV_CACHE_BLOCK_SIZE
    num_kvcache_blocks:int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model_path), f"Model path {self.model_path} does not exist or is not a directory."
        assert self.kvcache_block_size % CONST.MIN_KV_CACHE_BLOCK_SIZE == 0, f"kvcache_block_size must be a multiple of {CONST.MIN_KV_CACHE_BLOCK_SIZE}."
        assert 1<= self.tensor_parallel_size <= 8, "tensor_parallel_size must be between 1 and 8."
        self.hf_config = LlamaConfig.from_pretrained(self.model_path)
        self.max_model_length = min(self.max_model_length, self.hf_config.max_position_embeddings or self.max_model_length)
        assert self.max_num_batched_tokens >= self.max_model_length, "max_num_batched_tokens must be at least max_model_length."