import torch
from torch import nn
import triton
import triton.language as tl

from utils import Context, ContextManager, Logger

@triton.jit
def store_kvcache_kernel(key_ptr,key_stride, value_ptr, value_stride, k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        Logger().logger.error(f"Invalid slot mapping for index {idx}. Slot mapping must be non-negative.")
        return
    key_offset = idx * key_stride + tl.arrange(0, D)
    value_offset = idx * value_stride + tl.arrange(0, D)
    key = tl.load(key_ptr + key_offset)
    value = tl.load(value_ptr + value_offset)
    cache_offset = slot * D + tl.arrange(0, D)
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)
    
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1, "Key and value tensors must be contiguous in the last dimension"
    assert key.stride(1) == head_dim and value.stride(1) == head_dim, "Key and value tensors must have the correct stride for the last dimension"
    assert v_cache.stride(1) == D and k_cache.stride(1) == D, "Cache tensors must have the correct stride for the last dimension"
    assert slot_mapping.numel() == N, "Slot mapping must have the same number of elements as the batch size"
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        self.context_manager = ContextManager()
        self.logger = Logger()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = self.context_manager.get_default_context()
        k_cache = self.k_cache
        v_cache = self.v_cache
        if k_cache.numel() > 0 and v_cache.numel() > 0:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                #self.logger.info("Using block tables for attention computation.")
                # Implement block table based attention computation here
                k,v = k_cache, v_cache
            o = torch.empty_like(q)
            self.logger.error("Prefill mode with block tables is not implemented yet.")
        else: # decode mode
            o = torch.empty_like(q)
            self.logger.error("Decode mode attention computation is not implemented yet.")
            
        return o