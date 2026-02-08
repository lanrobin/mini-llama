import torch
from torch import nn
import triton
import triton.language as tl

from utils import Context, ContextManager, Logger

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# 在没有安装 flash-attn 的环境中，使用纯 PyTorch 实现的替代函数。这些函数的接口与 flash-attn 的函数保持一致，但性能较差，仅用于测试和调试。
# from .flash_attn_mock import flash_attn_varlen_func_shim as flash_attn_varlen_func
# from .flash_attn_mock import pytorch_paged_attention as flash_attn_with_kvcache

@triton.jit
def store_kvcache_kernel(key_ptr,key_stride, value_ptr, value_stride, k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offset = idx * key_stride + tl.arange(0, D)
    value_offset = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offset)
    value = tl.load(value_ptr + value_offset)
    cache_offset = slot * D + tl.arange(0, D)
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
    
    
def store_kvcache_pytorch(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor
):
    """
    Args:
        key: [Batch_Size, Num_Heads, Head_Dim]
        value: [Batch_Size, Num_Heads, Head_Dim]
        k_cache: [Total_Slots, Num_Heads, Head_Dim] (全局缓存池)
        v_cache: [Total_Slots, Num_Heads, Head_Dim] (全局缓存池)
        slot_mapping: [Batch_Size] (每个请求对应的物理槽位 ID，-1 代表无效/Padding)
    """
    
    # 1. 制作掩码 (Mask)：找出 slot_mapping 中所有不为 -1 的有效索引
    #    slot_mapping == -1 通常表示这个 token 是 padding，不需要存储
    valid_mask = slot_mapping >= 0

    # 2. 获取目标索引 (Destination Indices)
    #    利用掩码取出有效的 slot ID，并确保是 long 类型以用于索引
    valid_slots = slot_mapping[valid_mask].to(torch.long)

    # 3. 获取源数据 (Source Data)
    #    利用掩码取出当前 Batch 中需要写入的那部分 key 和 value
    #    形状从 [N, H, D] 变为 [Valid_N, H, D]
    valid_key = key[valid_mask]
    valid_value = value[valid_mask]

    # 4. 写入缓存 (Store / Scatter)
    #    利用 PyTorch 的高级索引机制，直接将数据写入 k_cache 对应的行
    #    k_cache[valid_slots] 会选中那些特定的行，然后用 valid_key 覆盖它们
    k_cache[valid_slots] = valid_key
    v_cache[valid_slots] = valid_value

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
            # store_kvcache_pytorch(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                #self.logger.info("Using block tables for attention computation.")
                # Implement block table based attention computation here
                k,v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q = context.max_seqlen_q,
                                       cu_seqlens_q = context.cu_seqlens_q,
                                       max_seqlen_k = context.max_seqlen_k,
                                       cu_seqlens_k = context.cu_seqlens_k,
                                       softmax_scale = self.scale,
                                       causal=True,
                                       block_table=context.block_tables)
        else: # decode mode
            unsuqeezed_q = q.unsqueeze(1)  # Add a sequence length dimension of 1 for the current token.
            o = flash_attn_with_kvcache(unsuqeezed_q,
                                        k_cache, v_cache,
                                        cache_seqlens=context.context_lens,
                                        block_table=context.block_tables,
                                        softmax_scale=self.scale,
                                        causal=True)
            
        return o