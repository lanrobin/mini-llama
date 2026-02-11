import torch
from torch import nn
import triton
import triton.language as tl

from utils import Context, ContextManager, Logger

# 原始高性能实现（需要安装 flash-attn）:
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# 纯 PyTorch 实现，便于理解逻辑:
#from layers.flash_attn_mock import flash_attn_varlen_func, flash_attn_with_kvcache

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

    
def store_kvcache_slow(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    store_kvcache 的纯 PyTorch 实现，用于学习理解 Triton 版本的逻辑。
    
    功能：将当前步的 key/value 按照 slot_mapping 指定的位置写入 KV Cache。
    
    参数:
        key:          [N, num_heads, head_dim] — 当前 N 个 token 的 key
        value:        [N, num_heads, head_dim] — 当前 N 个 token 的 value
        k_cache:      [num_blocks * block_size, num_heads * head_dim] — 全局 key cache（扁平化存储）
        v_cache:      [num_blocks * block_size, num_heads * head_dim] — 全局 value cache（扁平化存储）
        slot_mapping:  [N] — 每个 token 在 cache 中对应的 slot 索引，-1 表示跳过
    
    对应 Triton 版本的逻辑：
        - Triton 对每个 token (idx=0..N-1) 启动一个独立的 program（并行）
        - 每个 program 读取 slot_mapping[idx]，若为 -1 则跳过
        - 否则将 key[idx] 和 value[idx]（reshape 成 [num_heads * head_dim]）写入 cache[slot]
        - 这里用 Python for 循环逐个处理，等价但更慢
    """
    N, num_heads, head_dim = key.shape

    # k_cache 可能是 4D: [num_blocks, block_size, num_heads, head_dim]
    # slot 是全局索引，需要拆分为 (block_idx, block_offset)
    if k_cache.dim() == 4:
        block_size = k_cache.shape[1]
    else:
        block_size = None

    for idx in range(N):
        slot = slot_mapping[idx].item()
        if slot == -1:
            # Triton 版本中: if slot == -1: return（跳过该 token）
            continue
        if block_size is not None:
            # 4D cache: 将全局 slot 拆分为 block 索引和 block 内偏移
            block_idx = slot // block_size
            block_offset = slot % block_size
            k_cache[block_idx, block_offset] = key[idx]
            v_cache[block_idx, block_offset] = value[idx]
        else:
            # 2D cache: 直接用 slot 索引
            k_cache[slot] = key[idx]
            v_cache[slot] = value[idx]

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
            # self.logger.info(f"slot_mapping max:{context.slot_mapping.max().item()} min:{context.slot_mapping.min().item()}, k_cache shape: {k_cache.shape}, v_cache shape: {v_cache.shape}")
            store_kvcache_slow(k, v, k_cache, v_cache, context.slot_mapping)
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