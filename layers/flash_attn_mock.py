"""
flash_attn_varlen_func 和 flash_attn_with_kvcache 的纯 PyTorch 实现。

速度远不及真正的 Flash Attention，但代码逻辑清晰，便于学习理解。
所有参数名和语义都与 flash_attn 库保持一致。
"""

import math
from typing import Optional, Union

import torch


# ────────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────────

def _expand_kv_heads(k: torch.Tensor, v: torch.Tensor, num_heads_q: int):
    """
    处理 GQA / MQA：将 K/V 的 head 维度复制扩展，使之与 Q 的 head 数量一致。

    如果 num_kv_heads < num_heads_q，则每个 KV head 被重复
    (num_heads_q // num_kv_heads) 次。
    """
    num_kv_heads = k.shape[-2]
    if num_kv_heads == num_heads_q:
        return k, v
    assert num_heads_q % num_kv_heads == 0, \
        f"num_heads_q ({num_heads_q}) 必须能被 num_kv_heads ({num_kv_heads}) 整除"
    repeat = num_heads_q // num_kv_heads
    # k: (..., num_kv_heads, head_dim) -> (..., num_heads_q, head_dim)
    k = k.repeat_interleave(repeat, dim=-2)
    v = v.repeat_interleave(repeat, dim=-2)
    return k, v


def _naive_attention(q, k, v, causal_mask=None, softmax_scale=None):
    """
    最基本的 Scaled Dot-Product Attention。

    参数:
        q: (seq_q, num_heads, head_dim)
        k: (seq_k, num_heads, head_dim)
        v: (seq_k, num_heads, head_dim)
        causal_mask: (seq_q, seq_k) bool，True 表示该位置被 mask（不参与 attention）
        softmax_scale: 缩放因子，默认 1/sqrt(head_dim)

    返回:
        output: (seq_q, num_heads, head_dim)
    """
    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # (num_heads, seq_q, head_dim) @ (num_heads, head_dim, seq_k) -> (num_heads, seq_q, seq_k)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * softmax_scale

    if causal_mask is not None:
        # causal_mask: (seq_q, seq_k) -> broadcast 到 (num_heads, seq_q, seq_k)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    # 如果某一行全是 -inf（全被 mask），softmax 会产生 nan，替换成 0
    attn_weights = attn_weights.nan_to_num(0.0)

    # (num_heads, seq_q, seq_k) @ (num_heads, seq_k, head_dim) -> (num_heads, seq_q, head_dim)
    out = torch.einsum("hqk,khd->qhd", attn_weights, v.float())
    return out.to(q.dtype)


def _build_causal_mask(seq_q: int, seq_k: int, device: torch.device) -> torch.Tensor:
    """
    构建 bottom-right 对齐的 causal mask。

    Flash Attention 的 causal mask 规则：
        对于 query 位置 i (0-indexed) 和 key 位置 j：
            保留 (False): j - i <= seq_k - seq_q   (即 i + seq_k - seq_q >= j)
            屏蔽 (True):  j - i >  seq_k - seq_q

    示例 (seq_q=2, seq_k=5, 0=保留, 1=屏蔽):
        0 0 0 0 1
        0 0 0 0 0

    示例 (seq_q=5, seq_k=2):
        1 1
        1 1
        1 1
        0 1
        0 0

    返回:
        mask: (seq_q, seq_k) bool, True = 被屏蔽
    """
    # i: query 位置, j: key 位置
    i = torch.arange(seq_q, device=device).unsqueeze(1)  # (seq_q, 1)
    j = torch.arange(seq_k, device=device).unsqueeze(0)  # (1, seq_k)
    # bottom-right 对齐: 允许 j <= i + (seq_k - seq_q)
    # mask就是一个由 True/False 组成的矩阵，True 表示被屏蔽，False 表示保留
    mask = j > (i + seq_k - seq_q)
    return mask


# ────────────────────────────────────────────────────────────────────
# 从 paged KV cache 中提取连续的 K/V（用于 block_table 场景）
# ────────────────────────────────────────────────────────────────────

def _gather_paged_kv(kv_cache: torch.Tensor, block_table: torch.Tensor, seqlen: int):
    """
    从 paged KV cache 中按 block_table 收集连续的 KV 序列。

    参数:
        kv_cache:    (num_blocks, block_size, num_kv_heads, head_dim) — 物理 block 池
        block_table: (max_num_blocks_per_seq,) int — 逻辑 block -> 物理 block 索引
        seqlen:      int — 实际序列长度（只取前 seqlen 个 token）

    返回:
        kv: (seqlen, num_kv_heads, head_dim) — 连续排列的 KV
    """
    block_size = kv_cache.shape[1]
    num_blocks_needed = math.ceil(seqlen / block_size)
    
    # 收集所需的物理 block
    # block_table[:num_blocks_needed] -> 物理 block 索引列表
    physical_blocks = block_table[:num_blocks_needed]  # (num_blocks_needed,)
    
    # 取出对应的 block 数据: (num_blocks_needed, block_size, num_kv_heads, head_dim)
    gathered = kv_cache[physical_blocks]
    
    # reshape 成连续序列: (num_blocks_needed * block_size, num_kv_heads, head_dim)
    gathered = gathered.reshape(-1, *kv_cache.shape[2:])
    
    # 截取实际长度
    return gathered[:seqlen]


# ────────────────────────────────────────────────────────────────────
# flash_attn_varlen_func 的纯 PyTorch 实现
# ────────────────────────────────────────────────────────────────────

def flash_attn_varlen_func(
    q: torch.Tensor,                          # (total_q, num_heads, head_dim)
    k: torch.Tensor,                          # (total_k, num_kv_heads, head_dim) 或 paged
    v: torch.Tensor,                          # 同 k
    cu_seqlens_q: torch.Tensor,               # (batch_size + 1,) int32 — query 的累计序列长度
    cu_seqlens_k: torch.Tensor,               # (batch_size + 1,) int32 — key 的累计序列长度
    max_seqlen_q: int,                        # batch 中最大 query 序列长度（此实现中不使用，仅保持接口一致）
    max_seqlen_k: int,                        # batch 中最大 key 序列长度（同上）
    dropout_p: float = 0.0,                   # dropout 概率（此简易实现中忽略）
    softmax_scale: Optional[float] = None,    # 缩放因子，默认 1/sqrt(head_dim)
    causal: bool = False,                     # 是否使用 causal mask
    window_size=(-1, -1),                     # 滑动窗口（此实现中忽略）
    softcap: float = 0.0,                     # softcap（此实现中忽略）
    alibi_slopes=None,                        # ALiBi（此实现中忽略）
    deterministic: bool = False,              # 确定性模式（此实现天然确定性）
    return_attn_probs: bool = False,          # 是否返回 attn probs（此实现中忽略）
    block_table: Optional[torch.Tensor] = None,  # (batch_size, max_num_blocks_per_seq) — paged attention
):
    """
    可变长度序列的 attention（纯 PyTorch 实现）。

    核心思路：
        1. 多个序列的 Q/K/V 被拼接成一个扁平张量，cu_seqlens 标记了每个序列的边界。
        2. 遍历 batch 中的每个序列，分别计算独立的注意力（序列之间没有交叉注意力）。
        3. 如果提供了 block_table，则 K/V 以 paged 方式存储（4D），需要按 block_table 收集。

    参数说明：
        q:             (total_q, num_heads, head_dim)
                       所有序列的 query 拼接在一起。total_q = sum(各序列 query 长度)
        k:             不使用 paged: (total_k, num_kv_heads, head_dim)
                       使用 paged:   (num_blocks, block_size, num_kv_heads, head_dim)
        v:             同 k
        cu_seqlens_q:  (batch_size+1,) int32, 累计 query 长度。
                       例如 batch 有 3 个序列，长度 [3,5,2]，则 cu_seqlens_q = [0,3,8,10]
                       第 i 个序列的 query 为 q[cu_seqlens_q[i] : cu_seqlens_q[i+1]]
        cu_seqlens_k:  (batch_size+1,) int32, 累计 key 长度，含义同上
        causal:        如果为 True，使用 bottom-right 对齐的 causal mask
        block_table:   (batch_size, max_num_blocks) 或 None
                       如果提供，K/V 以 paged 方式存储

    返回:
        output: (total_q, num_heads, head_dim) — 与 q 相同形状
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads_q = q.shape[1]
    head_dim = q.shape[2]
    is_paged = (block_table is not None)

    # 输出张量，和 q 相同 shape
    output = torch.zeros_like(q)

    # 逐个序列处理（真正的 Flash Attention 当然是并行的，这里为了清晰用循环）
    for i in range(batch_size):
        # 取出第 i 个序列的 Q
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        q_i = q[q_start:q_end]       # (seqlen_q_i, num_heads, head_dim)
        seqlen_q_i = q_end - q_start

        # 取出第 i 个序列的 K/V
        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i + 1].item()
        seqlen_k_i = k_end - k_start

        if is_paged:
            # paged 模式: k/v shape = (num_blocks, block_size, num_kv_heads, head_dim)
            # 需要按 block_table[i] 收集出连续的序列
            k_i = _gather_paged_kv(k, block_table[i], seqlen_k_i)
            v_i = _gather_paged_kv(v, block_table[i], seqlen_k_i)
        else:
            # 非 paged 模式: 直接按 cu_seqlens_k 切片
            k_i = k[k_start:k_end]   # (seqlen_k_i, num_kv_heads, head_dim)
            v_i = v[k_start:k_end]

        # 处理 GQA/MQA: 扩展 K/V 的 head 数量以匹配 Q
        k_i, v_i = _expand_kv_heads(k_i, v_i, num_heads_q)

        # 构建 causal mask
        mask = _build_causal_mask(seqlen_q_i, seqlen_k_i, q.device) if causal else None

        # 计算 attention
        out_i = _naive_attention(q_i, k_i, v_i, causal_mask=mask, softmax_scale=softmax_scale)
        output[q_start:q_end] = out_i

    return output


# ────────────────────────────────────────────────────────────────────
# flash_attn_with_kvcache 的纯 PyTorch 实现
# ────────────────────────────────────────────────────────────────────

def flash_attn_with_kvcache(
    q: torch.Tensor,                                                        # (batch_size, seqlen_q, num_heads, head_dim)
    k_cache: torch.Tensor,                                                  # 非 paged: (batch_size, seqlen_cache, num_kv_heads, head_dim)
                                                                            # paged:   (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,                                                  # 同 k_cache
    k: Optional[torch.Tensor] = None,                                      # (batch_size, seqlen_new, num_kv_heads, head_dim) 新的 key
    v: Optional[torch.Tensor] = None,                                      # 同上，新的 value
    rotary_cos: Optional[torch.Tensor] = None,                             # 忽略
    rotary_sin: Optional[torch.Tensor] = None,                             # 忽略
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,              # int 或 (batch_size,) — 每个序列 cache 中已有的 token 数
    cache_batch_idx: Optional[torch.Tensor] = None,                        # 忽略
    cache_leftpad: Optional[torch.Tensor] = None,                          # 忽略
    block_table: Optional[torch.Tensor] = None,                            # (batch_size, max_num_blocks_per_seq)
    softmax_scale: Optional[float] = None,                                 # 缩放因子
    causal: bool = False,                                                  # 是否使用 causal mask
    window_size=(-1, -1),                                                  # 忽略
    softcap: float = 0.0,                                                  # 忽略
    rotary_interleaved: bool = True,                                       # 忽略
    alibi_slopes=None,                                                     # 忽略
    num_splits: int = 0,                                                   # 忽略
    return_softmax_lse: bool = False,                                      # 忽略
):
    """
    带 KV Cache 的 attention（纯 PyTorch 实现），主要用于推理阶段的 decode。

    典型使用场景：
        - decode 阶段每步只有 1 个新 token (seqlen_q=1)
        - 之前的 K/V 已经存在 cache 里
        - 新的 K/V（如果有）被追加到 cache，然后对整个历史做 attention

    核心流程：
        1. 如果提供了新的 k/v，将其写入 cache（从 cache_seqlens 位置开始）
        2. 根据 cache_seqlens（+ seqlen_new）确定每个序列的实际 KV 长度
        3. 如果使用 paged attention (block_table)，按 block_table 从 cache 中收集 KV
        4. 对每个序列独立计算 attention

    参数说明：
        q:             (batch_size, seqlen_q, num_heads, head_dim)
                       decode 时通常 seqlen_q=1
        k_cache:       非 paged: (batch_size, seqlen_cache, num_kv_heads, head_dim)
                       paged:   (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache:       同 k_cache
        k/v:           可选的新 K/V，会被追加到 cache 中
        cache_seqlens: 每个序列 cache 中已有的 token 数，
                       如果是 int 则 broadcast 到所有序列
        block_table:   paged attention 的 block 映射表
        causal:        是否应用 causal mask

    返回:
        output: (batch_size, seqlen_q, num_heads, head_dim)
    """
    batch_size, seqlen_q, num_heads_q, head_dim = q.shape
    is_paged = (block_table is not None)

    # 处理 cache_seqlens
    if cache_seqlens is None:
        if is_paged:
            cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
        else:
            # 非 paged 时，默认整个 cache 都有效
            cache_seqlens = torch.full((batch_size,), k_cache.shape[1], dtype=torch.int32, device=q.device)
    elif isinstance(cache_seqlens, int):
        cache_seqlens = torch.full((batch_size,), cache_seqlens, dtype=torch.int32, device=q.device)

    seqlen_new = k.shape[1] if k is not None else 0

    # ── 步骤 1: 将新的 k/v 写入 cache（如果有） ──
    if k is not None and v is not None and seqlen_new > 0:
        for b in range(batch_size):
            start = cache_seqlens[b].item()
            end = start + seqlen_new
            if is_paged:
                # paged 模式: 需要逐个 token 找到对应的物理位置
                block_size = k_cache.shape[1]
                for t in range(seqlen_new):
                    pos = start + t
                    block_idx = block_table[b, pos // block_size].item()
                    block_offset = pos % block_size
                    k_cache[block_idx, block_offset] = k[b, t]
                    v_cache[block_idx, block_offset] = v[b, t]
            else:
                # 非 paged 模式: 直接按位置写入
                k_cache[b, start:end] = k[b]
                v_cache[b, start:end] = v[b]

    # ── 步骤 2: 计算每个序列的实际 KV 长度 ──
    # 实际长度 = cache 中已有长度 + 新追加的长度
    total_seqlens_k = cache_seqlens + seqlen_new  # (batch_size,)

    # ── 步骤 3: 对每个序列独立计算 attention ──
    output = torch.zeros_like(q)

    for b in range(batch_size):
        seqlen_k_b = total_seqlens_k[b].item()
        q_b = q[b]  # (seqlen_q, num_heads, head_dim)

        if is_paged:
            # paged 模式: 按 block_table 从 cache block 池中收集 KV
            k_b = _gather_paged_kv(k_cache, block_table[b], seqlen_k_b)
            v_b = _gather_paged_kv(v_cache, block_table[b], seqlen_k_b)
        else:
            # 非 paged 模式: 直接切片
            k_b = k_cache[b, :seqlen_k_b]  # (seqlen_k_b, num_kv_heads, head_dim)
            v_b = v_cache[b, :seqlen_k_b]

        # 处理 GQA/MQA
        k_b, v_b = _expand_kv_heads(k_b, v_b, num_heads_q)

        # 构建 causal mask
        mask = _build_causal_mask(seqlen_q, seqlen_k_b, q.device) if causal else None

        # 计算 attention
        out_b = _naive_attention(q_b, k_b, v_b, causal_mask=mask, softmax_scale=softmax_scale)
        output[b] = out_b

    return output
