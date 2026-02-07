import torch
import torch.nn.functional as F

def flash_attn_varlen_func_shim(
    q, k, v, 
    cu_seqlens_q, cu_seqlens_k, 
    max_seqlen_q, max_seqlen_k, 
    softmax_scale=None, 
    causal=True, 
    **kwargs # 吞掉 block_table 等不兼容的参数
) -> torch.Tensor:
    """
    flash_attn_varlen_func 的纯 PyTorch 替代实现。
    
    原理：
    1. 输入是 "Packed" (一维) 的: [Total_Tokens, Heads, Dim]
    2. 内部展开为 "Padded" (Batch) 的: [Batch, Max_Len, Heads, Dim]
    3. 调用标准 SDPA
    4. 压缩回 "Packed" 格式返回
    """
    
    # 1. 准备基础参数
    # cu_seqlens_q 是 [0, len1, len1+len2, ...]
    batch_size = len(cu_seqlens_q) - 1
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    
    # 计算 scale (如果没传，默认是 1 / sqrt(dim))
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    # 2. Unpack & Pad (解包并填充)
    # 申请 Padding 后的容器: [Batch, Max_Seq, Heads, Dim]
    q_padded = torch.zeros(batch_size, max_seqlen_q, num_heads, head_dim, device=q.device, dtype=q.dtype)
    k_padded = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=k.device, dtype=k.dtype)
    v_padded = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=v.device, dtype=v.dtype)
    
    # 循环 batch 填充数据 (这一步有点慢，但逻辑最清晰)
    for i in range(batch_size):
        # 获取当前 seq 在 flat tensor 中的起止位置
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
        
        # 实际长度
        len_q = end_q - start_q
        len_k = end_k - start_k
        
        # 复制数据
        q_padded[i, :len_q] = q[start_q:end_q]
        k_padded[i, :len_k] = k[start_k:end_k]
        v_padded[i, :len_k] = v[start_k:end_k]

    # 3. 处理 GQA (如果 Q 和 KV 头数不一样)
    # PyTorch 2.0+ 的 SDPA 部分版本支持广播，但为了稳妥，手动 repeat
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        k_padded = k_padded.repeat_interleave(n_rep, dim=2)
        v_padded = v_padded.repeat_interleave(n_rep, dim=2)

    # 4. 构建 Attention Mask
    # 我们需要一个 [Batch, 1, Seq_Q, Seq_K] 的 mask
    # True 表示被 Mask (不参与计算), False 表示保留
    # 注意：PyTorch SDPA 的 attn_mask 语义较混乱，这里构建 explicit bias mask 比较稳妥
    
    # 初始化全为 0 (保留)
    mask = torch.zeros(batch_size, 1, max_seqlen_q, max_seqlen_k, device=q.device, dtype=torch.bool)
    
    for i in range(batch_size):
        len_q = cu_seqlens_q[i+1] - cu_seqlens_q[i]
        len_k = cu_seqlens_k[i+1] - cu_seqlens_k[i]
        
        # Mask 掉 Padding 部分 (Query 超出长度的部分其实无所谓，主要是 Key 超出长度的部分)
        mask[i, :, :, len_k:] = True # Key 的 Padding 部分设为 True (遮蔽)
        mask[i, :, len_q:, :] = True # Query 的 Padding 部分设为 True (虽然输出会被丢弃，但为了计算正确)

    # 转换 mask 为 float (-inf)
    # 创建一个极小的数用于 masking
    min_val = torch.finfo(q.dtype).min
    attn_bias = torch.zeros_like(mask, dtype=q.dtype)
    attn_bias.masked_fill_(mask, min_val)

    # 如果是 Causal (自回归)，叠加 Causal Mask
    if causal:
        # 创建上三角 Mask (Triu)
        # [1, 1, Max_Q, Max_K]
        causal_mask = torch.triu(
            torch.ones(max_seqlen_q, max_seqlen_k, device=q.device, dtype=torch.bool), 
            diagonal=1
        )
        attn_bias.masked_fill_(causal_mask, min_val)

    # 5. 计算 Attention (SDPA)
    # 调整维度为 [Batch, Heads, Seq, Dim] 以符合 PyTorch API
    q_in = q_padded.transpose(1, 2)
    k_in = k_padded.transpose(1, 2)
    v_in = v_padded.transpose(1, 2)
    
    output_padded = F.scaled_dot_product_attention(
        q_in, k_in, v_in,
        attn_mask=attn_bias,
        dropout_p=0.0,
        is_causal=False # 我们手动处理了 mask，所以设为 False
    )
    
    # output_padded: [Batch, Heads, Seq, Dim] -> [Batch, Seq, Heads, Dim]
    output_padded = output_padded.transpose(1, 2)

    # 6. Repack (重打包回 Flat Tensor)
    output_flat = torch.empty_like(q)
    
    for i in range(batch_size):
        start, end = cu_seqlens_q[i], cu_seqlens_q[i+1]
        length = end - start
        # 把有效部分切出来放回 flat tensor
        output_flat[start:end] = output_padded[i, :length]

    return output_flat

def pytorch_paged_attention(
    query,          # [Batch, Num_Heads, Head_Dim] (当前的 token)
    k_cache,        # [Num_Blocks, Head_Num, Block_Size, Head_Dim] (物理显存池)
    v_cache,        # [Num_Blocks, Head_Num, Block_Size, Head_Dim]
    block_tables,   # [Batch, Max_Num_Blocks] (页表)
    context_lens,   # [Batch] (每个请求当前的实际长度)
    **kwargs # 吞掉其它不兼容的参数
) -> torch.Tensor:
    """
    用纯 PyTorch 模拟 flash_attn_with_kvcache
    逻辑：从物理块中把数据“搬运”出来，拼成连续的 Tensor，再算 Attention
    """
    batch_size, num_heads, head_dim = query.shape
    
    # -----------------------------------------------------------
    # 步骤 1: 物理 -> 逻辑 的映射 (最核心的教学部分)
    # -----------------------------------------------------------
    # 这是一个比较慢的 Python 实现，但在 nano-vllm 中足够了
    
    max_context_len = context_lens.max().item() # 获取当前 batch 中最长的上下文长度
    
    # 准备一个临时的连续 KV 容器
    # Shape: [Batch, Max_Context_Len, Num_Heads, Head_Dim]
    keys = torch.zeros(batch_size, max_context_len, num_heads, head_dim, device=query.device)
    values = torch.zeros(batch_size, max_context_len, num_heads, head_dim, device=query.device)

    block_size = k_cache.shape[2]
    
    # 循环每个请求 (实际可以用高级索引优化，但循环更易懂)
    for b in range(batch_size):
        current_len = context_lens[b]
        table = block_tables[b] # 获取该请求的页表 [block_id_0, block_id_1, ...]
        
        # 遍历页表，把数据拷出来
        for i, block_idx in enumerate(table):
            if block_idx == -1: break # 遇到空指针停止
            
            # 计算这一块对应逻辑空间的范围
            start = i * block_size
            end = min(start + block_size, current_len)
            real_len_in_block = end - start
            
            # 从物理显存池 (k_cache) 复制到 临时容器 (keys)
            # 物理索引: [block_idx, :, 0:real_len, :]
            keys[b, start:end] = k_cache[block_idx, :, :real_len_in_block, :]
            values[b, start:end] = v_cache[block_idx, :, :real_len_in_block, :]

    # -----------------------------------------------------------
    # 步骤 2: 计算标准 Attention
    # -----------------------------------------------------------
    # query 需要扩充维度以匹配: [Batch, 1, Num_Heads, Head_Dim]
    q = query.unsqueeze(1) 
    
    # 使用 PyTorch 原生 Attention
    # 注意：这里 keys/values 已经是连续的 [Batch, Seq, Head, Dim] 了
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),     # [Batch, Head, 1, Dim]
        keys.transpose(1, 2),  # [Batch, Head, Seq, Dim]
        values.transpose(1, 2),
        is_causal=False # 这里不需要 causal，因为 query 只有 1 个 token，且 keys 包含历史所有
    )
    
    # output: [Batch, Head, 1, Dim] -> Squeeze 回 [Batch, Head, Dim]
    return output.squeeze(2)