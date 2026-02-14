import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from utils import Context, ContextManager

class VocabParallelEmbeddingHead(nn.Module):
    '''
    这个类实现了一个词汇表并行的嵌入层，用于在分布式训练中将词汇表划分到多个设备上，从而提高内存效率和计算效率。
    就是把一个含有token ids的张量，映射到对应的词向量表示上。

    比如:
    weight = 
        [[0.11, 0.12, 0.13, 0.14],
         [0.21, 0.22, 0.23, 0.24],
         [0.31, 0.32, 0.33, 0.34],
         [0.41, 0.42, 0.43, 0.44],
         [0.51, 0.52, 0.53, 0.54],
         [0.61, 0.62, 0.63, 0.64]]

    x = [2,5]

    输出:
    y = [[0.31, 0.32, 0.33, 0.34],
         [0.61, 0.62, 0.63, 0.64]]

    在多个GPU的情况，会把weight拆分，比如2个GPU:
    GPU 0:
        [[0.11, 0.12, 0.13, 0.14],
         [0.21, 0.22, 0.23, 0.24],
         [0.31, 0.32, 0.33, 0.34]]

    GPU 1:
        [[0.41, 0.42, 0.43, 0.44],
         [0.51, 0.52, 0.53, 0.54],
         [0.61, 0.62, 0.63, 0.64]]

    具体的运算过程，我在forward函数中有详细注释。
    '''
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        # Tensor Parallel Rank, usually rank 0 is the master node.
        self.tp_rank = dist.get_rank()
        # Tensor Parallel Size
        self.tp_world_size = dist.get_world_size()
        self.num_embeddings = vocab_size
        self.embedding_dim = hidden_size
        self.embeddings_per_partition = self.num_embeddings // self.tp_world_size
        self.vocab_start_index = self.tp_rank * self.embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.embeddings_per_partition, self.embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, weight: nn.Parameter, loaded_weight: torch.Tensor):
        assert loaded_weight.size(0) == self.embeddings_per_partition, f"Weight size {loaded_weight.size(0)} does not match embeddings per partition {self.embeddings_per_partition}."
        data = weight.data
        data_size = data.size(0)
        start_idx = self.tp_rank * data_size
        picked_weight = loaded_weight.narrow(0, start_idx, data_size)
        data.copy_(picked_weight)

       
    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        '''
        如果只有一个GPU，那么直接调用F.embedding函数进行嵌入查找。
        如果有多个GPU，那么需要先创建一个掩码(mask)，用于标识输入中哪些token属于当前GPU的词汇范围，然后进行嵌入查找，最后通过all_reduce操作将结果汇总。

        我们假设 x = [2,5], weight 在类的注释部分已经给出。
        假设有2个GPU: self.tp_world_size = 2
        
        '''
        if self.tp_world_size > 1:
            mask = (x >= self.vocab_start_index) & (x < self.vocab_end_index)
            '''
             GPU 0: mask = [True, False]
             GPU 1: mask = [False, True]
            '''
            x = mask * (x - self.vocab_start_index)

            '''
            GPU 0: x = [2, 0]
            GPU 1: x = [0, 2]
            '''

        y = F.embedding(x, self.weight)
        '''
        GPU 0: y = 
        [[0.31, 0.32, 0.33, 0.34],
         [0.11, 0.12, 0.13, 0.14]]

        GPU 1: y =
        [[0.41, 0.42, 0.43, 0.44],
         [0.61, 0.62, 0.63, 0.64]]
        '''
        if self.tp_world_size > 1:
            unsqueezeed_mask = mask.unsqueeze(-1)
            '''
            GPU 0: unsqueezeed_mask =
            [[True],
             [False]]
            
            GPU 1: unsqueezeed_mask =
            [[False],
             [True]]
            '''
            y = unsqueezeed_mask * y
            '''
            GPU 0: y =
            [[0.31, 0.32, 0.33, 0.34],
             [0.00, 0.00, 0.00, 0.00]]
            
            GPU 1: y =
            [[0.00, 0.00, 0.00, 0.00],
             [0.61, 0.62, 0.63, 0.64]]
            ''' 
            dist.all_reduce(y, op=dist.ReduceOp.SUM)

        '''
        最终输出结果:
        y =
        [[0.31, 0.32, 0.33, 0.34],
         [0.61, 0.62, 0.63, 0.64]]
        这正是我们想要的结果。
        '''
        return y
    

class ParallelLMHead(VocabParallelEmbeddingHead):
    '''
    Parallel Language Model Head。
    '''
    def __init__(self, vocab_size: int, hidden_size: int, bias: bool = False):
        assert not bias, "Bias is not supported in ParallelLMHead."
        super().__init__(vocab_size, hidden_size)
        self.context_manager = ContextManager()

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        '''
        把model计算得到的hidden states映射到vocab size维度的logits上，供后续的softmax和采样使用。
        如果是在prefill阶段：
            x.shape = [context.cu_seqlen_q[-1], hidden_size]
        如果是在decode阶段：
            x.shape = [len(seqs), hidden_size]
        '''
        context = self.context_manager.get_default_context()
        logits:torch.Tensor | None = None
        if context is not None and context.is_prefill:
            '''
            在预填充阶段(prefill stage)，我们只需要计算当前步的token对应的logits，而不需要计算整个序列的logits。
            比如 context.cu_seqlens_q = [  0,  49,  99, 146]，表示当前步的token是第48个,第98个token和第145个token，
            那么我们只需要计算这三个token对应的logits，而不需要计算第0到第47个或者其它中间的token的logits
            '''
            last_indices = context.cu_seqlens_q[1:] -1
            '''
            last_indices = [48, 98, 145]
            '''
            x = x[last_indices].contiguous()
            '''
            x.shape = [len(seqs), hidden_size]
            '''

        '''
           logits = x[len(seqs), hidden_size] @ self.weight[vocab_size, hidden_size].T[hidden_size, vocab_size] = [len(seqs), vocab_size]
           
           F.linear(x,w,b)的计算过程是: output = x @ w.T + b
           这里我们不使用bias，所以b为None。
        '''
        logits = F.linear(x, self.weight)

        if self.tp_world_size > 1:
            '''
            如果在多块GPU中，每块GPU计算得到的logits只是整个词汇表的一部分，我们需要通过all_gather操作将所有GPU的logits汇总起来，得到完整的logits。
            '''
            logits_list = [torch.empty_like(logits) for _ in range(self.tp_world_size)] if self.tp_rank == 0 else None
            dist.all_gather(logits_list, logits)
            logits = torch.cat(logits_list, dim=-1) if self.tp_rank == 0 else None

        return logits