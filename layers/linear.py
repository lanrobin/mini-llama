import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from abc import abstractmethod, ABC
from utils import Logger


class LinearBase(nn.Module, ABC):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim:int | None = None):
        super().__init__()
        self.tp_size = dist.get_world_size() 
        self.tp_rank = dist.get_rank()
        self.tp_dim = tp_dim
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None)
        self.logger = Logger()

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        return output



    @staticmethod
    def divide(numerator, denominator) -> int:
        assert numerator % denominator == 0 or denominator == 0, "Numerator must be divisible by denominator or denominator is zero"
        return numerator // denominator


class ReplicatedLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias, tp_dim=None)
        # In replicated linear, all ranks have the full weight matrix
        self.logger.info(f"Initialized ReplicatedLinear with input_size={input_size}, output_size={output_size}, bias={bias}")

    def weight_loader(self, param: nn.Parameter, weight_tensor: torch.Tensor):
        param.data.copy_(weight_tensor)


class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        per_partition_output_size = LinearBase.divide(output_size, dist.get_world_size())
        super().__init__(input_size, per_partition_output_size, bias, tp_dim=0)
        self.logger.info(f"Initialized ColumnParallelLinear with input_size={input_size}, output_size={output_size}, bias={bias}")

    def weight_loader(self, param: nn.Parameter, weight_tensor: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_index = self.tp_rank * shard_size
        column_data = weight_tensor.narrow(self.tp_dim, start_index, shard_size)
        param.data.copy_(column_data)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        sum_output_size = sum(output_sizes)
        super().__init__(input_size, sum_output_size, bias)
        self.logger.info(f"Initialized MergedColumnParallelLinear with input_size={input_size}, output_sizes={output_sizes}, bias={bias}")

    def weight_loader(self, param: nn.Parameter, weight_tensor: torch.Tensor, loaded_shard_id:int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        pd = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        merged_data = weight_tensor.chunk(self.tp_size,self.tp_dim)[self.tp_rank]
        pd.copy_(merged_data)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size: int, head_size: int, total_num_heads:int, total_num_kv_heads:int, bias: bool = False):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = LinearBase.divide(total_num_heads, tp_size)
        self.num_kv_heads = LinearBase.divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)
        self.logger.info(f"Initialized QKVParallelLinear with hidden_size={hidden_size}, head_size={head_size}, total_num_heads={total_num_heads}, total_num_kv_heads={total_num_kv_heads}, bias={bias}")
        
    def weight_loader(self, param: nn.Parameter, weight_tensor: torch.Tensor, loaded_shard_id:str):
        assert loaded_shard_id in ["q","k","v"], "loaded_shard_id must be one of 'q_proj', 'k_proj' or 'v_proj'"
        if loaded_shard_id == 'q':
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == 'k':
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else: # loaded_shard_id == 'v':
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        
        pd = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        merged_data = weight_tensor.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        pd.copy_(merged_data)
       

class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        per_partition_input_size = LinearBase.divide(input_size, tp_size)
        super().__init__(per_partition_input_size, output_size, bias, tp_dim=1)
        self.logger.info(f"Initialized RowParallelLinear with input_size={input_size}, output_size={output_size}, bias={bias}")

    def weight_loader(self, param: nn.Parameter, weight_tensor: torch.Tensor):
        shard_size = param.data.size(self.tp_dim)
        start_index = self.tp_rank * shard_size
        row_data = weight_tensor.narrow(self.tp_dim, start_index, shard_size)
        param.data.copy_(row_data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # All-reduce across the tp_dim (dim=0) after linear transformation
        partial_output = F.linear(input, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(partial_output, op=dist.ReduceOp.SUM)
        return partial_output