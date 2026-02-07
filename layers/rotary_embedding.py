from functools import lru_cache
import torch
import torch.nn as nn

def apply_rotary_embedding(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_size:int, rotary_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        assert rotary_dim == head_size, "Currently only full rotary embedding is supported."
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        
        ''' This fixed: Exception has occurred: RuntimeError
        The size of tensor a (24) must match the size of tensor b (16384) at non-singleton dimension 1.
        As when multiplying: x1.shape=[16384, 24, 64] and cos.shape=[16384, 24].
        After unsqueeze, cos.shape=[16384, 1, 24], which can be broadcasted to x1.shape. '''
        cos.unsqueeze_(1)
        sin.unsqueeze_(1)
        
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
       cos = self.cos[positions]
       sin = self.sin[positions]
       q = apply_rotary_embedding(query, sin, cos)
       k = apply_rotary_embedding(key, sin, cos)
       return q, k

@lru_cache(maxsize=1)  
def get_rope(head_size: int, rotary_dime: int, max_position: int, base:float, rope_scaling:tuple | None = None):
    assert rope_scaling is not None, "Tuple RoPE scaling is not supported yet."
    rotary_embedding = RotaryEmbedding(head_size, rotary_dime, max_position_embeddings=max_position, base=base)
    return rotary_embedding