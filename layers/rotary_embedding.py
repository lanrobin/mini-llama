from functools import lru_cache
import torch
import torch.nn as nn

def apply_rotary_embedding(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_size:int, rotary_dim: int, max_position: int = 2048, base: int = 10000):
        super().__init__()
        assert rotary_dim == head_size, "Currently only full rotary embedding is supported."
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
       cos = self.cos[positions]
       sin = self.sin[positions]
       q = apply_rotary_embedding(query, sin, cos)
       k = apply_rotary_embedding(key, sin, cos)
       return q, k


