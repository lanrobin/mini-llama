import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x.mul_(torch.rsqrt(variance + self.eps))
        x = x.to(original_dtype).mul_(self.weight)
        return x
    
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(original_dtype)
        variance = x.pow(2).mean(-1, keepdim=True)
        x.mul_(torch.rsqrt(variance + self.eps))
        x = x.to(original_dtype).mul_(self.weight)
        return x,residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.rms_forward(x) if residual is None else self.add_rms_forward(x, residual)