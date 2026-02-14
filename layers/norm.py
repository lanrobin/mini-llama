import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        '''
        self.weight.shape = [hidden_size] = [3072]
        '''
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
          variance = sum(x^2)/n on each row (dim = -1).
          x = x / sqrt(variance + eps) * weight
          
          Example:
            x = torch.randint(1, 5, (3,4))
            xf = x.float()
            x = tensor(
                    [[1, 4, 3, 3],
                    [2, 1, 1, 1],
                    [3, 3, 4, 4]])
            xf = tensor(
                    [[1., 4., 3., 3.],
                    [2., 1., 1., 1.],
                    [3., 3., 4., 4.]])
            xpow2 = xf.pow(2)
            xpow2 = tensor(
                    [[ 1., 16.,  9.,  9.],
                    [ 4.,  1.,  1.,  1.],
                    [ 9.,  9., 16., 16.]])

            xpow2_mean = xpow2.mean(dim=-1, keepdim=True)
            xpow2_mean = tensor(
                    [[ 8.7500],
                    [ 1.7500],
                    [12.5000]])
            x_rms = xf.div_(xpow2_mean)
            x_rms = tensor(
                    [[0.1143, 0.4571, 0.3429, 0.3429],
                    [1.1429, 0.5714, 0.5714, 0.5714],
                    [0.2400, 0.2400, 0.3200, 0.3200]])
            x = xrms * self.weight
        '''
        original_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x.mul_(torch.rsqrt(variance + self.eps))
        x = x.to(original_dtype).mul_(self.weight)
        return x
    
    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
          x = x + residual
          variance = sum(x^2)/n on each row (dim = -1).
          x = x / sqrt(variance + eps) * weight
        '''
        original_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(original_dtype)
        variance = x.pow(2).mean(-1, keepdim=True)
        x.mul_(torch.rsqrt(variance + self.eps))
        x = x.to(original_dtype).mul_(self.weight)
        return x,residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.rms_forward(x) if residual is None else self.add_rms_forward(x, residual)