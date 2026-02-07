import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        temperatured_logits = logits.float().div_(temperatures.unsqueeze(dim = -1))
        probs = torch.softmax(temperatured_logits, dim=-1)
        next_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return next_tokens