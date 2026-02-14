import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self, ):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, pick_max: bool = False) -> torch.Tensor:
        '''
        logits: [number_of_sequences, vocab_size], which contains the possibility distribution over the vocabulary for the next token for each sequence.
        temperatures: [number_of_sequences] for each sequence.
        pick_max: If True, pick the token with the highest probability (For debug, because it will choose the same token for same input.). 
        If False, sample from the distribution. It uses the "Gumbel-max trick" to sample from the distribution without explicitly computing the softmax, which is more efficient and numerically stable.
        
        Here is the how temperature affects the sampling:
        - When temperature is higher, the distribution is like a boiling water and the lower possibility token has more chance to be chosen.
        For example, logit = [0.1, 0.2, 0.3]:
        when temperature of 0.1, logit.div(0.1) it becomes [1, 2, 3],
        and after softmax, the probabilities will be close to [0.0900, 0.2447, 0.6652], the first element possibility is 9%.
        which means the model is more likely to pick the token with the highest logit (the 3rd token in this case). 
        And the model is more "conservative" and less creative.
        
        when temperature of 0.9, logit.div(0.9) it becomes [0.1111, 0.2222, 0.3333], and after softmax, 
        the probabilities will be close to[0.2971, 0.3320, 0.3710], the first element possibility is about 30%.
        which means the model is more likely to pick the token with lower logit (the 1st token in this case).
        And the model is more "creative" and less conservative.
        
        Gumbel-max trick:
        Here is a very good blog post that explains the Gumbel-max trick in detail: https://amid.fish/humble-gumbel
        For chinese: https://www.cnblogs.com/initial-h/p/9468974.html
        
        '''
        if pick_max:
            return logits.argmax(dim=-1)
        else:
            temperatured_logits = logits.float().div_(temperatures.unsqueeze(dim = -1))
            probs = torch.softmax(temperatured_logits, dim=-1)
            # The Gumbel-max trick: https://amid.fish/humble-gumbel,
            # same as following code but more efficient and numerically stable:
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            return next_tokens