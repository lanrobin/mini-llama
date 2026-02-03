
from copy import copy
from enum import Enum, auto
from itertools import count
from utils.sampling_params import SamplingParams
from utils import CONST


class SequenceState(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = CONST.DEFAULT_KV_CACHE_BLOCK_SIZE
    counter = count(start=0)

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.state = SequenceState.WAITING
        self.token_ids = copy(token_ids)
        self.last_token_id = self.token_ids[-1] if len(self.token_ids) > 0 else None
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos


    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token_id = token_id
        self.num_tokens += 1

    def block(self, block_idx: int) -> list[int]:
        assert 0 <= block_idx < self.num_blocks, f"Block index {block_idx} out of range for sequence with {self.num_blocks} blocks."
        start_idx = block_idx * Sequence.block_size
        end_idx = min(start_idx + Sequence.block_size, self.num_tokens)
        return self.token_ids[start_idx:end_idx]
    
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    def __len__(self) -> int:
        return self.num_tokens
    
    def __getitem__(self, idx: slice) -> list[int]:
        return self.token_ids[idx]
    
    def __getstate__(self) -> tuple:
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token_id)
    
    def __setstate__(self, state: tuple):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token_id = state[-1]

    @property
    def is_finished(self) -> bool:
        return self.state == SequenceState.FINISHED
    
    @property
    def prompt_tokens(self) -> list[int]:
        return self.token_ids[:self.num_prompt_tokens]
    
    @property
    def completion_tokens(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + Sequence.block_size - 1) // Sequence.block_size
    
    @property
    def num_cached_blocks(self) -> int:
        '''
        We can only cache the tokens that fill complete blocks. Partial blocks are not cached.
        '''
        return self.num_cached_tokens// Sequence.block_size
    
    @property
    def last_block_num_tokens(self) -> int:
        '''
        Returns the number of tokens in the last block.
        '''
        return self.num_tokens - (self.num_blocks - 1) * Sequence.block_size
  
    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens