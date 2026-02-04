
from collections import deque
from utils import Config, Logger
from .sequence import Sequence, SequenceState
from .block_manager import BlockManager

class Scheduler:
    def __init__(self, config:Config):
        self.logger = Logger()
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.end_of_sequence_token_id = config.eos_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add_task(self, seq:Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        raise NotImplementedError("This is a placeholder method. Implement the schedule method.")
    
    def is_finished(self) -> bool:
        return not self.waiting and not self.running
    
    def _preempt(self, seq:Sequence):
        seq.state = SequenceState.WAITING
        self.block_manager.free_blocks(seq)
        self.waiting.appendleft(seq)
    
    def post_process(self, token_ids: list[int]):
        raise NotImplementedError("This is a placeholder method. Implement the post_process method.")