
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
        # prefill stage.
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if (num_batched_tokens + seq.num_tokens > self.max_num_batched_tokens) or not self.block_manager.can_allocate(seq):
               self.logger.debug(f"Cannot schedule seq {seq.seq_id}: batched tokens {num_batched_tokens + seq.num_tokens}, available blocks {len(self.block_manager.free_block_ids)}, required blocks {seq.num_blocks}.")
               break

            num_seqs += 1
            self.block_manager.allocate_blocks(seq)
            num_batched_tokens += (seq.num_tokens - seq.num_cached_tokens)
            seq.state = SequenceState.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self.logger.info(f"Scheduled {len(scheduled_seqs)} sequences for processing. Total batched tokens: {num_batched_tokens}.")
            return scheduled_seqs, True
    
        # decode stage.
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                self.logger.debug(f"Preempting seq {seq.seq_id} due to insufficient blocks.")
                if self.running:
                    # If there are other running sequences, preempt the last one
                    self.preempt(self.running.pop())
                else:
                    # No other sequences to preempt, preempt my self.
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs,  "Scheduler must schedule at least one sequence."
        self.logger.info(f"Scheduled {len(scheduled_seqs)} sequences for decoding.")
        self.running.extendleft(reversed(scheduled_seqs))  # Put them back to running
        return scheduled_seqs, False

    def is_finished(self) -> bool:
        return not self.waiting and not self.running
    
    def preempt(self, seq:Sequence):
        seq.state = SequenceState.WAITING
        self.block_manager.free_blocks(seq)
        self.waiting.appendleft(seq)
    
    def post_process(self, seqs: list[Sequence], token_ids: list[int]) -> dict[int, bool]:

        if len(seqs) != len(token_ids):
            self.logger.warning(f"Number of sequences {len(seqs)} does not match number of token IDs {len(token_ids)}.")

        finished_flags = {}
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.end_of_sequence_token_id) or (seq.num_completion_tokens == seq.max_tokens):
                seq.state = SequenceState.FINISHED
                self.block_manager.free_blocks(seq)
                self.running.remove(seq)
                finished_flags[seq.seq_id] = True
            else:
                seq.state = SequenceState.WAITING
                finished_flags[seq.seq_id] = False
        return finished_flags