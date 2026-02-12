
from collections import deque
from utils import Config, Logger
from .sequence import Sequence, SequenceState
from .block_manager import BlockManager

class Scheduler:
    def __init__(self, config:Config):
        self.logger = Logger()
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.end_of_sequence_token_ids = config.eos_token_ids
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add_task(self, seq:Sequence):
        '''
        Add a new sequence to the scheduler. The new sequence will be added to the waiting queue and will be scheduled in the next scheduling round.
        '''
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        '''
        Schedule sequences for processing. This function will first try to schedule waiting sequences, and if there is still capacity, it will try to schedule running sequences for decoding.
        
        Prefill Stage (Prompt Processing):
        This is the "reading" phase. When you hit enter on a prompt, the model processes all the input tokens simultaneously.
        
        Decode Stage (Token Generation):
        This is the "writing" phase. The model generates the response one token at a time. And this newly generated token can be used as part of the input for generating the next token,
        which is called auto-regressive decoding until reaching the end of sequence token or max completion tokens.
        '''
        # prefill stage
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
            '''
            If there is not enough blocks for the new token, we need to preempt some sequences to free blocks until there are enough blocks for the new token.
            If we can not preempt any sequence, we have to preempt the current sequence itself and break the loop.
            When comes here, the newly generated token has been added to the sequence, so seq.num_tokens has been updated.
            So we need the call the may_append function to check if we need to allocate a new block for the new token and update the block table accordingly.
            '''
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
        '''
        No waiting sequences and no running sequences means all sequences are finished.
        '''
        return not self.waiting and not self.running
    
    def preempt(self, seq:Sequence):
        '''
        Remove the sequence from running state and put it back to waiting state.
        This function is called when the sequence is preempted due to insufficient blocks for appending new token.
        And also release the blocks occupied by the sequence.
        '''
        seq.state = SequenceState.WAITING
        self.block_manager.free_blocks(seq)
        self.waiting.appendleft(seq)
    
    def post_process(self, seqs: list[Sequence], token_ids: list[int]) -> dict[int, bool]:
        '''
        Check if the sequences are finished after appending the newly generated token. If a sequence is finished, we free its blocks and remove it from running sequences.
        There are two conditions for a sequence to be finished: 
        1) the newly generated token is the end of sequence token and the sequence is not set to ignore eos;
        2) the number of completed tokens reaches the max tokens.
        '''
        if len(seqs) != len(token_ids):
            self.logger.warning(f"Number of sequences {len(seqs)} does not match number of token IDs {len(token_ids)}.")

        finished_flags = {}
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id in self.end_of_sequence_token_ids) or (seq.num_completed_tokens == seq.max_tokens):
                seq.state = SequenceState.FINISHED
                self.block_manager.free_blocks(seq)
                self.running.remove(seq)
                finished_flags[seq.seq_id] = True
            else:
                seq.state = SequenceState.WAITING
                finished_flags[seq.seq_id] = False
        return finished_flags