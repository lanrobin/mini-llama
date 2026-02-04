
from collections import deque
import xxhash
import numpy as np
from .sequence import Sequence
from utils import Logger


class Block:
    def __init__(self, block_id:int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, token_ids:list[int], hash:int):
        self.token_ids = token_ids
        self.hash = hash
        
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks:int, block_size:int):
        self.logger = Logger()
        self.block_size = block_size
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, byteorder='little'))
        
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()
    
    def _allocate_block(self, block_id:int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Trying to allocate a used block: {block_id}"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block
    
    def _free_block(self, block_id:int):
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Trying to free a used block: {block_id}"
        #block.reset()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq:Sequence) -> bool:
        return len(self.free_block_ids) > seq.num_blocks
    
    def allocate_blocks(self, seq:Sequence) -> bool:

        assert not seq.block_table, "Blocks have already been allocated for this sequence."

        if not self.can_allocate(seq):
            self.logger.error(f"Not enough free blocks to allocate sequence of {seq.num_blocks} blocks, free blocks: {len(self.free_block_ids)}")
            return False
        
        prefix_hash = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.get_block_token_ids(i)
            # if the block is full, compute its hash
            hash = BlockManager.compute_hash(token_ids, prefix=prefix_hash) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(hash, -1)

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # cache miss
                cache_miss = True

            if cache_miss:
                # allocate a new block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
               seq.num_cached_tokens += self.block_size

               if block_id in self.used_block_ids:
                   block = self.blocks[block_id]
                   block.ref_count += 1
               else:
                   block = self._allocate_block(block_id)

               if hash != -1:
                   block.update(token_ids, hash)
                   self.hash_to_block_id[hash] = block_id
            
            seq.block_table.append(block_id)

        return True
    
    def free_blocks(self, seq:Sequence):
        assert seq.block_table is not None, "Blocks have not been allocated for this sequence."

        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table = None

    def can_append(self, seq:Sequence) -> bool:
        '''
        After prefill, we only add one token at a time. So we need to check if there is at least one free block.
        So that when the current block is full, we can allocate a new block for the new token.
        That num_tokens % block_size == 1 means the current block is full and we need a new block.
        :param self: Description
        :param seq: Description
        :type seq: Sequence
        :return: Description
        :rtype: bool
        '''
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
    
    def may_append(self, seq:Sequence):
        block_table = seq.block_table

        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # We are ready to allocate a new block
            assert last_block.hash != -1, "Last block should be full when allocating a new block."
            new_block_id = self.free_block_ids[0]
            new_block = self._allocate_block(new_block_id)
            block_table.append(new_block_id)
        elif len(seq) % self.block_size == 0:
            # Now, we just add a token to the last block and it reaches full. So we need to mark it as full round block.
            assert last_block.hash == -1, "Last block should not be full when appending to it."
            last_block_token_ids = seq.get_block_token_ids(seq.num_blocks - 1)
            hash_prefix = self.blocks[block_table[-2]].hash if len(block_table ) > 1 else -1
            hash = BlockManager.compute_hash(last_block_token_ids, prefix=hash_prefix)
            last_block.update(last_block_token_ids, hash)
            self.hash_to_block_id[hash] = last_block.block_id
        else:
            assert last_block.hash == -1, "Last block should not be full when appending to it."
