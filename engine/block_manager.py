
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
    '''
    BlockManager is responsible for managing the key-value cache blocks for all sequences. It maintains a pool of blocks and allocates/free blocks for sequences as needed.
    Each layer has the same number of kv cache blocks, they keep the same sharp, from management perspective, we treat them as a single layer.
    This is the bridge between the virtual blocks to physical blocks. Each sequence has its own virtual block table which is virtually continuous, which maps the virtual block index to the physical block id in the block manager.
    '''
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
    
    def _allocate_block(self) -> int:
        '''
        Allocate a block and return its id. This function will remove the block id from free_block_ids and add it to used_block_ids.
        '''
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Trying to allocate a used block: {block_id}"
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id
    
    def _free_block(self, block_id:int):
        '''
        Free a block and return it to the pool of free blocks. This function will remove the block id from used_block_ids and add it to free_block_ids.
        '''
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Trying to free a used block: {block_id}"
        #block.reset()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq:Sequence) -> bool:
        '''
        Check if there are enough free blocks to allocate for the sequence.
        This function is called in prefill stage, where we need to allocate all blocks for the sequence.
        So we need to check if there are enough free blocks for the whole sequence.
        '''
        return len(self.free_block_ids) >= seq.num_blocks
    
    def allocate_blocks(self, seq:Sequence) -> bool:
        '''
        Allocate blocks for the sequence. This function is called in prefill stage, where we need to allocate all blocks for the sequence.
        So we need to allocate blocks for the whole sequence.
        
        Firstly, we will check if we can share blocks with existing sequences in the cache. Very block has a hash value (64bit int),
        if the hash is same, then we will compare the token ids to make sure they are the same block. If there is a cache hit, we will share this block and increase its reference count. 
        
        If there is a cache miss, we will allocate a new block for this block and update the hash table.
        
        For the cache mechanism, we can only share all the blocks from sequences beginning to current block if possible,
        as we caculate the hash with the prefix of previous block hash. So if there is a mismatch in the middle block,
        we will not share any blocks after this block, even there are identical tokens with other sequences after hash mismatch.
        Take an example, we have 2 sequences, seq1 with token ids [1,2,3,4,5,6] and seq2 with token ids [1,2,0,4,5,6], and the block size is 2.
        When we allocate blocks for seq1, we will compute the hash for block 0 with token ids [1,2] and prefix -1, 
        and compute the hash for block 1 with token ids [3,4] and prefix hash of block 0 and block 2 with token ids [5,6] and prefix hash of block 1.
        So we have 3 blocks allocated for seq1 and the hash table is updated accordingly.
        When we allocate blocks for seq2, we will compute the hash for block 0 with token ids [1,2] and prefix -1,
        which is identical to block 0 of seq1. So in seq2.num_cached_tokens += 2, but in block 1, we will compute the hash with token ids [0,4] and prefix hash of block 0, 
        which is different from block 1 [3,4] of seq1. So we have a cache miss and we can not share block 1, even there are identical tokens [5,6] in block 2 of both seq1 and seq2.
        This is because we only compute the hash with the prefix of previous block hash, so once there is a mismatch, we will not share any blocks after this block.
        Finally, seq1.num_cached_tokens == 0 and seq2.num_cached_tokens == 2, and both of them have 3 blocks allocated, but seq1 shares block 0 with seq2, and seq2 has its own block 1 and block 2.
        
        So, in prefill stage, it will reuse the shared tokens to reduce the computation. And it assumes that the shared token are from the beginning of the sequence.
        This is what we called Prefix Caching.
        '''
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
                block_id = self._allocate_block()
            else:
               seq.num_cached_tokens += self.block_size

               if block_id in self.used_block_ids:
                   block = self.blocks[block_id]
                   block.ref_count += 1
               else:
                   # Should not happen, because if there is a cache hit, the block should be in used_block_ids.
                   self.logger.error(f"Cache hit but block {block_id} is not in used_block_ids.")
                   block_id = self._allocate_block()
                   block = self.blocks[block_id]

               if hash != -1:
                   block.update(token_ids, hash)
                   self.hash_to_block_id[hash] = block_id
            
            seq.block_table.append(block_id)

        return True
    
    def free_blocks(self, seq:Sequence):
        '''
        Free blocks for the sequence. This function is called when a sequence is finished or preempted.
        We free the block in reverse order, so that we can update the hash table correctly.
        If the block is shared by multiple sequences, we will decrease its reference count and only free it when the reference count reaches zero.
        '''
        assert seq.block_table is not None, "Blocks have not been allocated for this sequence."

        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            assert block.ref_count > 0, f"Trying to free a block with non-positive ref count: {block_id}"
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table = None

    def can_append(self, seq:Sequence) -> bool:
        '''
        After prefill, we only add one token at a time. So we need to check if there is at least one free block.
        So that when the last block is full, we can allocate a new block for the new token.
        That num_tokens % block_size == 1 means the current block is full and we need a new block.
        '''
        return len(self.free_block_ids) >= (seq.num_tokens % self.block_size == 1)
    
    def may_append(self, seq:Sequence):
        '''
        This function is called in decode stage. In this stage, we only add one token at a time.
        So we need to check if we need to allocate a new block or just append to the last block.
        '''
        block_table = seq.block_table

        last_block = self.blocks[block_table[-1]]

        if seq.num_tokens % self.block_size == 1:
            # We are ready to allocate a new block
            assert last_block.hash != -1, "Last block should be full when allocating a new block."
            new_block_id = self._allocate_block()
            block_table.append(new_block_id)
        elif seq.num_tokens % self.block_size == 0:
            # Now, we just add a token to the last block and it reaches full. So we need to mark it as full round block.
            assert last_block.hash == -1, "Last block should not be full when appending to it."
            last_block_token_ids = seq.get_block_token_ids(seq.num_blocks - 1)
            hash_prefix = self.blocks[block_table[-2]].hash if len(block_table ) > 1 else -1
            hash = BlockManager.compute_hash(last_block_token_ids, prefix=hash_prefix)
            last_block.update(last_block_token_ids, hash)
            self.hash_to_block_id[hash] = last_block.block_id
        else:
            assert last_block.hash == -1, "Last block should not be full when appending to it."
