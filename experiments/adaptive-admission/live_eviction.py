"""Install cost-ranked eviction in the vLLM 0.32.0 GPU block pool."""

import ctypes
import importlib.metadata
from pathlib import Path


class Candidate(ctypes.Structure):
    _fields_ = [("position", ctypes.c_size_t), ("score", ctypes.c_double),
                ("cached", ctypes.c_int)]


class BlockEviction:
    # Full-prefix cold minus warm TTFT from the saved H100 baseline.
    COSTS = {512: 0.966199, 1024: 1.932398, 2048: 11.102428, 4096: 38.339725}

    def __init__(self):
        library = ctypes.CDLL(str(Path(__file__).with_name("block_rank.so")))
        self.rank = library.rank_blocks
        self.rank.argtypes = [ctypes.POINTER(Candidate), ctypes.c_size_t]
        self.rank.restype = None
        self.age = 0.0
        self.density = {}
        self.priority = {}
        print("Cost-ranked KV block eviction enabled", flush=True)

    def cached(self, block, request):
        length = request.num_prompt_tokens - 1
        cost = self.COSTS.get(length, 0)
        self.density[block.block_id] = cost / length if cost else 0.0

    def released(self, blocks):
        for block in blocks:
            self.priority[block.block_id] = self.age + self.density.get(block.block_id, 0.0)

    def allocate(self, queue, count):
        if not count:
            return queue.popleft_n(count)
        # Avoid sorting when all requested blocks are already uncached.
        block = queue.fake_free_list_head.next_free_block
        for _ in range(count):
            if block.block_hash is not None:
                break
            block = block.next_free_block
        else:
            return queue.popleft_n(count)

        blocks = queue.get_all_free_blocks()
        ranked = (Candidate * len(blocks))(*(
            Candidate(index, self.priority.get(block.block_id, self.age)
                      if block.block_hash is not None else 0.0,
                      block.block_hash is not None)
            for index, block in enumerate(blocks)
        ))
        self.rank(ranked, len(blocks))
        selected = []
        for rank in ranked[:count]:
            block = blocks[rank.position]
            queue.remove(block)
            if rank.cached:
                self.age = max(self.age, rank.score)
            self.density.pop(block.block_id, None)
            self.priority.pop(block.block_id, None)
            selected.append(block)
        return selected


def install():
    distribution = importlib.metadata.distribution("vllm")
    if distribution.version != "0.32.0":
        raise RuntimeError("Expected vLLM 0.32.0")
    path = Path(distribution.locate_file("vllm/v1/core/block_pool.py"))
    source = path.read_text(encoding="utf-8")
    edits = (
        ("        self.enable_caching = enable_caching",
         "        self.enable_caching = enable_caching\n"
         "        import os\n"
         "        from live_eviction import BlockEviction\n"
         "        self.cost_eviction = (BlockEviction() if enable_caching and\n"
         "                              os.getenv('KV_ADMISSION') == 'cost' else None)"),
        ("            block_hash = new_block_hashes[i]",
         "            block_hash = new_block_hashes[i]\n"
         "            if self.cost_eviction is not None:\n"
         "                self.cost_eviction.cached(blk, request)"),
        ("        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)",
         "        ret: list[KVCacheBlock] = (self.free_block_queue.popleft_n(num_blocks)\n"
         "            if self.cost_eviction is None else\n"
         "            self.cost_eviction.allocate(self.free_block_queue, num_blocks))"),
        ("        self.free_block_queue.append_n(blocks_to_evict_last)",
         "        self.free_block_queue.append_n(blocks_to_evict_last)\n"
         "        if self.cost_eviction is not None:\n"
         "            self.cost_eviction.released(blocks_to_evict_last)"),
    )
    if all(new in source for _, new in edits):
        return
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError(f"Unexpected block pool source near {old!r}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    install()
