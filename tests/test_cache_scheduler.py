import torch

from ai_assistant_pro.engine.cache import CacheManager, PagedKVCache
from ai_assistant_pro.engine.scheduler import ContinuousBatchScheduler


def test_paged_kv_cache_allocate_write_read_and_free():
    cache = PagedKVCache(
        num_blocks=3,
        block_size=2,
        num_heads=1,
        head_dim=1,
        num_layers=1,
        dtype=torch.float16,
        device="cpu",
    )

    # Allocate blocks for one sequence
    allocated = cache.allocate_blocks(seq_id=0, num_blocks=2)
    assert allocated == [0, 1]
    assert cache.get_num_free_blocks() == 1
    assert cache.get_num_allocated_blocks(seq_id=0) == 2

    # Write then read back a token
    k = torch.tensor([[1.0]], dtype=torch.float16)
    v = torch.tensor([[2.0]], dtype=torch.float16)
    cache.write_kv(layer_idx=0, seq_id=0, token_idx=0, k=k, v=v)
    k_out, v_out = cache.read_kv(layer_idx=0, seq_id=0, token_idx=0)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)

    # Free sequence returns blocks
    cache.free_sequence(seq_id=0)
    assert cache.get_num_free_blocks() == 3
    assert cache.get_num_allocated_blocks(seq_id=0) == 0


def test_cache_manager_add_append_and_remove_sequence():
    manager = CacheManager(
        max_num_blocks=4,
        block_size=2,
        num_heads=1,
        head_dim=1,
        num_layers=1,
        dtype=torch.float16,
        device="cpu",
    )

    manager.add_sequence(seq_id=1, prompt_len=3)  # needs 2 blocks
    assert manager.get_sequence_length(1) == 3
    assert manager.cache.get_num_allocated_blocks(1) == 2

    # Append one token, still within same blocks
    manager.append_token(1)
    assert manager.get_sequence_length(1) == 4
    assert manager.cache.get_num_allocated_blocks(1) == 2

    # Append another token to trigger new block
    manager.append_token(1)
    assert manager.get_sequence_length(1) == 5
    assert manager.cache.get_num_allocated_blocks(1) == 3

    manager.remove_sequence(1)
    assert manager.cache.get_num_allocated_blocks(1) == 0
    assert manager.get_sequence_length(1) == 0


def test_scheduler_schedule_and_finish():
    scheduler = ContinuousBatchScheduler(max_batch_size=2, max_num_sequences=4, block_size=2)

    id1 = scheduler.add_request([1, 2], max_tokens=2)
    id2 = scheduler.add_request([3, 4, 5], max_tokens=2)
    id3 = scheduler.add_request([6], max_tokens=2)

    # Assume plenty of free blocks; should schedule up to batch size
    batch = scheduler.schedule(num_free_blocks=10)
    assert len(batch) == 2
    running_ids = {r.request_id for r in scheduler.running_requests.values()}
    assert running_ids == {id1, id2}
    assert scheduler.get_request(id3).status.name.lower() == "waiting"

    # Mark one finished
    scheduler.update_finished(id1, new_token=7, is_eos=True)
    assert scheduler.get_request(id1).status.name.lower() == "finished"

    # Next schedule should pull the waiting request
    batch2 = scheduler.schedule(num_free_blocks=10)
    scheduled_ids = {r.request_id for r in batch2}
    assert id3 in scheduled_ids
