"""The encoded dataset: writer layouts, in-memory loader and the chunk-streaming loader (CPU)."""

import torch

from latentgen.data import EncodedDataset, EncodedStream, EncodedWriter, ImageStats, open_encoded

N, V, H, W, C = 37, 2, 2, 3, 4


def _write(directory, chunk_items):
    writer = EncodedWriter(directory, N, chunk_items)
    order = torch.randperm(N) if chunk_items else torch.arange(N)
    for start in range(0, N, 5):  # ids arrive in batches, in shuffled order for the chunked layout
        ids = order[start : start + 5]
        # every item's codes / latents encode its own id, so the tests can check what was served
        codes = ids.view(-1, 1, 1, 1).expand(-1, V, H, W).to(torch.int16)
        latents = (ids % 100).view(-1, 1, 1, 1, 1).expand(-1, V, C, H, W).to(torch.int8)
        writer.add(ids, codes.clone(), latents.clone())
    return writer.finish(ImageStats(), {"codebook_size": 64})


def _epoch(data, batch_size):
    served = []
    for codes, latents in data.batches(batch_size):
        assert codes.shape == (batch_size, H, W) and codes.dtype == torch.long
        assert latents.shape == (batch_size, C, H, W) and latents.abs().max() <= 1
        ids = codes[:, 0, 0]
        assert torch.equal(codes, ids.view(-1, 1, 1).expand(-1, H, W))  # rows were not mixed up
        assert torch.allclose(latents[:, 0, 0, 0], (ids % 100).float() / 127, atol=1 / 127)
        served += ids.tolist()
    return served


def test_single_file_layout(tmp_path):
    assert _write(tmp_path, None)["layout"] == "single"
    data = open_encoded(tmp_path, torch.device("cpu"), need_latents=True)
    assert isinstance(data, EncodedDataset)
    assert (data.num_items, data.num_variants, data.grid_h, data.grid_w, data.latent_dim) == (N, V, H, W, C)
    served = _epoch(data, 4)
    assert len(served) == N // 4 and len(set(served)) == len(served)
    coarse_only = open_encoded(tmp_path, torch.device("cpu"), need_latents=False)
    assert coarse_only.latents is None and coarse_only.latent_dim is None


def test_chunked_layout_streams_every_item_once_per_epoch(tmp_path):
    summary = _write(tmp_path, chunk_items=6)
    assert summary["layout"] == "chunked" and summary["chunks"] == -(-N // 6)
    data = open_encoded(tmp_path, torch.device("cpu"), need_latents=True, prefetch_batches=2)
    assert isinstance(data, EncodedStream)
    try:
        for _epoch_index in range(3):  # leftovers are carried across chunks: only the epoch's tail is dropped
            served = _epoch(data, 4)
            assert len(served) == N // 4 * 4 == len(set(served))
            assert data.tracker.num_seen() == len(served)
            data.tracker.reset()  # what the trainer does between epochs
    finally:
        data.close()


def test_stream_resume_skips_seen_items(tmp_path):
    _write(tmp_path, chunk_items=6)
    data = open_encoded(tmp_path, torch.device("cpu"), need_latents=True, prefetch_batches=2)
    try:
        first = []
        for codes, _ in data.batches(3):
            first += codes[:, 0, 0].tolist()
            if len(first) >= 15:
                break  # interrupted mid-epoch
        seen = data.tracker.seen_ids()
        assert sorted(seen.tolist()) == sorted(first)

        resumed = open_encoded(tmp_path, torch.device("cpu"), need_latents=True, prefetch_batches=2)
        resumed.tracker.set_seen(seen)
        rest = _epoch(resumed, 3)
        resumed.close()
        assert not set(rest) & set(first)
        assert len(rest) == (N - len(first)) // 3 * 3
    finally:
        data.close()


def test_chunked_layout_can_be_loaded_into_memory(tmp_path):
    _write(tmp_path, chunk_items=6)
    data = open_encoded(tmp_path, torch.device("cpu"), need_latents=True, loading="memory")
    assert isinstance(data, EncodedDataset)
    assert torch.equal(data.codes[:, 0, 0, 0], torch.arange(N, dtype=torch.int16))
