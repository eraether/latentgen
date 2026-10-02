"""
The encoded dataset: output of stage 1c, input of stages 2 and 3.

Two files in ``data.encoded_dir``, so stage 2 never has to read the large one::

    coarse_encoded.pt   VQ code grids    int16 [N, V, H, W]                + stats + meta   (stages 2 and 3)
    fine_encoded.pt     AE latents       int8  [N, V, C, H, W] = round(x * 127)             (stage 3 only)

``N`` is the number of items, ``V`` the variants per item (1, or 2 = original + horizontal flip /
inverted polarity). On FFHQ the coarse file is ~290 MB and the fine one ~4.6 GB.

**Chunked layout.** When the fine latents would be larger than ``encode.max_file_gb``, stage 1c
writes them in chunks instead, and both files become small indexes::

    coarse_encoded.pt   stats, meta, list of chunks (+ the item ids in each)
    fine_encoded.pt     list of chunks
    coarse/00000.pt     {"ids": [n], "codes":   int16 [n, V, H, W]}
    fine/00000.pt       {"ids": [n], "latents": int8  [n, V, C, H, W]}

Chunk ``k`` of both holds the same items in the same order, and every chunk is a random subset
of the dataset (1c encodes in shuffled order), so reading chunk by chunk still gives well-mixed
batches. :func:`open_encoded` picks the loader:

* :class:`EncodedDataset` -- everything in CPU RAM or VRAM, random batches by index. The default for
  single-file datasets (and ``data.encoded_loading: memory`` for chunked ones that fit).
* :class:`EncodedStream` -- the default for chunked datasets. A reader thread loads the next chunk
  from disk while the current one is consumed, a batch thread gathers batches into pinned host
  buffers and uploads them on a side CUDA stream, and the training loop pops batches that are already
  on the GPU. The only synchronization on the training thread is a GPU-side ``wait_event``.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from latentgen.data.epoch import EpochTracker
from latentgen.data.normalization import ImageStats

LATENT_SCALE = 127  # int8 quantization of the [-1, 1] AE latent
FORMAT_VERSION = 3
COARSE_FILE = "coarse_encoded.pt"
FINE_FILE = "fine_encoded.pt"
GB = 1024**3


def dequantize(latents: torch.Tensor) -> torch.Tensor:
    """int8 latents -> float in ``[-1, 1]``, with uniform noise one int8 step wide.

    The noise turns the 255 stored levels back into a continuous distribution (otherwise the GAN
    discriminator could spot "real" latents by their quantization); the clamp keeps a real latent
    inside the generator's ``tanh`` range.
    """
    fine = latents.float() / LATENT_SCALE
    return (fine + (torch.rand_like(fine) - 0.5) / LATENT_SCALE).clamp_(-1.0, 1.0)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ----------------------------------------------------------------------------- writing (stage 1c)


class EncodedWriter:
    """Collects ``(ids, codes, latents)`` batches from stage 1c and writes the two files.

    Args:
        directory: ``data.encoded_dir``.
        num_items: ``N``.
        chunk_items: ``None`` for the single-file layout, else the number of items per chunk. Items
            may then arrive in any order (1c feeds them shuffled); chunks are written as they fill,
            so memory stays bounded however large the dataset is.
    """

    def __init__(self, directory: str | Path, num_items: int, chunk_items: int | None = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.num_items = num_items
        self.chunk_items = chunk_items
        self.codes: torch.Tensor | None = None  # single-file layout: [N, V, H, W]
        self.latents: torch.Tensor | None = None
        self._buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []  # chunked layout
        self._buffered = 0
        self.chunks: list[dict] = []
        for sub in ("coarse", "fine"):  # chunks of an earlier run would be orphaned (or worse, mixed in)
            if (self.dir / sub).is_dir():
                shutil.rmtree(self.dir / sub)
        if chunk_items is not None:
            (self.dir / "coarse").mkdir()
            (self.dir / "fine").mkdir()

    def add(self, ids: torch.Tensor, codes: torch.Tensor, latents: torch.Tensor) -> None:
        """``ids [b]``, ``codes [b, V, H, W]`` int16, ``latents [b, V, C, H, W]`` int8, all on the CPU."""
        if self.chunk_items is None:
            if self.codes is None:
                self.codes = torch.zeros((self.num_items, *codes.shape[1:]), dtype=torch.int16)
                self.latents = torch.zeros((self.num_items, *latents.shape[1:]), dtype=torch.int8)
            self.codes[ids] = codes
            self.latents[ids] = latents
            return
        self._shapes = (codes.shape[1:], latents.shape[1:])
        self._buffer.append((ids, codes, latents))
        self._buffered += ids.numel()
        while self._buffered >= self.chunk_items:
            self._flush(self.chunk_items)

    def _flush(self, n: int) -> None:
        ids, codes, latents = (torch.cat(t) for t in zip(*self._buffer, strict=True))
        rest = (ids[n:], codes[n:], latents[n:])
        self._buffer = [rest] if rest[0].numel() else []
        self._buffered = rest[0].numel()
        k = len(self.chunks)
        name = f"{k:05d}.pt"
        _atomic_save({"ids": ids[:n].clone(), "codes": codes[:n].clone()}, self.dir / "coarse" / name)
        _atomic_save({"ids": ids[:n].clone(), "latents": latents[:n].clone()}, self.dir / "fine" / name)
        self.chunks.append({"name": name, "ids": ids[:n].clone()})

    def finish(self, stats: ImageStats, meta: dict) -> dict:
        """Write the remaining chunk and both index / data files. Returns a short summary."""
        if self.chunk_items is None:
            item_shapes = (tuple(self.codes.shape[1:]), tuple(self.latents.shape[1:]))
        else:
            item_shapes = (tuple(self._shapes[0]), tuple(self._shapes[1]))
        header = {
            "format_version": FORMAT_VERSION,
            "num_items": self.num_items,
            "codes_shape": item_shapes[0],  # per item: [V, H, W]
            "latents_shape": item_shapes[1],  # per item: [V, C, H, W]
        }
        meta = {"latent_scale": LATENT_SCALE, **meta}
        if self.chunk_items is None:
            _atomic_save(
                {**header, "layout": "single", "codes": self.codes, "stats": stats.to_dict(), "meta": meta},
                self.dir / COARSE_FILE,
            )
            _atomic_save({**header, "layout": "single", "latents": self.latents}, self.dir / FINE_FILE)
            return {
                "layout": "single",
                "codes": tuple(self.codes.shape),
                "latents": tuple(self.latents.shape),
            }
        if self._buffered:
            self._flush(self._buffered)
        _atomic_save(
            {
                **header,
                "layout": "chunked",
                "chunks": [c["name"] for c in self.chunks],
                "chunk_ids": [c["ids"] for c in self.chunks],
                "stats": stats.to_dict(),
                "meta": meta,
            },
            self.dir / COARSE_FILE,
        )
        _atomic_save(
            {**header, "layout": "chunked", "chunks": [c["name"] for c in self.chunks]}, self.dir / FINE_FILE
        )
        return {"layout": "chunked", "chunks": len(self.chunks), "items_per_chunk": self.chunk_items}


# ----------------------------------------------------------------------------- reading (stages 2, 3)


def _read_index(directory: Path, need_latents: bool) -> tuple[dict, dict | None]:
    coarse_path = directory / COARSE_FILE
    if not coarse_path.is_file():
        raise FileNotFoundError(
            f"encoded dataset not found: {coarse_path}. Run scripts/01c_encode_dataset.py first "
            f"(and check data.encoded_dir)."
        )
    coarse = torch.load(coarse_path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(coarse, dict) or coarse.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{coarse_path} is not a format-{FORMAT_VERSION} encoded dataset; re-run stage 1c")
    fine = None
    if need_latents:
        fine_path = directory / FINE_FILE
        if not fine_path.is_file():
            raise FileNotFoundError(
                f"{fine_path} not found; stage 3 needs the fine latents (re-run stage 1c)"
            )
        fine = torch.load(fine_path, map_location="cpu", mmap=True, weights_only=False)
        if fine.get("layout") != coarse["layout"] or fine.get("num_items") != coarse["num_items"]:
            raise ValueError(f"{fine_path} does not belong to {coarse_path} (written by different 1c runs?)")
    return coarse, fine


def open_encoded(
    directory: str | Path,
    device: torch.device,
    need_latents: bool,
    storage: str = "cpu",
    loading: str = "auto",
    prefetch_batches: int = 16,
) -> EncodedDataset | EncodedStream:
    """Open ``data.encoded_dir`` with the right loader (see module docstring).

    ``loading``: ``"auto"`` streams chunked datasets and keeps single-file ones in memory;
    ``"memory"`` / ``"stream"`` force one or the other (a single-file dataset is always in memory).
    """
    if loading not in ("auto", "memory", "stream"):
        raise ValueError(f"data.encoded_loading must be 'auto', 'memory' or 'stream', got {loading!r}")
    directory = Path(directory)
    coarse, fine = _read_index(directory, need_latents)
    if coarse["layout"] == "chunked" and loading != "memory":
        return EncodedStream(directory, coarse, fine, device, prefetch_batches)
    if coarse["layout"] == "single" and loading == "stream":
        print("data.encoded_loading=stream: this dataset is a single file, keeping it in memory instead")
    return EncodedDataset(directory, coarse, fine, device, storage)


class _EncodedInfo:
    """Shapes and metadata shared by both loaders."""

    def _init_info(self, coarse: dict, codes_shape, latents_shape) -> None:
        self.stats = ImageStats.from_dict(coarse.get("stats"))
        self.meta = dict(coarse.get("meta", {}))
        self.num_items = int(coarse["num_items"])
        self.num_variants, self.grid_h, self.grid_w = codes_shape
        self.codebook_size = int(self.meta["codebook_size"])
        self.latent_dim = None if latents_shape is None else latents_shape[1]

    def describe(self, where: str) -> str:
        return (
            f"Encoded dataset: {self.num_items} items x {self.num_variants} variants, grid "
            f"{self.grid_h}x{self.grid_w}, codebook {self.codebook_size}, fine latents "
            f"{'not loaded' if self.latent_dim is None else f'{self.latent_dim} ch'}, {where}"
        )


class EncodedDataset(_EncodedInfo):
    """Random-access batches of ``(codes, latents)`` with epoch tracking, everything held in memory.

    ``storage`` is where the tensors live between batches: ``"cpu"`` (saves VRAM; each batch is
    copied over) or ``"cuda"`` (fastest). Batches are always returned on ``device``.
    """

    def __init__(
        self, directory: Path, coarse: dict, fine: dict | None, device: torch.device, storage: str
    ) -> None:
        self.device = device
        self.storage = torch.device(storage if storage != "cuda" else device)
        if coarse["layout"] == "single":
            codes = coarse["codes"]
            latents = None if fine is None else fine["latents"]
        else:  # chunked dataset loaded whole (data.encoded_loading: memory)
            codes, latents = self._concat_chunks(directory, coarse, fine is not None)
        self._init_info(coarse, codes.shape[1:], None if latents is None else latents.shape[1:])
        self.codes = self._store(codes)
        self.latents = None if latents is None else self._store(latents)
        self.tracker = EpochTracker(self.num_items, device=self.storage)
        print(self.describe(f"{self.size_gb():.2f} GB in memory on {self.storage}"))

    @staticmethod
    def _concat_chunks(directory: Path, coarse: dict, with_latents: bool):
        codes = latents = None
        for name, ids in zip(coarse["chunks"], coarse["chunk_ids"], strict=True):
            c = torch.load(directory / "coarse" / name, map_location="cpu", weights_only=True)["codes"]
            if codes is None:
                codes = torch.zeros((coarse["num_items"], *c.shape[1:]), dtype=c.dtype)
            codes[ids] = c
            if with_latents:
                lat = torch.load(directory / "fine" / name, map_location="cpu", weights_only=True)["latents"]
                if latents is None:
                    latents = torch.zeros((coarse["num_items"], *lat.shape[1:]), dtype=lat.dtype)
                latents[ids] = lat
        return codes, latents

    def _store(self, t: torch.Tensor) -> torch.Tensor:
        # the file is memory-mapped; copying makes the tensor live in RAM (or VRAM) for fast random gathers
        return t.to(self.storage) if self.storage.type == "cuda" else t.clone()

    def size_gb(self) -> float:
        n = self.codes.numel() * self.codes.element_size()
        if self.latents is not None:
            n += self.latents.numel() * self.latents.element_size()
        return n / GB

    def get_batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``idx: [B]`` long on the storage device -> ``(codes [B, H, W] long, latents [B, C, H, W] float or None)``.

        A random variant (flip) is picked per sample; latents are dequantized (see :func:`dequantize`).
        """
        variant = torch.randint(0, self.num_variants, (idx.numel(),), device=self.storage)
        codes = self.codes[idx, variant].to(self.device).long()
        latents = None
        if self.latents is not None:
            latents = dequantize(self.latents[idx, variant].to(self.device))
        self.tracker.mark(idx)
        return codes, latents

    def batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        """Batches for the part of the epoch not yet seen (ragged tail dropped)."""
        order = self.tracker.remaining_order()
        for b in range(order.numel() // batch_size):
            yield self.get_batch(order[b * batch_size : (b + 1) * batch_size])

    def batches_per_epoch(self, batch_size: int) -> int:
        return self.num_items // batch_size


# ----------------------------------------------------------------------------- streaming


class _StreamStats:
    """Thread-safe counters for one logging window (the stream threads and the training loop write)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sums: dict[str, float] = {}
        self._maxes: dict[str, float] = {}

    def add(self, key: str, value: float) -> None:
        with self._lock:
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._maxes[key] = max(self._maxes.get(key, value), value)

    def pop(self) -> tuple[dict[str, float], dict[str, float]]:
        with self._lock:
            sums, maxes = self._sums, self._maxes
            self._sums, self._maxes = {}, {}
        return sums, maxes


_EPOCH_END = object()


class EncodedStream(_EncodedInfo):
    """Chunk-streaming twin of :class:`EncodedDataset` (same interface, see module docstring).

    Epochs visit the chunks in a fresh random order and each chunk's items in a fresh random order;
    leftovers smaller than a batch are carried into the next chunk, so an epoch yields
    ``unseen // batch_size`` batches just like the in-memory loader. Items are marked as seen when the
    training loop *receives* them, not when they are prefetched, so a checkpoint never skips data.
    The producer runs ahead across epoch boundaries: the next epoch's first chunk is already loaded
    when the current epoch ends.
    """

    def __init__(
        self, directory: Path, coarse: dict, fine: dict | None, device: torch.device, prefetch_batches: int
    ) -> None:
        self.dir = directory
        self.device = device
        self.with_latents = fine is not None
        self.chunk_names: list[str] = list(coarse["chunks"])
        self.chunk_ids: list[torch.Tensor] = [
            torch.as_tensor(i, dtype=torch.long) for i in coarse["chunk_ids"]
        ]
        self._init_info(coarse, coarse["codes_shape"], coarse["latents_shape"] if self.with_latents else None)
        self.depth = max(2, prefetch_batches)
        self.tracker = EpochTracker(self.num_items, device="cpu")
        self.stats_window = _StreamStats()
        self._queue: queue.Queue = queue.Queue(maxsize=self.depth)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._batch_size: int | None = None
        self._at_epoch_boundary = False
        item_bytes = 2 * torch.Size(coarse["codes_shape"]).numel()
        if self.with_latents:
            item_bytes += torch.Size(coarse["latents_shape"]).numel()
        chunk_gb = item_bytes * self.num_items / max(1, len(self.chunk_names)) / GB
        print(self.describe(f"streamed from {len(self.chunk_names)} chunks of ~{chunk_gb:.2f} GB"))

    # ------------------------------------------------------------------ interface used by the trainer

    def batches_per_epoch(self, batch_size: int) -> int:
        return self.num_items // batch_size

    def batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        # continue the running producer only if the previous epoch was consumed to the end with the same
        # batch size; otherwise (first call, resume, interrupted epoch) start fresh from the tracker
        if self._thread is None or not self._at_epoch_boundary or batch_size != self._batch_size:
            self._restart(batch_size)
        self._at_epoch_boundary = False
        while True:
            item = self._get()
            if item is _EPOCH_END:
                self._at_epoch_boundary = True
                return
            codes, latents, ids = self._receive(item)
            self.tracker.mark(ids)
            yield codes, latents

    def pop_timing(self) -> dict[str, float]:
        """Data-pipeline numbers for the last logging window (the trainer writes them to TensorBoard)."""
        sums, maxes = self.stats_window.pop()
        return {
            "stalls": sums.get("stall_count", 0.0),  # times the training loop found no batch ready
            "stall_ms": 1000 * sums.get("stall_s", 0.0),
            "stall_max_ms": 1000 * maxes.get("stall_s", 0.0),
            "chunk_wait_ms": 1000
            * sums.get("chunk_wait_s", 0.0),  # producer blocked on a chunk still loading
            "chunk_load_max_s": maxes.get("chunk_load_s", 0.0),
            "queue_depth_mean": sums.get("depth", 0.0) / max(1.0, sums.get("fetches", 0.0)),
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            while self._thread.is_alive():  # unblock a producer waiting on a full queue
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    self._thread.join(timeout=0.1)
            self._thread = None
        self._queue = queue.Queue(maxsize=self.depth)
        self._stop = threading.Event()

    # ------------------------------------------------------------------ consumer side

    def _restart(self, batch_size: int) -> None:
        self.close()
        self._batch_size = batch_size
        seen = self.tracker.seen.clone()
        self._thread = threading.Thread(
            target=self._produce, args=(batch_size, seen), name="encoded_stream", daemon=True
        )
        self._thread.start()

    def _get(self):
        self.stats_window.add("fetches", 1.0)
        self.stats_window.add("depth", float(self._queue.qsize()))
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            t0 = time.perf_counter()
            item = self._queue.get()
            self.stats_window.add("stall_count", 1.0)
            self.stats_window.add("stall_s", time.perf_counter() - t0)
        if isinstance(item, BaseException):
            raise item
        return item

    def _receive(self, item):
        codes, latents, ids, ready = item
        if ready is not None:  # CUDA: make the compute stream wait for the upload (no host sync)
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(ready)
            codes.record_stream(stream)  # allocated on the copy stream, used on this one
            if latents is not None:
                latents.record_stream(stream)
        codes = codes.long()
        if latents is not None:
            latents = dequantize(latents)
        return codes, latents, ids

    # ------------------------------------------------------------------ producer side (background threads)

    def _load_chunk(self, k: int) -> dict:
        t0 = time.perf_counter()
        name = self.chunk_names[k]
        chunk = torch.load(self.dir / "coarse" / name, map_location="cpu", weights_only=True)
        if self.with_latents:
            fine = torch.load(self.dir / "fine" / name, map_location="cpu", weights_only=True)
            if not torch.equal(fine["ids"], chunk["ids"]):
                raise ValueError(f"coarse and fine chunk {name} hold different items; re-run stage 1c")
            chunk["latents"] = fine["latents"]
        self.stats_window.add("chunk_load_s", time.perf_counter() - t0)
        return chunk

    def _plan(self, seen: torch.Tensor | None, generator: torch.Generator):
        """Infinite sequence of ``("chunk", k, local_positions)`` jobs with ``("end",)`` after each epoch."""
        while True:
            for k in torch.randperm(len(self.chunk_names), generator=generator).tolist():
                ids = self.chunk_ids[k]
                local = torch.arange(ids.numel()) if seen is None else (~seen[ids]).nonzero().squeeze(1)
                if local.numel():
                    yield ("chunk", k, local[torch.randperm(local.numel(), generator=generator)])
            yield ("end",)
            seen = None  # every epoch after the (possibly resumed) first one covers everything

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self, batch_size: int, seen: torch.Tensor) -> None:
        try:
            self._produce_loop(batch_size, seen)
        except BaseException as e:  # noqa: BLE001 - re-raised on the training thread
            self._put(e)

    def _produce_loop(self, batch_size: int, seen: torch.Tensor) -> None:
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.set_device(self.device)
            copy_stream = torch.cuda.Stream(device=self.device)
        generator = torch.Generator().manual_seed(torch.initial_seed() + 1)
        jobs = self._plan(seen if bool(seen.any()) else None, generator)
        reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk_read")
        pending: deque = deque()

        def lookahead() -> None:  # keep the next chunk loading while the current one is consumed
            while not any(j[0] == "chunk" for j in pending):
                job = next(jobs)
                pending.append((*job, reader.submit(self._load_chunk, job[1])) if job[0] == "chunk" else job)

        n_slots = self.depth + 2  # queued + being filled + slack
        slots: list[dict] | None = None
        slot_events: list = [None] * n_slots
        slot_i = 0
        carry: dict | None = None  # < batch_size items left over from the previous chunk (already gathered)
        try:
            lookahead()
            while not self._stop.is_set():
                job = pending.popleft()
                lookahead()
                if job[0] == "end":
                    carry = None  # the ragged tail of an epoch is dropped, as in the in-memory loader
                    if not self._put(_EPOCH_END):
                        return
                    continue
                _, _k, local, future = job
                t0 = time.perf_counter()
                chunk = future.result()
                self.stats_window.add("chunk_wait_s", time.perf_counter() - t0)

                V = self.num_variants
                # flat [n * V, ...] views: item i, variant v lives at row i * V + v
                sources = {"codes": chunk["codes"].flatten(0, 1)}
                if self.with_latents:
                    sources["latents"] = chunk["latents"].flatten(0, 1)
                variant = torch.randint(0, V, (local.numel(),), generator=generator)
                rows = local * V + variant
                item_ids = chunk["ids"]
                if slots is None:
                    slots = [
                        {
                            name: torch.empty((batch_size, *src.shape[1:]), dtype=src.dtype, pin_memory=cuda)
                            for name, src in sources.items()
                        }
                        for _ in range(n_slots)
                    ]

                pos = 0
                while not self._stop.is_set():
                    have = 0 if carry is None else carry["ids"].numel()
                    take = min(batch_size - have, rows.numel() - pos)
                    if have + take < batch_size:  # not enough for a full batch: carry into the next chunk
                        idx = rows[pos : pos + take]
                        part = {name: src.index_select(0, idx) for name, src in sources.items()}
                        part["ids"] = item_ids.index_select(0, local[pos : pos + take])
                        carry = part if carry is None else {n: torch.cat([carry[n], part[n]]) for n in part}
                        break
                    idx = rows[pos : pos + take]
                    ids = item_ids.index_select(0, local[pos : pos + take])
                    pos += take

                    if cuda and slot_events[slot_i] is not None:
                        slot_events[slot_i].synchronize()  # the last upload out of this pinned buffer is done
                    host = slots[slot_i]
                    for name, src in sources.items():
                        if have:
                            host[name][:have].copy_(carry[name])
                        torch.index_select(src, 0, idx, out=host[name][have:])
                    if have:
                        ids = torch.cat([carry["ids"], ids])
                        carry = None

                    if cuda:
                        with torch.cuda.stream(copy_stream):
                            on_device = {n: t.to(self.device, non_blocking=True) for n, t in host.items()}
                            ready = torch.cuda.Event()
                            ready.record(copy_stream)
                        slot_events[slot_i] = ready
                        slot_i = (slot_i + 1) % n_slots
                    else:
                        on_device = {n: t.clone() for n, t in host.items()}
                        ready = None
                    if not self._put((on_device["codes"], on_device.get("latents"), ids, ready)):
                        return
        finally:
            reader.shutdown(wait=False, cancel_futures=True)
