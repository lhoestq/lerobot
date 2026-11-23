#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import os
import random
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from functools import cached_property, partial
from itertools import batched, islice
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import datasets
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image as PILImage

from lerobot.configs import DEFAULT_DEPTH_UNIT, DEPTH_METER_UNIT, DepthEncoderConfig
from lerobot.utils.constants import HF_LEROBOT_HOME

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .feature_utils import get_delta_indices
from .io_utils import pil_to_chw_tensor
from .language import LANGUAGE_COLUMNS
from .utils import check_version_compatibility, resolve_episode_indices
from .video_utils import (
    _decode_torchcodec_frames,
    _match_torchcodec_frames,
    _open_streaming_video,
    _StreamingVideoDecoderPool,
    decode_video_frames,
    decode_video_frames_torchcodec,
)

logger = logging.getLogger(__name__)

DEFAULT_STREAMING_SEED = 42
_DEFAULT_EPISODE_POOL_SIZE = 32
_DECODE_WORKERS = 32
# Retain several camera decoders per concurrent sample while bounding open sources.
_DECODER_POOL_CAPACITY = 3 * _DECODE_WORKERS
_RGB_DECODE_GROUP_SIZE = 8


def _balanced_episode_shards(
    episode_indices: list[int],
    episode_frame_counts: Mapping[int, int],
    *,
    world_size: int,
) -> list[list[int]]:
    """Assign whole episodes deterministically with greedy frame-count balancing."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    shards: list[list[int]] = [[] for _ in range(world_size)]
    shard_frames = [0] * world_size
    for episode in sorted(episode_indices, key=lambda item: (-episode_frame_counts[item], item)):
        rank = min(range(world_size), key=lambda item: (shard_frames[item], item))
        shards[rank].append(episode)
        shard_frames[rank] += episode_frame_counts[episode]
    return shards


class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """LeRobotDataset with streaming capabilities.

    A finite iteration covers the complete episodes owned by this data-parallel rank and
    DataLoader worker. Each level uses deterministic frame-count balancing. Episodes stay
    intact across Parquet row groups so temporal targets have the full episode available.

    Episode read-ahead and the remote video cache have separate limits.

    For RGB-only reads of current frames without transforms, the reader groups requests for the
    same video within eight consecutive samples. This group size is independent of the DataLoader
    batch size and preserves sample order.

    Example:
        Basic usage:
        ```python
        from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

        # Create a streaming dataset with delta timestamps
        delta_timestamps = {
            "observation.image": [-1.0, -0.5, 0.0],  # 1 sec ago, 0.5 sec ago, current
            "action": [0.0, 0.1, 0.2],  # current, 0.1 sec future, 0.2 sec future
        }

        dataset = StreamingLeRobotDataset(
            repo_id="your-dataset-repo-id",
            delta_timestamps=delta_timestamps,
            episode_pool_size=32,
        )

        # Iterate over the dataset
        for i, item in enumerate(dataset):
            print(f"Sample {i}: Episode {item['episode_index']} Frame {item['frame_index']}")
            # item will contain stacked frames according to delta_timestamps
            if i >= 10:
                break
        ```
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        buffer_size: int | None = None,
        seed: int = DEFAULT_STREAMING_SEED,
        shuffle: bool = True,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        *,
        repo_type: Literal["dataset", "bucket"] = "dataset",
        token: str | bool | None = None,
        episode_pool_size: int | None = None,
        prefetch_episodes: int = 8,
        video_cache_bytes: int | None = None,
        byte_budget_gb: float | None = None,
        decode_threads: int = _DECODE_WORKERS,
        decoded_queue_size: int = _DECODE_WORKERS,
        video_decoder_cache_size: int | None = None,
        max_num_shards: int | None = None,
    ):
        """Initialize a StreamingLeRobotDataset.

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset.
            root (Path | None, optional): Local directory to use for local datasets. In bucket mode,
                this is an optional local metadata-cache directory; parquet and video data remain remote.
                When omitted, Hub metadata is resolved through the cache under ``$HF_LEROBOT_HOME/hub``.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list.
            image_transforms (Callable | None, optional): Transform to apply to image data.
            tolerance_s (float, optional): Tolerance in seconds for timestamp matching.
            revision (str, optional): Git revision id (branch name, tag, or commit hash).
            force_cache_sync (bool, optional): Flag to sync and refresh local files first.
            buffer_size: Legacy alias for the per-worker episode pool size.
            seed (int, optional): Reproducibility random seed.
            shuffle (bool, optional): Whether to shuffle the dataset across exhaustions. Defaults to True.
            depth_output_unit (str, optional): Physical unit depth maps are dequantized to ("m" or "mm").
                Defaults to "mm".
            repo_type: "dataset" (default) or "bucket" to stream from an HF Storage Bucket
                over ``hf://buckets/``.
            token: Authentication token used while streaming this dataset from
                the Hub. Pass a string token, ``True`` to require the locally
                stored token, ``False`` to disable authentication, or ``None``
                to use the Hugging Face Hub default. DataLoader workers use it to
                open their Parquet files.
            episode_pool_size: Complete episodes to mix per worker. If omitted, divide the
                default of 32 across workers, round to the nearest integer and use at least one.
            prefetch_episodes: Complete Parquet episodes to read ahead per worker. Zero disables read-ahead.
            video_cache_bytes: Maximum cached remote video bytes per worker process and cache
                configuration. Zero disables retention. None uses the Datasets cache default.
            byte_budget_gb: Legacy per-rank budget in GiB, divided across worker processes and
                applied to cached remote video bytes. Use video_cache_bytes for new datasets.
            decode_threads: Maximum parallel video decode calls per worker.
            decoded_queue_size: Maximum pending sample requests per worker's decode queue.
            video_decoder_cache_size: Legacy per-rank maximum open RGB decoders, divided across workers.
            max_num_shards: Deprecated. Worker ownership determines which shards to read.
        """
        super().__init__()
        if repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {repo_type!r}")

        if episode_pool_size is not None and buffer_size is not None:
            raise ValueError("Set episode_pool_size or buffer_size.")
        for name, value in (
            ("episode_pool_size", episode_pool_size),
            ("buffer_size", buffer_size),
            ("decode_threads", decode_threads),
            ("decoded_queue_size", decoded_queue_size),
            ("video_decoder_cache_size", video_decoder_cache_size),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive.")
        if prefetch_episodes < 0:
            raise ValueError("prefetch_episodes must be non-negative.")
        if video_cache_bytes is not None and video_cache_bytes < 0:
            raise ValueError("video_cache_bytes must be non-negative.")
        if byte_budget_gb is not None:
            if not math.isfinite(byte_budget_gb) or byte_budget_gb <= 0:
                raise ValueError("byte_budget_gb must be finite and positive.")
            if video_cache_bytes is not None:
                raise ValueError("Set video_cache_bytes or byte_budget_gb.")
        if max_num_shards is not None:
            warnings.warn(
                "max_num_shards is deprecated. Streaming workers now read their assigned episodes.",
                FutureWarning,
                stacklevel=2,
            )

        self.repo_id = repo_id
        self.repo_type = repo_type
        self._requested_root = Path(root) if root is not None else None
        self.root = self._requested_root if self._requested_root is not None else HF_LEROBOT_HOME / repo_id
        self.streaming_from_local = root is not None and self.repo_type == "dataset"

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision: str | None = revision if revision else CODEBASE_VERSION
        self.seed = seed
        self.shuffle = shuffle

        self.episode_pool_size = episode_pool_size if episode_pool_size is not None else buffer_size
        self.buffer_size = self.episode_pool_size or _DEFAULT_EPISODE_POOL_SIZE
        self.prefetch_episodes = prefetch_episodes
        self.video_cache_bytes = video_cache_bytes
        self.byte_budget_gb = byte_budget_gb
        self.decode_threads = decode_threads
        self.decoded_queue_size = decoded_queue_size
        self.video_decoder_cache_size = video_decoder_cache_size
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit

        if self._requested_root is not None:
            self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id,
            self._requested_root,
            self.revision,
            force_cache_sync=force_cache_sync,
            repo_type=self.repo_type,
            token=token,
        )
        self.root = self.meta.root
        self.revision = self.meta.revision
        self.meta.rescale_depth_stats(self._depth_output_unit)
        # Check version
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        self._depth_encoder_configs: dict[str, DepthEncoderConfig] = {
            vid_key: DepthEncoderConfig.from_video_info(self.meta.features[vid_key].get("info"))
            for vid_key in self.meta.depth_keys
        }

        # Input unit of each depth feature stored as raw images (dequantized separately from videos).
        self._image_depth_units: dict[str, str | None] = {
            key: (self.meta.features[key].get("info") or {}).get("depth_unit")
            for key in self.meta.depth_keys
            if key in self.meta.image_keys
        }

        self.delta_timestamps = None
        self.delta_indices = None

        if delta_timestamps is not None:
            self._validate_delta_timestamp_keys(delta_timestamps)  # raises ValueError if invalid
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self._parquet_token = None if self.streaming_from_local else token
        self._rank_episodes: list[int] | None = None
        self._spawned_rank: tuple[int, int] | None = None
        self._worker_source: tuple[list[int], datasets.IterableDataset] | None = None
        self._epoch = 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self._rank_episodes is None and torch.distributed.is_initialized():
            # A spawned worker has no process group, and a manually initialized group
            # may have no rank in the environment.
            state["_spawned_rank"] = (torch.distributed.get_rank(), torch.distributed.get_world_size())
        return state

    @cached_property
    def hf_dataset(self) -> datasets.IterableDataset:
        """Open the unfiltered source for a single consumer of the full dataset."""
        token_kwargs = {} if self._parquet_token is None else {"token": self._parquet_token}
        if self.streaming_from_local:
            dataset_path = str(self.root)
            data_files = "data/*/*.parquet"
        elif self.repo_type == "dataset":
            dataset_path = self.repo_id
            data_files = "data/*/*.parquet"
        else:
            dataset_path = "parquet"
            data_files = f"hf://buckets/{self.repo_id}/data/*/*.parquet"
        return load_dataset(
            dataset_path,
            split="train",
            streaming=True,
            data_files=data_files,
            revision=self.revision,
            **token_kwargs,
        )

    @property
    def num_shards(self) -> int:
        return self.hf_dataset.num_shards

    @cached_property
    def _selected_episodes(self) -> list[int]:
        selected = resolve_episode_indices(self.episodes, self.meta.total_episodes)
        return list(range(self.meta.total_episodes)) if selected is None else sorted(set(selected))

    @cached_property
    def _episode_frame_counts(self) -> dict[int, int]:
        if not self._selected_episodes:
            return {}
        boundaries = self.meta.episodes.select_columns(["dataset_from_index", "dataset_to_index"])[:]
        frame_counts = {
            episode: boundaries["dataset_to_index"][episode] - boundaries["dataset_from_index"][episode]
            for episode in self._selected_episodes
        }
        if any(count <= 0 for count in frame_counts.values()):
            raise ValueError("Streaming episode lengths must be positive")
        return frame_counts

    @property
    def num_frames(self) -> int:
        return self.meta.total_frames if self.episodes is None else sum(self._episode_frame_counts.values())

    @property
    def num_episodes(self) -> int:
        return len(self._selected_episodes)

    def set_data_parallel_shard(self, rank: int, world_size: int) -> None:
        """Set the data-parallel coordinate before starting DataLoader workers."""
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid streaming rank/world size: rank={rank}, world_size={world_size}")
        shards = _balanced_episode_shards(
            self._selected_episodes, self._episode_frame_counts, world_size=world_size
        )
        self._rank_episodes = shards[rank]

    @property
    def num_rank_episodes(self) -> int:
        return len(self._get_rank_episodes())

    def _get_rank_episodes(self) -> list[int]:
        if self._rank_episodes is not None:
            return self._rank_episodes
        if torch.distributed.is_initialized():
            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        elif self._spawned_rank is not None:
            rank, world_size = self._spawned_rank
        else:
            rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
        if rank == 0 and world_size == 1:
            return self._selected_episodes
        self.set_data_parallel_shard(rank, world_size)
        assert self._rank_episodes is not None
        return self._rank_episodes

    def _episodes_for_worker(self, worker_id: int, num_workers: int) -> list[int]:
        rank_episodes = self._get_rank_episodes()
        if num_workers == 1:
            return rank_episodes
        shards = _balanced_episode_shards(rank_episodes, self._episode_frame_counts, world_size=num_workers)
        return shards[worker_id]

    def _stream_root(self) -> str:
        if self.streaming_from_local:
            return str(self.root)
        if self.repo_type == "dataset" and self.revision:
            return f"{self.meta.url_root}@{quote(self.revision, safe='')}"
        return self.meta.url_root

    def set_epoch(self, epoch: int) -> None:
        """Set the shuffle epoch before constructing a new iterator or spawning workers."""
        if epoch < 0:
            raise ValueError(f"Streaming epoch must be non-negative, got {epoch}")
        self._epoch = epoch

    def _owned_parquet(self, episodes: list[int]) -> datasets.IterableDataset:
        if len(episodes) == self.meta.total_episodes:
            return self.hf_dataset
        if self._worker_source is None or self._worker_source[0] != episodes:
            root = self._stream_root()
            paths = sorted({f"{root}/{self.meta.get_data_file_path(episode)}" for episode in episodes})
            ds = load_dataset(
                "parquet",
                split="train",
                streaming=True,
                data_files=paths,
                filters=[("episode_index", "in", episodes)],
                token=self._parquet_token,
            )
            self._worker_source = (episodes, ds)
        return self._worker_source[1]

    @property
    def fps(self):
        return self.meta.fps

    @property
    def depth_output_unit(self) -> str:
        """Physical unit (``"m"`` or ``"mm"``) depth maps are returned in on read."""
        return self._depth_output_unit

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker is not None else (0, 1)
        owned_episodes = self._episodes_for_worker(worker_id, num_workers)
        if not owned_episodes:
            return
        seed = self.seed + self._epoch if self.shuffle else None
        self._epoch += 1
        ds = self._owned_parquet(owned_episodes).batch(by_column="episode_index")
        if seed is not None:
            # Shuffle the files. The sample pool mixes frames without a second episode buffer.
            ds = ds.shuffle(seed=seed, buffer_size=1)

        with (
            closing(self._prefetch_episodes(self._iter_complete_episodes(ds, owned_episodes))) as episodes,
            closing(
                _StreamingVideoDecoderPool(
                    self._decoder_pool_capacity_for_worker(num_workers),
                    cache_bytes=self._video_cache_bytes_for_worker(num_workers),
                )
            ) as decoder_pool,
        ):
            samples = self._mix_episodes(episodes, self._episode_pool_size_for_worker(num_workers), seed)
            executor = ThreadPoolExecutor(self.decode_threads)
            try:
                if (
                    self.delta_indices is None
                    and self.image_transforms is None
                    and not self.meta.depth_keys
                    and self.meta.video_keys
                    and self._uses_default_video_query()
                ):
                    yield from self._decode_rgb_groups(samples, executor, decoder_pool)
                else:
                    yield from self._decode_samples(samples, executor, decoder_pool)
            finally:
                # Wait for leased decoders to return before closing their sources.
                executor.shutdown(cancel_futures=True)

    def _episode_pool_size_for_worker(self, num_workers: int) -> int:
        if self.episode_pool_size is not None:
            return self.episode_pool_size
        return max(1, round(_DEFAULT_EPISODE_POOL_SIZE / max(1, num_workers)))

    def _decoder_pool_capacity_for_worker(self, num_workers: int) -> int:
        if self.video_decoder_cache_size is not None:
            return max(1, self.video_decoder_cache_size // max(1, num_workers))
        return _DECODER_POOL_CAPACITY

    def _video_cache_bytes_for_worker(self, num_workers: int) -> int | None:
        if self.byte_budget_gb is not None:
            numerator, denominator = self.byte_budget_gb.as_integer_ratio()
            return (numerator << 30) // (denominator * max(1, num_workers))
        return self.video_cache_bytes

    def log_worker_limits(self, num_workers: int, dataloader_prefetch_factor: int | None) -> None:
        """Log per-worker limits and the total if every worker on the rank is active."""
        workers = max(1, num_workers)
        pool_size = self._episode_pool_size_for_worker(workers)
        logger.info(
            "Streaming episode pool: %d episodes per worker, up to %d per rank. Worker count per rank: %d.",
            pool_size,
            pool_size * workers,
            workers,
        )
        logger.info(
            "Streaming episode read-ahead: %d Parquet episodes per worker, up to %d per rank.",
            self.prefetch_episodes,
            self.prefetch_episodes * workers,
        )
        logger.info(
            "Streaming decode queue: %d samples per worker, up to %d per rank.",
            self.decoded_queue_size,
            self.decoded_queue_size * workers,
        )
        prefetch_batches = dataloader_prefetch_factor or 0
        logger.info(
            "DataLoader prefetch: %d batches per worker, up to %d per rank.",
            prefetch_batches,
            prefetch_batches * workers,
        )
        if self.streaming_from_local or not self.meta.video_keys:
            return
        from datasets.filesystems.caching import DEFAULT_MAX_MEMORY

        cache_bytes = self._video_cache_bytes_for_worker(workers)
        if cache_bytes is None:
            cache_bytes = DEFAULT_MAX_MEMORY
        logger.info(
            "Streaming remote video cache: %s bytes per worker process and cache configuration, "
            "up to %s cached bytes per rank across worker processes.",
            f"{cache_bytes:,}",
            f"{cache_bytes * workers:,}",
        )
        if self.byte_budget_gb is not None:
            logger.info(
                "Converted the legacy streaming_byte_budget_gb value to a per-process remote video "
                "cache cap. Use streaming_video_cache_bytes for new runs."
            )

    def _prefetch_episodes(self, episodes: Iterator[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        if not self.prefetch_episodes:
            yield from episodes
            return

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            pending = deque(executor.submit(next, episodes, None) for _ in range(self.prefetch_episodes))
            while pending:
                episode = pending.popleft().result()
                if episode is None:
                    break
                pending.append(executor.submit(next, episodes, None))
                yield episode
        finally:
            executor.shutdown(cancel_futures=True)

    def _iter_complete_episodes(
        self, ds: datasets.IterableDataset, owned_episodes: list[int]
    ) -> Iterator[dict[str, Any]]:
        remaining = set(owned_episodes)
        # The explicit batched API consumes this worker's complete source. The default
        # Datasets iterator would again partition its files among DataLoader workers.
        for batch in ds.iter(batch_size=1):
            episode = {key: values[0] for key, values in batch.items()}
            episode_index = episode["episode_index"][0]
            if episode_index not in remaining:
                raise ValueError(f"Unexpected or repeated streaming episode {episode_index}")
            expected = self._episode_frame_counts[episode_index]
            actual = len(episode["episode_index"])
            if actual != expected:
                raise ValueError(
                    f"Streaming episode {episode_index}: expected {expected} rows, found {actual}"
                )
            remaining.remove(episode_index)
            yield episode
        if remaining:
            raise ValueError(
                f"Missing {len(remaining)} streaming episodes, including episode {min(remaining)}"
            )

    def _mix_episodes(
        self, episodes: Iterator[dict[str, Any]], pool_size: int, seed: int | None
    ) -> Iterator[dict[str, Any]]:
        for pool in batched(episodes, pool_size):
            columns = {key: [episode[key] for episode in pool] for key in pool[0]}
            samples = self._sample_pool_with_delta(columns, shuffle_seed=seed)
            for values in zip(*samples.values(), strict=True):
                yield dict(zip(samples, values, strict=True))

    def _uses_default_video_query(self) -> bool:
        return (
            getattr(self.query_frames, "__func__", None) is StreamingLeRobotDataset.query_frames
            and getattr(self._query_videos, "__func__", None) is StreamingLeRobotDataset._query_videos
        )

    def _decode_samples(
        self,
        samples: Iterator[dict[str, Any]],
        executor: ThreadPoolExecutor,
        decoder_pool: _StreamingVideoDecoderPool,
    ) -> Iterator[dict[str, Any]]:
        def read_rgb_video(video_path: str, timestamps: list[float]) -> torch.Tensor:
            with decoder_pool.lease(video_path) as (decoder, video_file):
                return _decode_torchcodec_frames(
                    decoder, timestamps, self.tolerance_s, video_file, return_uint8=self._return_uint8
                )

        def query_frames_direct(item: dict[str, Any]) -> dict[str, Any]:
            # Match the map merge for subclasses that return updated columns only.
            return {**item, **self.query_frames(item.copy())}

        query_frames: Callable[[dict[str, Any]], dict[str, Any]] = query_frames_direct
        if self._uses_default_video_query():
            query_videos = partial(self._query_videos, read_rgb_video=read_rgb_video)
            query_frames = partial(self._query_frames, query_videos=query_videos)

        pending: deque[Future[dict[str, Any]]] = deque()
        for item in samples:
            pending.append(executor.submit(query_frames, item))
            if len(pending) == self.decoded_queue_size:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()

    def _sample_pool_with_delta(
        self, episodes_batch: dict[str, list[list[Any]]], *, shuffle_seed: int | None = 42
    ) -> dict[str, list[Any]]:
        episode_batches: list[dict[str, Any]] = [
            {col: episodes_batch[col][ep] for col in episodes_batch}
            for ep in range(len(episodes_batch["episode_index"]))
        ]
        for episode_batch in episode_batches:
            for key, values in episode_batch.items():
                if key == "task" or key in LANGUAGE_COLUMNS:
                    continue
                first = values[0]
                if first is None or isinstance(first, (str, bytes, dict)):
                    continue
                if isinstance(first, PILImage.Image):
                    episode_batch[key] = torch.stack([pil_to_chw_tensor(image) for image in values])
                    continue
                try:
                    array = np.asarray(values)
                except ValueError:
                    # Variable-length annotations can be ragged.
                    continue
                if array.dtype.kind in "biuf":
                    dtype = torch.get_default_dtype() if array.dtype.kind == "f" else None
                    episode_batch[key] = torch.as_tensor(array, dtype=dtype)

            if self.delta_indices is not None:
                query_results, paddings = self._get_delta_frames(episode_batch)
                episode_batch.update(query_results)
                episode_batch.update(paddings)

        samples = [
            (ep, frame)
            for ep in range(len(episode_batches))
            for frame in range(len(episode_batches[ep]["episode_index"]))
        ]
        if shuffle_seed is not None:
            pool_key = "|".join(str(ep[0]) for ep in episodes_batch["episode_index"])
            if shuffle_seed != 42:
                pool_key = f"{shuffle_seed}:{pool_key}"
            random.Random(pool_key).shuffle(samples)

        return {col: [episode_batches[ep][col][frame] for ep, frame in samples] for col in episode_batches[0]}

    def _decode_rgb_groups(
        self,
        samples: Iterator[dict[str, Any]],
        executor: ThreadPoolExecutor,
        decoder_pool: _StreamingVideoDecoderPool,
    ) -> Iterator[dict[str, Any]]:
        group_size = min(_RGB_DECODE_GROUP_SIZE, self.decoded_queue_size)
        groups = batched(samples, group_size)
        # Count samples, including groups that have ready results, against the same queue limit.
        max_pending_groups = self.decoded_queue_size // group_size
        pending = deque(
            self._submit_rgb_group(group, executor, decoder_pool)
            for group in islice(groups, max_pending_groups)
        )
        while pending:
            yield from pending.popleft()
            if next_group := next(groups, None):
                pending.append(self._submit_rgb_group(next_group, executor, decoder_pool))

    def _submit_rgb_group(
        self,
        group: tuple[dict[str, Any], ...],
        executor: ThreadPoolExecutor,
        decoder_pool: _StreamingVideoDecoderPool,
    ) -> Iterator[dict[str, Any]]:
        requests_by_path: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
        try:
            for row, sample in enumerate(group):
                episode = int(sample["episode_index"])
                frame = int(sample["frame_index"])
                video_queries, _ = self._get_video_queries(episode, frame)
                for key, (query_timestamp,) in video_queries.items():
                    path = self._video_path(episode, key)
                    requests_by_path[path].append((row, key, query_timestamp))
        except Exception:
            # Report metadata errors in sample order with the native reader.
            return self._decode_samples(iter(group), executor, decoder_pool)

        frame_reads: list[dict[str, tuple[Future[list[torch.Tensor]], int, bool]]] = [{} for _ in group]
        for path, requests in requests_by_path.items():
            timestamps = [timestamp for _, _, timestamp in requests]
            future = executor.submit(self._read_rgb_group, decoder_pool, path, timestamps)
            is_grouped = len(requests) > 1
            for offset, (row, key, _) in enumerate(requests):
                frame_reads[row][key] = (future, offset, is_grouped)

        def ordered_results() -> Iterator[dict[str, Any]]:
            video_keys = self.meta.video_keys
            for row, (sample, reads) in enumerate(zip(group, frame_reads, strict=True)):
                task_idx = int(sample["task_index"])
                result = sample.copy()
                for key in video_keys:
                    future, offset, is_grouped = reads[key]
                    try:
                        result[key] = future.result()[offset]
                    except Exception:
                        if not is_grouped:
                            raise
                        # A grouped read can fail on a later frame. Recover the valid prefix
                        # and report the failing sample via the existing per-sample path.
                        yield from self._decode_samples(iter(group[row:]), executor, decoder_pool)
                        return
                result["task"] = self.meta.tasks.iloc[task_idx].name
                yield result

        return ordered_results()

    def _read_rgb_group(
        self, decoder_pool: _StreamingVideoDecoderPool, path: str, timestamps: list[float]
    ) -> list[torch.Tensor]:
        with decoder_pool.lease(path) as (decoder, video_file):
            if len(timestamps) == 1:
                return [
                    _decode_torchcodec_frames(
                        decoder, timestamps, self.tolerance_s, video_file, return_uint8=self._return_uint8
                    ).squeeze(0)
                ]

            fps = decoder.metadata.average_fps
            frame_indices = [round(timestamp * fps) for timestamp in timestamps]
            unique_indices = sorted(set(frame_indices))
            batch = decoder.get_frames_at(indices=unique_indices)
            offsets = {index: offset for offset, index in enumerate(unique_indices)}

            # A singleton request must pass tolerance against its own frame.
            # Matching against the whole batch could accept a neighboring frame.
            frames = []
            for timestamp, index in zip(timestamps, frame_indices, strict=True):
                offset = offsets[index]
                frame = _match_torchcodec_frames(
                    [timestamp],
                    batch.data[offset : offset + 1],
                    batch.pts_seconds[offset : offset + 1],
                    self.tolerance_s,
                    video_file,
                    return_uint8=self._return_uint8,
                )
                frames.append(frame.squeeze(0))
            return frames

    def query_frames(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._query_frames(item, self._query_videos)

    def _query_frames(
        self,
        item: dict[str, Any],
        query_videos: Callable[[dict[str, list[float]], int], dict[str, torch.Tensor]],
    ) -> dict[str, Any]:
        ep_idx = int(item["episode_index"])
        task_idx = int(item["task_index"])
        result = item.copy()
        if self.meta.video_keys:
            timestamps, padding = self._get_video_queries(ep_idx, int(item["frame_index"]))
            result.update(query_videos(timestamps, ep_idx))
            result.update(padding)

        if self.image_transforms is not None:
            for key in self.meta.camera_keys:
                if key in result and key not in self.meta.depth_keys:
                    result[key] = self.image_transforms(result[key])

        # Convert raw-image depth features to the output unit (video depth is already converted).
        for key, stored_unit in self._image_depth_units.items():
            if key in result and stored_unit is not None and stored_unit != self._depth_output_unit:
                result[key] = (
                    result[key] * MM_PER_METRE
                    if stored_unit == DEPTH_METER_UNIT
                    else result[key] / MM_PER_METRE
                )

        result["task"] = self.meta.tasks.iloc[task_idx].name

        return result

    def _get_video_queries(
        self, episode: int, frame: int
    ) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
        episode_metadata = self.meta.episodes[episode]
        last_frame = episode_metadata["length"] - 1
        timestamps = {}
        padding = {}
        for key in self.meta.video_keys:
            offsets = self.delta_indices.get(key) if self.delta_indices is not None else None
            start = episode_metadata[f"videos/{key}/from_timestamp"]
            if offsets is None:
                timestamps[key] = [start + frame / self.fps]
                continue

            requested = [frame + offset for offset in offsets]
            clipped = [max(0, min(index, last_frame)) for index in requested]
            # Add each camera's MP4 offset in double precision after clipping in episode frame units.
            timestamps[key] = [start + index / self.fps for index in clipped]
            padding[f"{key}_is_pad"] = torch.tensor(
                [index != bound for index, bound in zip(requested, clipped, strict=True)], dtype=torch.bool
            )
        return timestamps, padding

    def _video_path(self, ep_idx: int, video_key: str) -> str:
        video_path = f"{self._stream_root()}/{self.meta.get_video_file_path(ep_idx, video_key)}"
        # sharedblockcache shares cached blocks safely between decode threads.
        return video_path if self.streaming_from_local else f"sharedblockcache::{video_path}"

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        ep_idx: int,
        *,
        read_rgb_video: Callable[[str, list[float]], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        item = {}
        worker = torch.utils.data.get_worker_info()
        cache_bytes = self._video_cache_bytes_for_worker(worker.num_workers if worker is not None else 1)
        for video_key, query_ts in query_timestamps.items():
            video_path = self._video_path(ep_idx, video_key)
            if video_key in self.meta.depth_keys:
                # Depth maps are 12-bit quantized and only decodable via pyav; dequantize back
                # to physical units to match the non-streaming reader.
                with _open_streaming_video(video_path, cache_bytes) as video_file:
                    frames = decode_video_frames(
                        video_file,
                        query_ts,
                        self.tolerance_s,
                        backend="pyav",
                        return_uint8=False,
                        is_depth=True,
                    )
                depth_encoder = self._depth_encoder_configs[video_key]
                frames = dequantize_depth(
                    frames,
                    depth_min=depth_encoder.depth_min,
                    depth_max=depth_encoder.depth_max,
                    shift=depth_encoder.shift,
                    use_log=depth_encoder.use_log,
                    output_unit=self._depth_output_unit,
                )
            elif read_rgb_video is not None:
                frames = read_rgb_video(video_path, query_ts)
            else:
                with _open_streaming_video(video_path, cache_bytes) as video_file:
                    frames = decode_video_frames_torchcodec(
                        video_file,
                        query_ts,
                        self.tolerance_s,
                        return_uint8=self._return_uint8,
                    )

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames

        return item

    def _get_delta_frames(
        self, episode_batch: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Gather each non-video window and its padding mask within one episode."""
        if self.delta_indices is None:
            raise RuntimeError("Delta frames were requested but no delta_timestamps are configured.")

        num_steps = len(episode_batch["episode_index"])
        anchors = torch.arange(num_steps).unsqueeze(1)
        frames = {}
        padding = {}
        for key, offsets in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue
            requested = anchors + torch.tensor(offsets, dtype=torch.long)
            clipped = requested.clamp(0, num_steps - 1)
            values = episode_batch[key]
            if isinstance(values, torch.Tensor):
                frames[key] = values[clipped]
            else:
                frames[key] = [[values[index] for index in window] for window in clipped.tolist()]
            padding[f"{key}_is_pad"] = requested != clipped
        return frames, padding

    def _validate_delta_timestamp_keys(self, delta_timestamps: dict[str, list[float]] | None) -> None:
        """
        Validate that all keys in delta_timestamps correspond to actual features in the dataset.

        Raises:
            ValueError: If any delta timestamp key doesn't correspond to a dataset feature.
        """
        if delta_timestamps is None:
            return

        # Get all available feature keys from the dataset metadata
        available_features = set(self.meta.features.keys())

        # Get all keys from delta_timestamps
        delta_keys = set(delta_timestamps.keys())

        # Find any keys that don't correspond to features
        invalid_keys = delta_keys - available_features

        if invalid_keys:
            raise ValueError(
                f"The following delta_timestamp keys do not correspond to dataset features: {invalid_keys}. "
                f"Available features are: {sorted(available_features)}"
            )
