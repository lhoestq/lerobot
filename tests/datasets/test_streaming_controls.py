#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from contextlib import closing, nullcontext
from threading import Event
from types import SimpleNamespace

import draccus
import fsspec
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import factory, streaming_dataset as streaming
from lerobot.datasets.storage import DEFAULT_STORAGE_FORMAT
from lerobot.datasets.video_utils import _make_streaming_video_opener, _open_streaming_video
from lerobot.policies.act.configuration_act import ACTConfig

pytestmark = pytest.mark.dataset


@pytest.fixture
def streaming_factory(tmp_path, monkeypatch):
    metadata = SimpleNamespace(
        root=tmp_path,
        revision=streaming.CODEBASE_VERSION,
        _version=streaming.CODEBASE_VERSION,
        fps=30,
        features={},
        total_episodes=512,
        storage_format=DEFAULT_STORAGE_FORMAT,
        depth_keys=[],
        image_keys=[],
        camera_keys=[],
        video_keys=[],
        rescale_depth_stats=lambda _: None,
    )
    monkeypatch.setattr(factory, "load_dataset_metadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(streaming, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(streaming, "load_dataset", lambda *args, **kwargs: SimpleNamespace(num_shards=8))

    def make_config(*args):
        return draccus.parse(
            TrainPipelineConfig,
            args=[
                "--dataset.repo_id=test/stream",
                "--dataset.streaming=true",
                f"--policy.type={ACTConfig.get_choice_name(ACTConfig)}",
                "--policy.device=cpu",
                *args,
            ],
        )

    return make_config


@pytest.mark.parametrize("num_workers", [0, 3, 4])
def test_streaming_cli_pool_size_keeps_per_worker_scope(streaming_factory, num_workers, caplog):
    cfg = streaming_factory(
        f"--num_workers={num_workers}",
        "--dataset.streaming_episode_pool_size=128",
        "--dataset.streaming_prefetch_episodes=5",
        "--dataset.streaming_video_cache_bytes=8192",
        "--prefetch_factor=2",
        "--seed=91",
    )
    dataset = factory.make_dataset(cfg)

    assert cfg.prefetch_factor == 2
    assert dataset._episode_pool_size_for_worker(num_workers) == 128
    assert dataset.prefetch_episodes == 5
    assert dataset._video_cache_bytes_for_worker(num_workers) == 8192
    assert dataset.seed == 91

    with caplog.at_level("INFO"):
        dataset.log_worker_limits(num_workers, cfg.prefetch_factor if num_workers else None)
    assert f"128 episodes per worker, up to {128 * max(1, num_workers)} per rank" in caplog.text
    assert f"DataLoader prefetch: {2 if num_workers else 0} batches per worker" in caplog.text


@pytest.mark.parametrize("num_workers, pool_size", [(0, 32), (3, 11), (4, 8), (64, 1)])
def test_streaming_default_pool_and_legacy_byte_budget(streaming_factory, num_workers, pool_size):
    cfg = streaming_factory(
        f"--num_workers={num_workers}",
        "--dataset.streaming_byte_budget_gb=8",
        "--dataset.streaming_decoded_queue_size=8",
        "--dataset.streaming_sampling_strategy=remaining",
        "--dataset.streaming_native_http_subranges=1",
        "--dataset.streaming_sidecar_lock_timeout_s=1800",
    )
    dataset = factory.make_dataset(cfg)
    assert dataset._episode_pool_size_for_worker(num_workers) == pool_size
    assert dataset._video_cache_bytes_for_worker(num_workers) == (8 << 30) // max(1, num_workers)
    assert dataset.decoded_queue_size == 8


@pytest.mark.parametrize("prefetch", [0, 3])
def test_episode_read_ahead_preserves_order_and_uses_its_own_limit(streaming_factory, prefetch):
    dataset = factory.make_dataset(streaming_factory(f"--dataset.streaming_prefetch_episodes={prefetch}"))
    complete = Event()
    read = []

    def episodes():
        for episode in range(10):
            read.append(episode)
            if len(read) == prefetch + 1:
                complete.set()
            yield {"episode_index": episode}

    with closing(dataset._prefetch_episodes(episodes())) as iterator:
        assert next(iterator) == {"episode_index": 0}
        assert complete.wait(timeout=5)
        assert read == list(range(prefetch + 1))
        assert list(iterator) == [{"episode_index": episode} for episode in range(1, 10)]


def test_remote_video_byte_limit_is_applied_to_the_shared_datasets_store():
    pytest.importorskip("datasets.filesystems.caching", reason="Datasets shared block cache is required")

    fs = fsspec.filesystem("memory")
    for name in ("first", "second"):
        fs.pipe_file(f"/streaming-controls/{name}", b"1234")

    first = _open_streaming_video("sharedblockcache::memory:///streaming-controls/first", cache_bytes=5)
    second = _open_streaming_video("sharedblockcache::memory:///streaming-controls/second", cache_bytes=5)
    off = _open_streaming_video("sharedblockcache::memory:///streaming-controls/second", cache_bytes=0)

    assert first.fs.store is second.fs.store
    assert first.fs.store is not off.fs.store
    for source in (first, second, off):
        with source as file:
            assert file.read() == b"1234"
    assert first.fs.store.stats()["max_memory"] == 5
    assert first.fs.store.stats()["cached_bytes"] == 4
    assert off.fs.store.stats()["cached_bytes"] == 0


def test_explicit_hub_tokens_use_private_video_stores(monkeypatch):
    pytest.importorskip("datasets.filesystems.caching", reason="Datasets shared block cache is required")
    monkeypatch.setenv("HF_TOKEN", "test-local-hub-token")
    path = "sharedblockcache::hf://datasets/test/stream@v3.0/video.mp4"

    open_file = _make_streaming_video_opener(cache_bytes=5, token="test-explicit-hub-token")
    first = open_file(path)
    second = open_file(path)
    assert first.fs.target.token == "test-explicit-hub-token"
    assert first.fs.store is second.fs.store
    assert first.fs is not second.fs
    assert first.fs.store.max_memory == 5

    other_tokens = [
        ("test-other-hub-token", "test-other-hub-token"),
        (False, False),
        (True, "test-local-hub-token"),
        (None, None),
    ]
    stores = [first.fs.store]
    for token, expected in other_tokens:
        other = _make_streaming_video_opener(cache_bytes=5, token=token)(path)
        assert other.fs.target.token == expected
        assert all(other.fs.store is not store for store in stores)
        stores.append(other.fs.store)


def test_remote_dataset_forwards_video_token_on_direct_read(streaming_factory, monkeypatch):
    pytest.importorskip("datasets.filesystems.caching", reason="Datasets shared block cache is required")
    dataset = streaming.StreamingLeRobotDataset("test/stream", token="test-explicit-hub-token")
    path = "sharedblockcache::hf://datasets/test/stream@v3.0/video.mp4"
    monkeypatch.setattr(dataset, "_video_path", lambda *_: path)

    fsspec_open = fsspec.open
    opened = []

    def open_without_io(*args, **kwargs):
        file = fsspec_open(*args, **kwargs)
        opened.append(file)
        return nullcontext(file)

    monkeypatch.setattr(fsspec, "open", open_without_io)
    monkeypatch.setattr(streaming, "decode_video_frames_torchcodec", lambda *_, **__: torch.zeros(1, 3, 2, 2))

    assert dataset._query_videos({"camera": [0.0]}, 0)["camera"].shape == (3, 2, 2)
    assert len(opened) == 1
    assert opened[0].fs.target.token == "test-explicit-hub-token"
