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

from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("torchcodec", reason="torchcodec is required (install lerobot[dataset])")

from torchcodec.decoders import VideoDecoder

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.video_utils import FrameTimestampError, _StreamingVideoDecoderPool
from tests.fixtures.constants import DUMMY_REPO_ID

pytestmark = pytest.mark.dataset

CAMERAS = ("observation.images.front", "observation.images.wrist")
IMAGE_SHAPE = (32, 48, 3)
EPISODE_LENGTHS = (11, 8)


@pytest.fixture(scope="module")
def rgb_stream_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("streaming_rgb") / "dataset"
    features = {
        key: {"dtype": "video", "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]}
        for key in CAMERAS
    }
    features["action"] = {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}
    writer = LeRobotDataset.create(
        DUMMY_REPO_ID,
        fps=25,
        features=features,
        root=root,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", g=4),
        encoder_threads=1,
    )

    for episode_index, length in enumerate(EPISODE_LENGTHS):
        for frame_index in range(length):
            frame = {
                "action": np.array([episode_index, frame_index], dtype=np.float32),
                "task": f"Task {episode_index}",
            }
            for camera_index, key in enumerate(CAMERAS):
                value = 10 + 80 * camera_index + 40 * episode_index + 3 * frame_index
                frame[key] = np.full(IMAGE_SHAPE, value, dtype=np.uint8)
            writer.add_frame(frame)
        writer.save_episode(parallel_encoding=False)
    writer.finalize()
    return root


@pytest.mark.parametrize("return_uint8", [True, False])
def test_grouped_stream_matches_single_frame_reads(rgb_stream_root, return_uint8, monkeypatch):
    calls = []
    get_frames_at = VideoDecoder.get_frames_at

    def record_frames_at(self, *, indices):
        calls.append(len(indices))
        return get_frames_at(self, indices=indices)

    monkeypatch.setattr(VideoDecoder, "get_frames_at", record_frames_at)
    # The absolute path selects the local Parquet data in the Datasets loader.
    kwargs = {
        "repo_id": str(rgb_stream_root),
        "root": rgb_stream_root,
        "buffer_size": 2,
        "return_uint8": return_uint8,
    }
    grouped = StreamingLeRobotDataset(**kwargs)
    singleton = StreamingLeRobotDataset(**kwargs, image_transforms=lambda frame: frame)

    grouped_batches = list(DataLoader(grouped, batch_size=5))
    frame_count = sum(EPISODE_LENGTHS)
    assert sum(calls) == frame_count * len(CAMERAS)
    assert max(calls) > 1
    assert len(calls) < frame_count * len(CAMERAS)
    calls.clear()

    singleton_batches = list(DataLoader(singleton, batch_size=5))
    assert calls == [1] * (frame_count * len(CAMERAS))
    assert [len(batch["index"]) for batch in grouped_batches] == [5, 5, 5, 4]
    assert sorted(index for batch in grouped_batches for index in batch["index"].tolist()) == list(
        range(frame_count)
    )
    dtype = torch.uint8 if return_uint8 else torch.float32
    for got, expected in zip(grouped_batches, singleton_batches, strict=True):
        assert list(got) == list(expected)
        assert got["task"] == expected["task"]
        for key in CAMERAS:
            assert got[key].dtype == dtype
            assert got[key].shape == (len(got["index"]), 3, *IMAGE_SHAPE[:2])
        for key in got.keys() - {"task"}:
            torch.testing.assert_close(got[key], expected[key], rtol=0, atol=0)


def test_grouped_reads_share_duplicate_frame_indices(rgb_stream_root, monkeypatch):
    dataset = StreamingLeRobotDataset(str(rgb_stream_root), root=rgb_stream_root, return_uint8=True)
    path = rgb_stream_root / dataset.meta.get_video_file_path(0, CAMERAS[0])
    calls = []
    get_frames_at = VideoDecoder.get_frames_at

    def record_frames_at(self, *, indices):
        calls.append(list(indices))
        return get_frames_at(self, indices=indices)

    monkeypatch.setattr(VideoDecoder, "get_frames_at", record_frames_at)
    timestamps = [2 / 25, 0, 2 / 25]
    with closing(_StreamingVideoDecoderPool(capacity=1)) as pool:
        frames = dataset._read_rgb_group(pool, str(path), timestamps)

    assert calls == [[0, 2]]
    torch.testing.assert_close(frames[0], frames[2], rtol=0, atol=0)
    assert not torch.equal(frames[0], frames[1])
    frames[0].zero_()
    assert torch.count_nonzero(frames[2]) > 0


def test_grouped_reads_check_tolerance_for_each_frame(rgb_stream_root, monkeypatch):
    dataset = StreamingLeRobotDataset(str(rgb_stream_root), root=rgb_stream_root)
    path = rgb_stream_root / dataset.meta.get_video_file_path(0, CAMERAS[0])
    get_frames_at = VideoDecoder.get_frames_at

    def get_swapped_frames(self, *, indices):
        # Each requested timestamp matches the other frame's presentation timestamp.
        return get_frames_at(self, indices=[1 - index for index in indices])

    monkeypatch.setattr(VideoDecoder, "get_frames_at", get_swapped_frames)
    with closing(_StreamingVideoDecoderPool(capacity=1)) as pool, pytest.raises(FrameTimestampError):
        dataset._read_rgb_group(pool, str(path), [0, 1 / 25])


def test_grouped_stream_yields_valid_prefix_before_decode_error(rgb_stream_root, monkeypatch):
    kwargs = {"repo_id": str(rgb_stream_root), "root": rgb_stream_root, "buffer_size": 2}
    singleton = StreamingLeRobotDataset(**kwargs, image_transforms=lambda frame: frame)
    order = [sample["index"] for sample in singleton]
    valid_prefix_length = 5
    bad_index = order[valid_prefix_length]
    other_index = (bad_index + 1) % sum(EPISODE_LENGTHS)
    get_frames_at = VideoDecoder.get_frames_at

    def get_wrong_frame(self, *, indices):
        return get_frames_at(
            self, indices=[other_index if index == bad_index else index for index in indices]
        )

    monkeypatch.setattr(VideoDecoder, "get_frames_at", get_wrong_frame)
    grouped = StreamingLeRobotDataset(**kwargs)
    with closing(iter(grouped)) as samples:
        assert [next(samples)["index"] for _ in range(valid_prefix_length)] == order[:valid_prefix_length]
        with pytest.raises(FrameTimestampError):
            next(samples)
