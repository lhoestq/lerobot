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

import hashlib
import json
import os
import subprocess
import sys
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("torchcodec", reason="torchcodec is required (install lerobot[dataset])")

import pyarrow.parquet as pq

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.distributed import ParallelDims
from lerobot.scripts.lerobot_train import _iter_training_batches, make_dataloaders
from tests.fixtures.constants import DUMMY_REPO_ID

pytestmark = pytest.mark.dataset

CAMERA = "observation.images.front"
LENGTHS = (11, 8, 4, 9, 12, 3, 5, 1, 7, 13, 6)
SELECTED = [9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 3]
RANK_EPISODES = ({1, 3, 6, 7, 9}, {0, 2, 4, 5, 8})
WORKER_EPISODES = (({9}, {3}, {1}, {6, 7}), ({4}, {0}, {8}, {2, 5}))


class _TaggedStream(StreamingLeRobotDataset):
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        for sample in super().__iter__():
            yield {**sample, "worker": worker.id if worker is not None else 0}


@pytest.fixture(scope="module")
def packed_video_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("stream_ownership") / "dataset"
    shape = (32, 48, 3)
    writer = LeRobotDataset.create(
        DUMMY_REPO_ID,
        fps=25,
        features={
            CAMERA: {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["episode", "frame"]},
        },
        root=root,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", g=4),
        encoder_threads=1,
    )
    for episode, length in enumerate(LENGTHS):
        for frame in range(length):
            writer.add_frame(
                {
                    CAMERA: np.full(shape, 20 + 10 * episode + frame, dtype=np.uint8),
                    "action": np.array([episode, frame], dtype=np.float32),
                    "task": f"Episode {episode}",
                }
            )
        writer.save_episode(parallel_encoding=False)
    writer.finalize()

    # The same file contains episodes owned by different ranks and workers. Individual
    # episodes also cross row groups, including row groups shared by two episodes.
    (path,) = (root / "data").glob("*/*.parquet")
    table = pq.read_table(path)
    pq.write_table(table, path, row_group_size=3)
    return root


def _pixels(image: torch.Tensor) -> str:
    return hashlib.sha256(image.numpy().tobytes()).hexdigest()


def _collect_rank(root: Path, output: Path, num_workers: int, batch_size: int, repeat_steps: int) -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        parallel_dims = ParallelDims(2, 1, 1, 1, 2, "cpu")
        dataset = _TaggedStream(
            DUMMY_REPO_ID,
            root=root,
            episodes=SELECTED,
            buffer_size=8,
            seed=83,
            return_uint8=True,
        )
        cfg = SimpleNamespace(
            trainable_config=SimpleNamespace(),
            dataset=SimpleNamespace(streaming=True),
            num_workers=num_workers,
            batch_size=batch_size,
            seed=83,
            persistent_workers=num_workers > 1,
            prefetch_factor=2,
            dataloader_multiprocessing_context="spawn",
        )
        loader, _ = make_dataloaders(cfg, dataset, None, step=0, parallel_dims=parallel_dims)
        lengths = []
        samples = []
        for batch in loader:
            lengths.append(len(batch["index"]))
            for offset, index in enumerate(batch["index"].tolist()):
                samples.append(
                    [
                        index,
                        int(batch["episode_index"][offset]),
                        int(batch["worker"][offset]),
                        batch["action"][offset].tolist(),
                        _pixels(batch[CAMERA][offset]),
                    ]
                )

        # The trainer cycles each rank locally. A distributed backward call proves that
        # both ranks continue through their different finite tail lengths.
        batches = _iter_training_batches(loader)
        model = DistributedDataParallel(torch.nn.Linear(2, 1))
        repeated_indices = []
        for _ in range(repeat_steps):
            batch = next(batches)
            repeated_indices.extend(batch["index"].tolist())
            model(batch["action"]).mean().backward()
            model.zero_grad()
        (output / f"{rank}.json").write_text(json.dumps([lengths, samples, repeated_indices]))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires the Gloo distributed backend")
@pytest.mark.parametrize("num_workers,batch_size,batches_by_rank", [(1, 6, (6, 7)), (4, 4, (11, 10))])
def test_finite_coverage_through_trainer_dataloaders(
    packed_video_root: Path,
    tmp_path: Path,
    num_workers: int,
    batch_size: int,
    batches_by_rank: tuple[int, int],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "src"), env.get("PYTHONPATH", "")])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            __file__,
            str(packed_video_root),
            str(tmp_path),
            str(num_workers),
            str(batch_size),
            str(2 * max(batches_by_rank)),
        ],
        check=True,
        env=env,
        timeout=180,
    )

    reference = LeRobotDataset(DUMMY_REPO_ID, root=packed_video_root, return_uint8=True)
    seen = set()
    for rank, expected_batches in enumerate(batches_by_rank):
        lengths, samples, repeated = json.loads((tmp_path / f"{rank}.json").read_text())
        assert len(lengths) == expected_batches
        expected_lengths = []
        worker_episodes = (RANK_EPISODES[rank],) if num_workers == 1 else WORKER_EPISODES[rank]
        for episodes in worker_episodes:
            full, tail = divmod(sum(LENGTHS[episode] for episode in episodes), batch_size)
            expected_lengths.extend([batch_size] * full)
            if tail:
                expected_lengths.append(tail)
        assert sorted(lengths) == sorted(expected_lengths)
        offsets = list(accumulate((0, *LENGTHS)))
        expected_ids = {
            index
            for episode in RANK_EPISODES[rank]
            for index in range(offsets[episode], offsets[episode + 1])
        }
        assert sorted(sample[0] for sample in samples) == sorted(expected_ids)
        assert set(repeated[: len(expected_ids)]) == expected_ids
        assert set(repeated[len(expected_ids) : 2 * len(expected_ids)]) == expected_ids
        assert repeated[: len(expected_ids)] != repeated[len(expected_ids) : 2 * len(expected_ids)]
        assert set(repeated) == expected_ids
        for index, episode, worker, action, rgb in samples:
            assert index not in seen
            seen.add(index)
            expected = reference[index]
            assert episode == expected["episode_index"]
            assert action == expected["action"].tolist()
            assert rgb == _pixels(expected[CAMERA])
            expected_episodes = RANK_EPISODES[rank] if num_workers == 1 else WORKER_EPISODES[rank][worker]
            assert episode in expected_episodes
    assert len(seen) == sum(LENGTHS[:-1])


def test_direct_reader_uses_rank_environment_unless_pinned(
    packed_video_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    kwargs = {
        "root": packed_video_root,
        "episodes": [*SELECTED, -1, len(LENGTHS)],
        "shuffle": False,
    }
    dataset = StreamingLeRobotDataset(DUMMY_REPO_ID, **kwargs)
    assert dataset.num_episodes == len(LENGTHS) - 1
    assert dataset.num_frames == sum(LENGTHS[:-1])
    assert {item["episode_index"] for item in dataset} == RANK_EPISODES[1]

    dataset = StreamingLeRobotDataset(DUMMY_REPO_ID, **kwargs)
    dataset.set_data_parallel_shard(rank=0, world_size=2)
    assert {item["episode_index"] for item in dataset} == RANK_EPISODES[0]


if __name__ == "__main__":
    data_root, output_path, workers, batch, steps = sys.argv[1:]
    _collect_rank(Path(data_root), Path(output_path), int(workers), int(batch), int(steps))
