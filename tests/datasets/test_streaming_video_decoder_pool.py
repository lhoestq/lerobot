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
from unittest.mock import Mock

import pytest

decoders = pytest.importorskip(
    "torchcodec.decoders", reason="torchcodec is required (install lerobot[dataset])"
)

from lerobot.datasets.video_utils import _StreamingVideoDecoderPool  # noqa: E402

pytestmark = pytest.mark.dataset


def test_decoder_pool_leases_exclusively_and_evicts_idle_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(decoders, "VideoDecoder", Mock(side_effect=lambda *args, **kwargs: object()))
    first_path = tmp_path / "first.mp4"
    second_path = tmp_path / "second.mp4"
    first_path.touch()
    second_path.touch()

    with closing(_StreamingVideoDecoderPool(capacity=2)) as pool:
        with pool.lease(str(first_path)) as (first_decoder, first_source):
            with pool.lease(str(first_path)) as (other_decoder, other_source):
                assert first_decoder is not other_decoder
                assert first_source is not other_source
            with pool.lease(str(second_path)) as (_, second_source):
                assert other_source.closed
                assert not first_source.closed
                assert not second_source.closed

        with pool.lease(str(first_path)) as (reused_decoder, reused_source):
            assert reused_decoder is first_decoder
            assert reused_source is first_source
            assert not reused_source.closed

    assert first_source.closed
    assert second_source.closed


def test_decoder_pool_releases_capacity_after_create_and_decode_failures(tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.touch()
    opened_sources = []

    def create_decoder(source, *, seek_mode):
        opened_sources.append(source)
        if len(opened_sources) == 1:
            raise RuntimeError("decoder construction failed")
        return object()

    monkeypatch.setattr(decoders, "VideoDecoder", create_decoder)

    with closing(_StreamingVideoDecoderPool(capacity=1)) as pool:
        with pytest.raises(RuntimeError, match="decoder construction failed"), pool.lease(str(path)):
            pytest.fail("The decoder constructor must fail before the lease begins")
        assert opened_sources[0].closed

        with pytest.raises(RuntimeError, match="decode failed"), pool.lease(str(path)):
            raise RuntimeError("decode failed")
        assert opened_sources[1].closed

        with pool.lease(str(path)) as (_, source):
            assert source is opened_sources[2]
            assert not source.closed
    assert source.closed
