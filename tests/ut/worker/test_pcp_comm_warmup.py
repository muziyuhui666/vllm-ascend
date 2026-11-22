# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.worker.worker import NPUWorker


@pytest.mark.parametrize(
    ("pcp_size", "engram_config", "expected"),
    [
        (2, SimpleNamespace(cpu_offload=True), 1),
        (1, SimpleNamespace(cpu_offload=True), 0),
        (2, SimpleNamespace(cpu_offload=False), 0),
        (2, None, 0),
    ],
)
def test_eager_init_pcp_communicator_for_host_offload(
    pcp_size: int,
    engram_config: object | None,
    expected: int,
) -> None:
    worker = object.__new__(NPUWorker)
    worker.device = "npu:0"
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=pcp_size)
    worker.vllm_config = SimpleNamespace(engram_config=engram_config)
    group = MagicMock()
    token = MagicMock()

    with (
        patch("vllm_ascend.worker.worker.get_pcp_group", return_value=group),
        patch("vllm_ascend.worker.worker.torch.empty", return_value=token) as empty,
    ):
        worker._eager_init_pcp_communicator_for_host_offload()

    assert group.all_gather.call_count == expected
    assert empty.call_count == expected
    if expected:
        empty.assert_called_once_with(1, dtype=torch.uint8, device="npu:0")
        group.all_gather.assert_called_once_with(token, dim=0)
