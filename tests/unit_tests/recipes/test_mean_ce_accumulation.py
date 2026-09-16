# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


class TinyTokenModel(nn.Module):
    """Independent token logits make masking and weighting analytically testable."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10)

    def forward(self, input_ids):
        """Look up class logits without a causal model or a CUDA kernel.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].

        Returns:
            Output with logits of shape [batch, sequence, vocab=3].
        """
        return CausalLMOutput(logits=self.weight[input_ids])


def _batch(index):
    """Return one pack with one, three, or zero supervised tokens."""
    labels = [[0, -100, -100, -100], [1, 2, 0, -100], [-100] * 4][index % 3]
    return {"input_ids": torch.tensor([[0, 1, 2, 3]]), "labels": torch.tensor([labels])}


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("clip", [False, True])
def test_mean_loss_matches_equal_pack_reference_after_accumulation(count, clip):
    """Unequal label counts and a partial window must preserve the reference update."""
    model = TinyTokenModel()
    reference = TinyTokenModel()
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    for name, value in {
        "model_parts": [model],
        "loss_fn": MaskedCrossEntropy(reduction="mean"),
        "dist_env": SimpleNamespace(device=torch.device("cpu")),
        "device_mesh": None,
        "tokenizer": None,
        "te_fp8": None,
        "pp_enabled": False,
        "distributed_config": SimpleNamespace(defer_fsdp_grad_sync=True),
        "_get_dp_group_size": lambda **kwargs: 1,
    }.items():
        object.__setattr__(recipe, name, value)
    batches = [_batch(i) for i in range(count)]
    buffer = []
    total_labels = sum(int((b["labels"] != -100).sum()) for b in batches)
    for i, batch in enumerate(batches):
        recipe._forward_backward_step(i, batch, loss_buffer=buffer, num_label_tokens=total_labels, num_batches=count)
    expected = (
        sum(
            F.cross_entropy(reference(b["input_ids"]).logits.flatten(0, 1), b["labels"].flatten(), reduction="sum")
            / (b["labels"] != -100).sum().clamp_min(1)
            for b in batches
        )
        / count
    )
    expected.backward()
    torch.testing.assert_close(sum(buffer), expected.detach())
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    if clip:
        observed_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
        torch.testing.assert_close(observed_norm, expected_norm)
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    torch.optim.SGD(reference.parameters(), lr=0.1).step()
    torch.testing.assert_close(model.weight, reference.weight)


def test_fully_masked_mean_is_finite_and_has_zero_gradient():
    """A masked pack contributes zero without poisoning the complete optimizer step."""
    logits = torch.randn(1, 4, 3, requires_grad=True)
    loss = MaskedCrossEntropy(reduction="mean")(logits, torch.full((1, 4), -100))
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(logits.grad) == 0


def _distributed_mean_worker(rank, world_size, rendezvous):
    """Compare a real two-rank DDP update against the complete fp32 reference."""
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        for count in (1, 2, 3):
            model = torch.nn.parallel.DistributedDataParallel(TinyTokenModel())
            reference = TinyTokenModel()
            recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
            for name, value in {
                "model_parts": [model],
                "loss_fn": MaskedCrossEntropy(reduction="mean"),
                "dist_env": SimpleNamespace(device=torch.device("cpu")),
                "device_mesh": None,
                "tokenizer": None,
                "te_fp8": None,
                "pp_enabled": False,
                "distributed_config": SimpleNamespace(defer_fsdp_grad_sync=True),
                "_get_dp_group_size": lambda **kwargs: world_size,
            }.items():
                object.__setattr__(recipe, name, value)
            buffer = []
            for i in range(count):
                recipe._forward_backward_step(
                    i, _batch(rank + i), loss_buffer=buffer, num_label_tokens=999, num_batches=count
                )
            expected = sum(
                F.cross_entropy(reference(b["input_ids"]).logits.flatten(0, 1), b["labels"].flatten(), reduction="sum")
                / (b["labels"] != -100).sum().clamp_min(1)
                for r in range(world_size)
                for i in range(count)
                for b in [_batch(r + i)]
            ) / (count * world_size)
            expected.backward()
            reported = sum(buffer)
            torch.distributed.all_reduce(reported)
            torch.testing.assert_close(reported, expected.detach())
            torch.testing.assert_close(model.module.weight.grad, reference.weight.grad)
            observed_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
            torch.testing.assert_close(observed_norm, expected_norm)
            torch.optim.SGD(model.parameters(), lr=0.1).step()
            torch.optim.SGD(reference.parameters(), lr=0.1).step()
            torch.testing.assert_close(model.module.weight, reference.weight)
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_mean_loss_matches_global_equal_pack_reference(tmp_path):
    """Rank-local means must not overweight short packs or scale updates by DP size."""
    torch.multiprocessing.spawn(
        _distributed_mean_worker,
        args=(2, "file://" + str(tmp_path / "rendezvous")),
        nprocs=2,
        join=True,
    )


def _fsdp_mean_worker(rank, world_size, rendezvous):
    """Exercise actual FSDP2 reductions and AutoModel clipping on two CUDA ranks."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.distributed.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        for defer_sync in (False, True):
            for count in (1, 2, 3):
                model = TinyTokenModel().to(device)
                reference = TinyTokenModel().to(device)
                fully_shard(model, mesh=mesh)
                recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
                for name, value in {
                    "model_parts": [model],
                    "loss_fn": MaskedCrossEntropy(reduction="mean"),
                    "dist_env": SimpleNamespace(device=device),
                    "device_mesh": mesh,
                    "tokenizer": None,
                    "te_fp8": None,
                    "pp_enabled": False,
                    "distributed_config": SimpleNamespace(defer_fsdp_grad_sync=defer_sync),
                    "_get_dp_group_size": lambda **kwargs: world_size,
                }.items():
                    object.__setattr__(recipe, name, value)
                buffer = []
                for i in range(count):
                    recipe._forward_backward_step(
                        i,
                        _batch(rank + i),
                        loss_buffer=buffer,
                        num_label_tokens=999,
                        num_batches=count,
                    )
                expected = sum(
                    F.cross_entropy(
                        reference(b["input_ids"].to(device)).logits.flatten(0, 1),
                        b["labels"].to(device).flatten(),
                        reduction="sum",
                    )
                    / (b["labels"] != -100).sum().clamp_min(1).to(device)
                    for r in range(world_size)
                    for i in range(count)
                    for b in [_batch(r + i)]
                ) / (count * world_size)
                expected.backward()
                reported = sum(buffer)
                torch.distributed.all_reduce(reported)
                torch.testing.assert_close(reported, expected.detach())
                torch.testing.assert_close(model.weight.grad.full_tensor(), reference.weight.grad)
                norm = scale_grads_and_clip_grad_norm(0.1, [model], device_mesh=mesh)
                expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
                if hasattr(norm, "full_tensor"):
                    norm = norm.full_tensor()
                torch.testing.assert_close(norm.float(), expected_norm)
                torch.optim.SGD(model.parameters(), lr=0.1).step()
                torch.optim.SGD(reference.parameters(), lr=0.1).step()
                torch.testing.assert_close(model.weight.full_tensor(), reference.weight)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible GPUs")
def test_fsdp_mean_loss_matches_global_equal_pack_reference(tmp_path):
    """Both FSDP sync policies must preserve the clipped update at partial windows."""
    torch.multiprocessing.spawn(
        _fsdp_mean_worker,
        args=(2, "file://" + str(tmp_path / "fsdp-rendezvous")),
        nprocs=2,
        join=True,
    )
