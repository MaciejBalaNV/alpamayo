# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context-parallel inference for AlpamayoR1.

Launches one process per GPU via ``torchrun`` and uses context parallelism to
split the long prefill sequence across GPUs, reducing first-token latency.

Usage::

    torchrun --nproc_per_node=<N_GPUS> -m alpamayo_r1.test_inference_cp

The decode (autoregressive) phase runs identically on every rank after the
KV-cache is gathered, so all ranks produce the same output.
"""

import os
import time

import numpy as np
import torch
import torch.distributed as dist

from alpamayo_r1 import helper
from alpamayo_r1.context_parallel import (
    apply_context_parallel,
    get_cp_group,
    remove_context_parallel,
)
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


def main() -> None:
    # ---- distributed setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    cp = get_cp_group().initialize(backend="nccl")
    is_main = cp.rank == 0

    if is_main:
        print(f"Context parallelism: {cp.world_size} GPUs")

    # ---- dataset ----
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    if is_main:
        print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    if is_main:
        print("Dataset loaded.")
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    # ---- model (each rank loads onto its own GPU) ----
    device = f"cuda:{local_rank}"
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
    ).to(device)
    processor = helper.get_processor(model.tokenizer)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)

    if is_main:
        seq_len = inputs["input_ids"].shape[1]
        print(f"Input sequence length: {seq_len}")

    # ---- inference with context parallelism ----
    apply_context_parallel(model)

    num_warmup = 2
    inference_kwargs = dict(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

    # ---- warmup iterations (amortise CUDA graph capture, JIT, allocator) ----
    for i in range(num_warmup):
        if is_main:
            print(f"Warmup {i + 1}/{num_warmup}...")
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"warmup/{i}")
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(**inference_kwargs)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    # ---- timed iteration ----
    dist.barrier()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    torch.cuda.nvtx.range_push("inference")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.nvtx.range_push("vlm_generate_and_diffusion")
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            **inference_kwargs,
        )
        torch.cuda.nvtx.range_pop()  # vlm_generate_and_diffusion
    torch.cuda.nvtx.range_pop()  # inference

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    remove_context_parallel(model)

    # ---- results (rank 0 only) ----
    if is_main:
        print(f"\nInference time (after warmup): {t_end - t_start:.3f}s")
        print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
        print("minADE:", min_ade, "meters")
        print(
            "Note: VLA-reasoning models produce nondeterministic outputs due to "
            "trajectory sampling, hardware differences, etc. With "
            "num_traj_samples=1 (set for GPU memory compatibility), variance in "
            "minADE is expected."
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
