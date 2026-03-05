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

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import torch
import numpy as np
import os
import torch.distributed as dist

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.models.context_parallel import (
    init_context_parallel_group,
    apply_context_parallel_to_qwen,
    shard_model_inputs,
)

# Initialize distributed environment
dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}"

# Set Context Parallel size (using all GPUs for CP)
init_context_parallel_group(cp_size=world_size)

# Patch the VLM components before instantiating the model
apply_context_parallel_to_qwen(model=None)

# Example clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
if local_rank == 0:
    print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
if local_rank == 0:
    print("Dataset loaded.")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

# Load model to the correct local GPU
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to(device)
processor = helper.get_processor(model.tokenizer)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

# Shard inputs before passing them into the model to save memory
inputs = shard_model_inputs(inputs)

model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

model_inputs = helper.to_device(model_inputs, device)

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
        max_generation_length=256,
        return_extra=True,
    )

if local_rank == 0:
    # the size is [batch_size, num_traj_sets, num_traj_samples]
    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])
    
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print("minADE:", min_ade, "meters")
    print(
        "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
        "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
        "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
    )
