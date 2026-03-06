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

"""Context Parallelism for AlpamayoR1 prefill phase.

Splits the input sequence across GPUs along the sequence dimension during the
prefill (prompt-encoding) phase.  Each GPU computes local Q and all-gathers K/V
from every other GPU so that every query position attends to the full causal
context.  After prefill the KV-cache is gathered so that standard single-GPU
autoregressive decode can proceed unchanged.

Designed for NVLink-connected GPUs where all-gather bandwidth is high.

Usage with ``torchrun``::

    torchrun --nproc_per_node=<N_GPUS> test_inference_cp.py
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextModel,
    apply_rotary_pos_emb,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context-parallel process group
# ---------------------------------------------------------------------------

class ContextParallelGroup:
    """Manages the NCCL process group used for context parallelism."""

    def __init__(self) -> None:
        self.rank: int = 0
        self.world_size: int = 1
        self.group: dist.ProcessGroup | None = None

    def initialize(self, backend: str = "nccl") -> "ContextParallelGroup":
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.group = dist.group.WORLD
        return self

    @property
    def is_active(self) -> bool:
        return self.world_size > 1


_cp_group = ContextParallelGroup()


def get_cp_group() -> ContextParallelGroup:
    return _cp_group


# ---------------------------------------------------------------------------
# Tensor splitting / gathering helpers
# ---------------------------------------------------------------------------

def _split(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Return this rank's chunk of *tensor* along *dim*."""
    cp = get_cp_group()
    return tensor.chunk(cp.world_size, dim=dim)[cp.rank].contiguous()


def _all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """All-gather *tensor* from every rank and concatenate along *dim*."""
    cp = get_cp_group()
    if not cp.is_active:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(cp.world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=cp.group)
    return torch.cat(gathered, dim=dim)


def _pad_dim(tensor: torch.Tensor, dim: int, pad_size: int, value: int | float = 0) -> torch.Tensor:
    """Pad *tensor* with *pad_size* elements of *value* along *dim*."""
    if pad_size == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = pad_size
    pad = torch.full(shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


# ---------------------------------------------------------------------------
# KV-cache helpers
# ---------------------------------------------------------------------------

def _gather_kv_cache(cache: Any, original_seq_len: int) -> None:
    """All-gather every layer's K and V, then trim padding in-place."""
    cp = get_cp_group()
    if not cp.is_active:
        return
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = _all_gather(cache.key_cache[i], dim=2)[:, :, :original_seq_len, :]
        cache.value_cache[i] = _all_gather(cache.value_cache[i], dim=2)[:, :, :original_seq_len, :]
    cache._seen_tokens = original_seq_len


# ---------------------------------------------------------------------------
# CP-aware attention forward  (replaces Qwen3VLTextAttention.forward)
# ---------------------------------------------------------------------------

def _cp_attention_forward(
    self: Qwen3VLTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Drop-in replacement for ``Qwen3VLTextAttention.forward`` that
    all-gathers K/V across the CP group during prefill.

    During decode (seq_len == 1) it delegates to the original forward.
    """
    cp = get_cp_group()
    bsz, q_len = hidden_states.shape[:2]

    if q_len == 1 or not cp.is_active:
        return self._cp_orig_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values=past_key_values, cache_position=cache_position,
            **kwargs,
        )

    # --- local Q / K / V ---
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs,
        )

    # --- all-gather K, V then slice for causal ---
    full_key = _all_gather(key_states, dim=2)
    full_value = _all_gather(value_states, dim=2)

    # Rank r's Q covers global positions [r*C, (r+1)*C).
    # Keep K/V for positions [0, (r+1)*C) so that flash_attn with
    # is_causal=True (Q aligned to the END of K) produces the correct
    # lower-triangular mask.
    end_pos = (cp.rank + 1) * q_len
    causal_key = full_key[:, :, :end_pos, :]
    causal_value = full_value[:, :, :end_pos, :]

    attn_output, _ = flash_attention_forward(
        self,
        query_states,
        causal_key,
        causal_value,
        attention_mask=None,
        dropout=0.0,
        scaling=self.scaling,
        is_causal=True,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# TextModel hooks  (split before, gather after the transformer layers)
# ---------------------------------------------------------------------------

def _split_deepstack(
    visual_pos_masks: torch.Tensor,
    deepstack_visual_embeds: list[torch.Tensor],
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Split deepstack visual features to match the local sequence chunk."""
    chunks = visual_pos_masks.chunk(world_size, dim=1)
    local_mask = chunks[rank].contiguous()

    n_vis_before = sum(c.sum().item() for c in chunks[:rank])
    n_vis_local = int(local_mask.sum().item())

    local_embeds = [
        e[n_vis_before: n_vis_before + n_vis_local] for e in deepstack_visual_embeds
    ]
    return local_mask, local_embeds


def _text_model_pre_hook(
    module: Qwen3VLTextModel,
    args: tuple,
    kwargs: dict,
) -> tuple[tuple, dict]:
    """Split sequence-dimension tensors across the CP group (prefill only)."""
    cp = get_cp_group()
    inputs_embeds = kwargs.get("inputs_embeds")
    if inputs_embeds is None or inputs_embeds.shape[1] <= 1 or not cp.is_active:
        return args, kwargs

    seq_len = inputs_embeds.shape[1]
    pad_size = (cp.world_size - seq_len % cp.world_size) % cp.world_size

    # ---- pad for divisibility ----
    if pad_size > 0:
        kwargs["inputs_embeds"] = _pad_dim(inputs_embeds, dim=1, pad_size=pad_size)

        pos = kwargs.get("position_ids")
        if pos is not None:
            last = pos[..., -1:] + 1
            pad_pos = last + torch.arange(pad_size, device=pos.device)
            pad_pos = pad_pos.expand(*pos.shape[:-1], pad_size)
            kwargs["position_ids"] = torch.cat([pos, pad_pos], dim=-1)

        attn = kwargs.get("attention_mask")
        if attn is not None:
            kwargs["attention_mask"] = _pad_dim(attn, dim=1, pad_size=pad_size, value=0)

        vis_mask = kwargs.get("visual_pos_masks")
        if vis_mask is not None:
            kwargs["visual_pos_masks"] = _pad_dim(vis_mask, dim=1, pad_size=pad_size, value=0)

        cpos = kwargs.get("cache_position")
        if cpos is not None:
            last_val = cpos[-1] + 1
            pad_cpos = torch.arange(last_val, last_val + pad_size, device=cpos.device)
            kwargs["cache_position"] = torch.cat([cpos, pad_cpos])

    # ---- split ----
    kwargs["inputs_embeds"] = _split(kwargs["inputs_embeds"], dim=1)

    pos = kwargs.get("position_ids")
    if pos is not None:
        kwargs["position_ids"] = _split(pos, dim=-1)

    attn = kwargs.get("attention_mask")
    if attn is not None:
        kwargs["attention_mask"] = _split(attn, dim=1)

    cpos = kwargs.get("cache_position")
    if cpos is not None:
        kwargs["cache_position"] = _split(cpos, dim=0)

    vis_mask = kwargs.get("visual_pos_masks")
    ds_embeds = kwargs.get("deepstack_visual_embeds")
    if vis_mask is not None:
        # vis_mask is already padded (if needed) but not yet split.
        # _split_deepstack chunks it internally and returns the local slice.
        local_mask, local_embeds = _split_deepstack(
            vis_mask, ds_embeds or [], cp.rank, cp.world_size,
        )
        kwargs["visual_pos_masks"] = local_mask
        if ds_embeds is not None:
            kwargs["deepstack_visual_embeds"] = local_embeds

    # Stash metadata for the post-hook.
    module._cp_original_seq_len = seq_len
    return args, kwargs


def _text_model_post_hook(
    module: Qwen3VLTextModel,
    args: Any,
    kwargs: Any,
    output: Any,
) -> Any:
    """All-gather hidden states and KV cache after prefill."""
    cp = get_cp_group()
    original_seq_len = getattr(module, "_cp_original_seq_len", None)
    if original_seq_len is None or not cp.is_active:
        return output

    del module._cp_original_seq_len

    output.last_hidden_state = _all_gather(
        output.last_hidden_state, dim=1,
    )[:, :original_seq_len, :]

    if output.past_key_values is not None:
        _gather_kv_cache(output.past_key_values, original_seq_len)

    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_hook_handles: list[Any] = []


def apply_context_parallel(model: torch.nn.Module) -> None:
    """Monkey-patch *model*'s VLM text backbone for context-parallel prefill."""
    cp = get_cp_group()
    if not cp.is_active:
        return

    vlm = model.vlm if hasattr(model, "vlm") else model
    qwen_model = vlm.model if hasattr(vlm, "model") else vlm
    text_model = qwen_model.language_model

    # --- TextModel hooks ---
    h1 = text_model.register_forward_pre_hook(_text_model_pre_hook, with_kwargs=True)
    h2 = text_model.register_forward_hook(_text_model_post_hook, with_kwargs=True)
    _hook_handles.extend([h1, h2])

    # --- Attention forward replacement ---
    for layer in text_model.layers:
        attn: Qwen3VLTextAttention = layer.self_attn
        attn._cp_orig_forward = attn.forward
        attn.forward = lambda *a, _m=attn, **kw: _cp_attention_forward(_m, *a, **kw)

    logger.info("Context parallelism applied  (rank %d / %d)", cp.rank, cp.world_size)


def remove_context_parallel(model: torch.nn.Module) -> None:
    """Undo :func:`apply_context_parallel`."""
    for h in _hook_handles:
        h.remove()
    _hook_handles.clear()

    vlm = model.vlm if hasattr(model, "vlm") else model
    qwen_model = vlm.model if hasattr(vlm, "model") else vlm
    text_model = qwen_model.language_model

    for layer in text_model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_cp_orig_forward"):
            attn.forward = attn._cp_orig_forward
            del attn._cp_orig_forward


@contextmanager
def context_parallel_prefill(model: torch.nn.Module):
    """Context manager that enables CP for the lifetime of a prefill pass."""
    apply_context_parallel(model)
    try:
        yield
    finally:
        remove_context_parallel(model)
