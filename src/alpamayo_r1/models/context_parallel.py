import torch
import torch.distributed as dist
from typing import Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import sequence parallel attention implementations
# For example, using yunchang (Ulysses/Ring Attention) or similarly implemented flash-attn wrappers
try:
    # This is a conceptual import for sequence parallel flash attention
    # You would typically install a package like `yunchang` or `ring_flash_attn`
    from ring_flash_attn import ring_flash_attn_func
    from ring_flash_attn import ring_flash_attn_varlen_func
    HAS_RING_ATTN = True
except ImportError:
    HAS_RING_ATTN = False
    logger.warning("ring_flash_attn is not installed. Context Parallelism for prefill will fallback to local attention or raise an error.")

_CP_GROUP = None

def init_context_parallel_group(cp_size: int):
    """Initialize the process group for Context Parallelism."""
    global _CP_GROUP
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert world_size % cp_size == 0, "World size must be divisible by CP size"
    
    for i in range(world_size // cp_size):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _CP_GROUP = group
            
    logger.info(f"Initialized Context Parallelism group with size {cp_size}")

def get_cp_group():
    return _CP_GROUP

def get_cp_rank():
    if _CP_GROUP is None:
        return 0
    return dist.get_rank(group=_CP_GROUP)

def get_cp_size():
    if _CP_GROUP is None:
        return 1
    return dist.get_world_size(group=_CP_GROUP)

def shard_model_inputs(inputs: dict, cp_group=None) -> dict:
    """
    Shards the sequence dimension of inputs for Context Parallelism before passing them to the model.
    Only shards text-related sequences along the sequence dimension.
    """
    if cp_group is None:
        cp_group = get_cp_group()
        
    cp_size = get_cp_size()
    cp_rank = get_cp_rank()
    
    if cp_size <= 1:
        return inputs
        
    sharded_inputs = {}
    for k, v in inputs.items():
        # Shard standard text inputs along sequence dimension (dim=1)
        if k in ["input_ids", "attention_mask", "position_ids"] and isinstance(v, torch.Tensor):
            seq_len = v.shape[1]
            
            # For simplicity, we pad the sequence if it's not strictly divisible
            # Real implementations might use un-even sharding like ring_flash_attn_varlen_func
            pad_len = (cp_size - (seq_len % cp_size)) % cp_size
            if pad_len > 0:
                if k == "input_ids":
                    # Assume 0 is pad token for now, or fetch from tokenizer
                    v = torch.nn.functional.pad(v, (0, pad_len), value=0)
                elif k == "attention_mask":
                    v = torch.nn.functional.pad(v, (0, pad_len), value=0)
                elif k == "position_ids":
                    v = torch.nn.functional.pad(v, (0, pad_len), value=v[:, -1].item())
            
            chunk_size = v.shape[1] // cp_size
            start_idx = cp_rank * chunk_size
            end_idx = start_idx + chunk_size
            
            sharded_inputs[k] = v[:, start_idx:end_idx].contiguous()
        else:
            # Pass through other inputs like pixel_values unchanged
            # (In an advanced implementation, Vision inputs should also be spatial-partitioned)
            sharded_inputs[k] = v
            
    return sharded_inputs

def apply_context_parallel_to_qwen(model: torch.nn.Module):
    """
    Monkey-patch the Qwen3-VL attention to use Context Parallelism.
    Only patches the VLM, leaving the Expert model unaffected.
    """
    import transformers.models.qwen2_vl.modeling_qwen2_vl as qwen2_vl_modeling
    
    # We patch the Flash Attention 2 forward method of Qwen
    original_forward = qwen2_vl_modeling.Qwen2VLFlashAttention2.forward
    
    def cp_flash_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        cp_group = get_cp_group()
        cp_size = get_cp_size()
        
        bsz, q_len, _ = hidden_states.size()
        
        # If CP is disabled, CP size is 1, or it's the decode phase (q_len == 1), 
        # fallback to the original Hugging Face implementation.
        if cp_size <= 1 or q_len == 1:
            return original_forward(
                self, hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, 
                cache_position, position_embeddings
            )
            
        # PREFILL PHASE (q_len > 1) with Context Parallelism
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to (bsz, q_len, num_heads, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # apply_rotary_pos_emb is assumed to exist in the module namespace
            query_states, key_states = qwen2_vl_modeling.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            
        # KV Cache logic for Prefill Phase
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_position
            )

        # Use Ring Attention or Ulysses to split Q, K, V sequence across GPUs
        if HAS_RING_ATTN:
            # Context Parallelism splits the sequence dimension.
            # Assume input hidden_states is already sharded across the sequence dim:
            # (bsz, q_len // cp_size, num_heads, head_dim)
            
            # If causality is needed, ring_flash_attn supports it
            is_causal = True if attention_mask is None else False
            
            attn_output = ring_flash_attn_func(
                query_states, key_states, value_states,
                causal=is_causal,
                group=cp_group
            )
            
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value
        else:
            # Fallback if CP is requested but ring_flash_attn is unavailable
            logger.warning_once("Using fallback local attention for prefill. CP is disabled.")
            return original_forward(
                self, hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, 
                cache_position, position_embeddings
            )

    # Patch the method
    qwen2_vl_modeling.Qwen2VLFlashAttention2.forward = cp_flash_attention_forward
    logger.info("Successfully patched Qwen2VLFlashAttention2 with Context Parallelism support.")
