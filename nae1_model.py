#!/usr/bin/env python3
"""
Nae1 real-architecture model.

Components (each is one of the "modern architecture tricks" from the blueprint):

  Rotary Position Embeddings (RoPE)
    - Applied to query and key vectors before attention.
    - freq_i = 1 / (rope_theta ** (2i / head_dim))
    - q_rot = q * cos + rotate_half(q) * sin  (same for k)
    - rope_theta = 1,000,000 (Mistral default) by default.

  Grouped-Query Attention (GQA)
    - query: n_head heads
    - key/value: n_kv_heads heads, each shared across
      (n_head / n_query_groups) query heads.
    - This is what slashes KV-cache memory during generation and lets a small model
      run fast on the Mac.
    - Implemented with a grouped matmul + scaled_dot_product_attention.

  SwiGLU feedforward
    - gate = SiLU(W_gate @ x)
    - up   = W_up @ x
    - out  = W_down(gate * up)
    - SiLU(x) = x * sigmoid(x). "swish" in the blueprint = SiLU.

Top-level:
  Nae1Model: embed + [layer: attn + mlp] + final layernorm, with RoPE applied per
    layer (one RoPE instance per layer, identical params).
  Nae1ForCausalLM: Nae1Model + lm_head (tied to embed if tie_word_embeddings).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

# Allow running from the scripts/ dir where nae1_config.py lives alongside this file,
# and from elsewhere where both may have been exec'd into globals (Kaggle notebook).
try:
    from nae1_config import Nae1Config
except ImportError:
    from .nae1_config import Nae1Config


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Flip half the last dim and negate the first half (the RoPE rotate_half)."""
    # x: (..., head_dim) with head_dim even
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class Nae1RotaryEmbedding(nn.Module):
    """RoPE for Nae1. Same semantics as MistralRotaryEmbedding (rope_theta default
    1,000,000) so the Kaggle notebook and the Mac can share the same math."""

    def __init__(self, head_dim: int, max_seq_len: int, rope_theta: float = 1_000_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """Apply RoPE to x in (..., seq_len, head_dim) form (in-place math, returns new)."""
        # positions for this sequence (optionally offset for KV-cache continuation)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) + offset
        # freqs: (seq_len, head_dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=x.device, dtype=x.dtype))
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        cos = emb.cos()
        sin = emb.sin()
        return (x * cos) + (_rotate_half(x) * sin)


class Nae1SwiGLU(nn.Module):
    """SwiGLU FFN: gate = SiLU(W_gate x), up = W_up x, out = W_down(gate * up)."""

    def __init__(self, config: Nae1Config) -> None:
        super().__init__()
        self.w_gate = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.w_up = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.w_down = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.act = nn.SiLU()  # SiLU = swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))


class Nae1GQA(nn.Module):
    """Grouped-query attention for Nae1.

    query: n_head heads
    key/value: n_kv_heads heads, each shared across (n_head / n_query_groups) query
    heads. Implemented with a grouped matmul + scaled_dot_product_attention.
    """

    def __init__(self, config: Nae1Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_query_groups = config.n_query_groups
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** (-0.5)
        self.dropout_p = config.attn_pdrop

        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, rope: Nae1RotaryEmbedding,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                causal: bool = True,
                attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, L, _ = x.shape

        # Q: (B, L, n_head*head_dim) -> (B, n_head, L, head_dim)
        q = self.q_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        # K, V: (B, L, n_kv_heads*head_dim) -> (B, n_kv_heads, L, head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # When continuing from a cache, the new tokens sit at positions
        # [past_len, past_len + L) — RoPE must use absolute positions or the
        # cached keys (already rotated) won't line up with the new ones.
        offset = past_key_value[0].shape[2] if past_key_value is not None else 0

        # Apply RoPE to Q and K (per head).
        q = rope(q, seq_len=L, offset=offset).to(dtype=q.dtype)
        k = rope(k, seq_len=L, offset=offset).to(dtype=k.dtype)

        # GQA: repeat each KV head across its query heads.
        k = k.repeat_interleave(self.n_query_groups, dim=1)
        v = v.repeat_interleave(self.n_query_groups, dim=1)

        # Append to past if we have one (for generation / continuation).
        # NOTE: the returned cache must be the CONCATENATED pair — returning
        # the input past would silently drop the newest token on the next step.
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        past = (k, v)

        # Causal mask is only meaningful for a full-sequence pass (Q and K the
        # same length). On a cached decoding step Q is 1 token and every K is
        # already past by construction — and torch's is_causal aligns top-left
        # for Q<K, which would mask out all but the first key. So: no mask.
        # An explicit additive attention_mask (already combined with causality
        # by Nae1Model.forward) overrides is_causal entirely.
        if attention_mask is not None:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask.to(dtype=q.dtype),
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            eff_causal = causal and q.shape[2] == k.shape[2]
            attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=eff_causal,
            )
        # attn: (B, n_head, L, head_dim) -> (B, L, n_head*head_dim)
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.n_head * self.head_dim)
        return self.o_proj(attn), past


class Nae1MLP(nn.Module):
    """SwiGLU feedforward for Nae1.

    Nae1SwiGLU already includes the down projection (W_down(gate * up)), so this
    wrapper just delegates to it — there is no separate c_proj (that would be a
    second down projection and a shape mismatch).
    """

    def __init__(self, config: Nae1Config) -> None:
        super().__init__()
        self.c_fc = Nae1SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_fc(x)


class Nae1DecoderLayer(nn.Module):
    """One Nae1 decoder layer: rms-style layernorm + GQA attn + SwiGLU MLP, residuals."""

    def __init__(self, config: Nae1Config) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Nae1GQA(config)
        self.mlp = Nae1MLP(config)

    def forward(self, x: torch.Tensor, rope: Nae1RotaryEmbedding,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                causal: bool = True,
                attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        h = self.input_layernorm(x)
        h, past = self.attn(h, rope, past_key_value=past_key_value, causal=causal,
                            attention_mask=attention_mask)
        h = x + h
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h, past


class Nae1Model(nn.Module):
    """Nae1 backbone: embed + dropout + [decoder layer]*n_layer + final layernorm.

    Position info is carried by RoPE inside each decoder layer (applied to Q/K).
    """

    def __init__(self, config: Nae1Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.n_embd, padding_idx=config.pad_token_id
        )
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            Nae1DecoderLayer(config) for _ in range(config.n_layer)
        )
        self.final_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # One RoPE per layer (identical params; applied per layer to Q/K of that layer).
        self.rope = nn.ModuleList(
            Nae1RotaryEmbedding(config.head_dim, config.n_positions, config.rope_theta)
            for _ in range(config.n_layer)
        )

    def forward(self, input_ids: torch.Tensor,
                past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
                attention_mask: torch.Tensor | None = None,
                causal: bool = True,
                use_cache: bool = False) -> torch.Tensor | tuple:
        B, L = input_ids.shape
        h = self.dropout(self.embed_tokens(input_ids))

        # Convert a caller-supplied attention_mask into an additive bias for
        # SDPA. Accepted forms: (B, Lk) binary keep-mask (1 = attend,
        # 0 = padding), or an additive float mask broadcastable to
        # (B, n_head, Lq, Lk). Previously this argument was silently IGNORED
        # (a causal mask tensor was built and then never used), so padded
        # batches were never actually padding-masked.
        attn_bias: torch.Tensor | None = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                keep = attention_mask != 0
                attn_bias = torch.zeros(keep.shape, device=h.device, dtype=h.dtype)
                attn_bias = attn_bias.masked_fill(~keep, float("-inf"))[:, None, None, :]
                if causal and past_key_values is None:
                    triu = torch.full((L, L), float("-inf"), device=h.device,
                                      dtype=h.dtype).triu(diagonal=1)
                    attn_bias = attn_bias + triu
            else:
                attn_bias = attention_mask

        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if past_key_values is not None:
            past = list(past_key_values)

        new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            layer_past = past[i] if past is not None and i < len(past) else None
            h, new_past_i = layer(h, self.rope[i], past_key_value=layer_past,
                                  causal=causal, attention_mask=attn_bias)
            new_past.append(new_past_i)
        h = self.final_layernorm(h)
        if use_cache:
            return h, new_past
        return h


class Nae1ForCausalLM(PreTrainedModel):
    """Nae1 for causal LM: backbone + lm_head (tied to embed if tie_word_embeddings).

    Inherits PreTrainedModel so the init checkpoint can be saved/loaded with the
    standard save_pretrained / from_pretrained path (config.json + model.safetensors)
    and so num_parameters() is available for the smoke tests.
    """

    config_class = Nae1Config
    # Tell save_pretrained / remove_tied_weights_from_state_dict that lm_head.weight
    # is aliased to model.embed_tokens.weight (tie_word_embeddings). Without this the
    # safetensors save path refuses to write the aliased tensor.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Nae1Config) -> None:
        super().__init__(config)
        self.model = Nae1Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # Gather tied-weight metadata (and the other post-init static attrs) so
        # save_pretrained / from_pretrained can deduplicate and re-tie correctly.
        self.post_init()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
                past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
                causal: bool = True,
                use_cache: bool = False) -> dict[str, Any]:
        h = self.model(input_ids, past_key_values=past_key_values,
                        attention_mask=attention_mask, causal=causal,
                        use_cache=use_cache)
        past = None
        if use_cache:
            h, past = h
        logits = self.lm_head(h)  # (B, L, vocab_size)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            # NOTE: `logits` stays FULL length (B, L, vocab). An earlier version
            # replaced it with the shifted (B, L-1, vocab) tensor, which broke
            # the HF convention and silently off-by-one'd any caller pairing
            # returned logits with input positions.

        out: dict[str, Any] = {"loss": loss, "logits": logits}
        if use_cache:
            out["past_key_values"] = past
        return out

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        config: Nae1Config | None = None,
        **kwargs,
    ) -> "Nae1ForCausalLM":
        """Load an init checkpoint from a directory that contains config.json +
        model.safetensors (the format written by scripts/nae1_init_model.py).

        Delegates to PreTrainedModel.from_pretrained, which loads config.json as a
        Nae1Config (via config_class) and model.safetensors, then load_state_dict.
        If `config` is given it is used as the model config (and must match the
        saved config.json). Otherwise config.json is loaded and used."""
        return super().from_pretrained(str(pretrained_model_name_or_path), config=config, **kwargs)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        callback=None,
    ) -> list[int]:
        """Autoregressive generation with optional streaming callback.

        Args:
            input_ids: (B, L) prompt token ids.
            max_new_tokens: max tokens to generate.
            temperature: sampling temperature (1.0 = no scaling).
            top_k: top-k filtering (0 = off).
            top_p: nucleus sampling (1.0 = off).
            repetition_penalty: penalty for repeated tokens (1.0 = off).
            eos_token_id: stop at this token if set.
            callback: optional callable(token_id) called for each generated token
                      (for streaming SSE / UI updates).

        Returns:
            List of generated token ids (excluding the prompt).
        """
        cfg = self.config
        eos = eos_token_id if eos_token_id is not None else cfg.eos_token_id
        x = input_ids.clone()
        generated: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self(x)
                logits = out["logits"][0, -1, :].clone()
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[-1]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    remove_mask[..., 0] = False
                    indices_to_remove = remove_mask.scatter(0, sorted_idx, remove_mask)
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))
                if repetition_penalty != 1.0 and len(generated) > 0:
                    gen_tokens = torch.tensor(generated, device=logits.device, dtype=torch.long)
                    gen_logits = logits[gen_tokens]
                    gen_logits = torch.where(
                        gen_logits > 0,
                        gen_logits / repetition_penalty,
                        gen_logits * repetition_penalty,
                    )
                    logits[gen_tokens] = torch.min(logits[gen_tokens], gen_logits)
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                if nxt == eos:
                    break
                generated.append(nxt)
                if callback is not None:
                    callback(nxt)
                x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=x.device)], dim=1)
        return generated
