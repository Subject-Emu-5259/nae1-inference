#!/usr/bin/env python3
"""
Nae1 configuration (real architecture: RoPE + GQA + SwiGLU).

Nae1 is a small-footprint, from-scratch causal LM in the Llama 3 / Mistral style:

  - RoPE (Rotary Position Embeddings): position injected into Q/K.
  - GQA (Grouped-Query Attention): fewer KV heads than query heads -> smaller KV cache.
  - SwiGLU FFN: gated linear unit, more knowledge per parameter than plain GELU.

Config numbers (blueprint + sensible architectural defaults):
  vocab_size=32000 (from tokenizer manifest)
  n_positions=2048 (blueprint context window)
  n_embd=768, n_layer=12, n_head=12 (blueprint)
  n_kv_heads=4  (GQA: 3 query heads per KV head; 12/4=3)
  rope_theta=1,000,000 (Mistral default)
  ffn_mult=8/3 (SwiGLU intermediate ~ 2048 for n_embd=768)
  hidden_act="silu" (SwiGLU gate is SiLU)
  bos_token_id=2, eos_token_id=3, pad_token_id=0 (match tokenizer manifest)
  tie_word_embeddings=True

This is the ONE SOURCE OF TRUTH for the architecture. The init-checkpoint builder
(scripts/nae1_init_model.py) and the Kaggle notebook (nae1-retrain-v1) both derive
their model from this config, and the smoke tests verify the config, the model, and
the tokenizer all agree on the shared quantities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig


class Nae1Config(PretrainedConfig):
    """Nae1 real-architecture config (RoPE + GQA + SwiGLU)."""

    model_type = "nae1"

    def __init__(
        self,
        vocab_size: int = 32000,
        n_positions: int = 2048,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_kv_heads: int = 4,
        rope_theta: float = 1_000_000.0,
        ffn_mult: float = 8.0 / 3.0,
        hidden_act: str = "silu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        if n_head % n_kv_heads != 0:
            raise ValueError(
                f"n_head ({n_head}) must be divisible by n_kv_heads ({n_kv_heads}) "
                "for grouped-query attention."
            )
        # Honor explicit architecture sizes from a checkpoint's config.json.
        # Previously these kwargs flowed into PretrainedConfig and were then
        # silently overwritten by the derived values below — a checkpoint
        # whose head_dim / intermediate_size ever differed from the derived
        # defaults would have been misread with no error.
        explicit_head_dim = kwargs.pop("head_dim", None)
        explicit_intermediate = kwargs.pop("intermediate_size", None)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads
        self.n_query_groups = n_head // n_kv_heads
        self.head_dim = explicit_head_dim if explicit_head_dim is not None else n_embd // n_head
        self.rope_theta = rope_theta
        self.intermediate_size = (
            explicit_intermediate if explicit_intermediate is not None
            else int(ffn_mult * n_embd + 0.5)
        )
        self.hidden_act = hidden_act
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_word_embeddings = tie_word_embeddings

    @property
    def tokenizer_special_tokens(self) -> list[str]:
        """The five special tokens Nae1's tokenizer is trained with, in id order."""
        return ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "\n"]
