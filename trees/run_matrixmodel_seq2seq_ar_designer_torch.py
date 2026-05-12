#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from trees.rna_models import FractionalPositionEmbedding, sinusoidal_positions
from trees.rna_secondary import BASES
from trees.run_matrixmodel_universal_designer_torch import (
    CachedSequence,
    CONDITION_TOKENS,
    CONDITION_TO_ID,
    UNKNOWN_SIDE_TOKEN,
    choose_device,
    is_sequence_feasible_for_side,
    json_sanitize,
    length_summary,
    load_caches,
    make_logger,
    random_mask_condition,
    scheduled_learning_rate,
    source_counts,
)


IGNORE_INDEX = -100
PAIRED_UNKNOWN_TOKEN = "#"
FULL_STRUCTURE_TOKENS = ("L", "R", "x")
STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN = ("L", "R", "x", PAIRED_UNKNOWN_TOKEN, UNKNOWN_SIDE_TOKEN)
BASE_CONDITION_TOKENS = (*BASES, UNKNOWN_SIDE_TOKEN)
MODEL4_CONDITIONING_MODE = "structure_with_partial_bases_and_codons"
MODEL4_PRIME_CONDITIONING_MODE = "structure_with_partial_bases_and_codons_no_rho"
RHO_TOKENS = ("0", "1", "2")
AMINO_ACID_TOKENS = (
    "Ala",
    "Arg",
    "Asn",
    "Asp",
    "Cys",
    "Gln",
    "Glu",
    "Gly",
    "His",
    "Ile",
    "Leu",
    "Lys",
    "Met",
    "Phe",
    "Pro",
    "Ser",
    "Thr",
    "Trp",
    "Tyr",
    "Val",
    "Stop",
    UNKNOWN_SIDE_TOKEN,
)
CODON_TO_AMINO_ACID = {
    "UUU": "Phe",
    "UUC": "Phe",
    "UUA": "Leu",
    "UUG": "Leu",
    "UCU": "Ser",
    "UCC": "Ser",
    "UCA": "Ser",
    "UCG": "Ser",
    "UAU": "Tyr",
    "UAC": "Tyr",
    "UAA": "Stop",
    "UAG": "Stop",
    "UGU": "Cys",
    "UGC": "Cys",
    "UGA": "Stop",
    "UGG": "Trp",
    "CUU": "Leu",
    "CUC": "Leu",
    "CUA": "Leu",
    "CUG": "Leu",
    "CCU": "Pro",
    "CCC": "Pro",
    "CCA": "Pro",
    "CCG": "Pro",
    "CAU": "His",
    "CAC": "His",
    "CAA": "Gln",
    "CAG": "Gln",
    "CGU": "Arg",
    "CGC": "Arg",
    "CGA": "Arg",
    "CGG": "Arg",
    "AUU": "Ile",
    "AUC": "Ile",
    "AUA": "Ile",
    "AUG": "Met",
    "ACU": "Thr",
    "ACC": "Thr",
    "ACA": "Thr",
    "ACG": "Thr",
    "AAU": "Asn",
    "AAC": "Asn",
    "AAA": "Lys",
    "AAG": "Lys",
    "AGU": "Ser",
    "AGC": "Ser",
    "AGA": "Arg",
    "AGG": "Arg",
    "GUU": "Val",
    "GUC": "Val",
    "GUA": "Val",
    "GUG": "Val",
    "GCU": "Ala",
    "GCC": "Ala",
    "GCA": "Ala",
    "GCG": "Ala",
    "GAU": "Asp",
    "GAC": "Asp",
    "GAA": "Glu",
    "GAG": "Glu",
    "GGU": "Gly",
    "GGC": "Gly",
    "GGA": "Gly",
    "GGG": "Gly",
}


def source_vocab_tokens(conditioning_mode: str) -> tuple[str, ...]:
    if conditioning_mode == "full_structure":
        return FULL_STRUCTURE_TOKENS
    if conditioning_mode == "legacy_masked_structure":
        return tuple(CONDITION_TOKENS)
    if conditioning_mode == "structure_with_paired_unknown":
        return STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN
    if conditioning_mode == "structure_with_partial_bases":
        return tuple(f"{structure}|{base}" for structure in STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN for base in BASE_CONDITION_TOKENS)
    if conditioning_mode == MODEL4_CONDITIONING_MODE:
        return tuple(
            f"{structure}|{base}|rho{rho}|{amino_acid}"
            for structure in STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN
            for base in BASE_CONDITION_TOKENS
            for rho in RHO_TOKENS
            for amino_acid in AMINO_ACID_TOKENS
        )
    if conditioning_mode == MODEL4_PRIME_CONDITIONING_MODE:
        return tuple(
            f"{structure}|{base}|{amino_acid}"
            for structure in STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN
            for base in BASE_CONDITION_TOKENS
            for amino_acid in AMINO_ACID_TOKENS
        )
    raise ValueError(f"unsupported conditioning_mode={conditioning_mode!r}")


def sample_hidden_indices(length: int, rng: random.Random) -> set[int]:
    hidden_count = rng.randint(0, length)
    return set(rng.sample(range(length), hidden_count))


def corrupt_structure_with_paired_unknown(side: str, rng: random.Random) -> tuple[str, ...]:
    hidden = sample_hidden_indices(len(side), rng)
    paired_degrade_probability = rng.random()
    corrupted: list[str] = []
    for index, token in enumerate(side):
        if index in hidden:
            corrupted.append(UNKNOWN_SIDE_TOKEN)
        elif token in {"L", "R"} and rng.random() < paired_degrade_probability:
            corrupted.append(PAIRED_UNKNOWN_TOKEN)
        else:
            corrupted.append(token)
    return tuple(corrupted)


def encode_structure_with_paired_unknown(side: str, rng: random.Random) -> tuple[int, ...]:
    token_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
    return tuple(token_to_id[token] for token in corrupt_structure_with_paired_unknown(side, rng))


def encode_structure_with_partial_bases(side: str, base_ids: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    structure_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
    base_to_id = {token: index for index, token in enumerate(BASE_CONDITION_TOKENS)}
    base_hidden = sample_hidden_indices(len(base_ids), rng)
    structure_tokens = corrupt_structure_with_paired_unknown(side, rng)
    width = len(BASE_CONDITION_TOKENS)
    encoded: list[int] = []
    for index, structure_token in enumerate(structure_tokens):
        base_token = UNKNOWN_SIDE_TOKEN if index in base_hidden else BASES[base_ids[index]]
        encoded.append(structure_to_id[structure_token] * width + base_to_id[base_token])
    return tuple(encoded)


def codon_starts_for_phase(length: int, phase_shift: int) -> list[int]:
    return [index for index in range(length - 2) if (index + phase_shift) % 3 == 0]


def amino_acid_for_base_ids(base_ids: tuple[int, int, int]) -> str:
    codon = "".join(BASES[index] for index in base_ids)
    return CODON_TO_AMINO_ACID[codon]


def sample_random_frame_amino_acid_tokens(base_ids: tuple[int, ...], rng: random.Random) -> tuple[list[str], int, list[int], set[int]]:
    phase_shift = rng.randrange(3)
    codon_starts = codon_starts_for_phase(len(base_ids), phase_shift)
    revealed_codon_count = rng.randint(0, len(codon_starts))
    revealed_codon_indices = set(rng.sample(range(len(codon_starts)), revealed_codon_count))
    amino_acid_tokens = [UNKNOWN_SIDE_TOKEN] * len(base_ids)
    for codon_index, start in enumerate(codon_starts):
        if codon_index not in revealed_codon_indices:
            continue
        amino_acid = amino_acid_for_base_ids((base_ids[start], base_ids[start + 1], base_ids[start + 2]))
        amino_acid_tokens[start] = amino_acid
        amino_acid_tokens[start + 1] = amino_acid
        amino_acid_tokens[start + 2] = amino_acid
    return amino_acid_tokens, phase_shift, codon_starts, revealed_codon_indices


def encode_structure_with_partial_bases_and_codons(
    side: str,
    base_ids: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    structure_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
    base_to_id = {token: index for index, token in enumerate(BASE_CONDITION_TOKENS)}
    rho_to_id = {token: index for index, token in enumerate(RHO_TOKENS)}
    amino_acid_to_id = {token: index for index, token in enumerate(AMINO_ACID_TOKENS)}

    base_hidden = sample_hidden_indices(len(base_ids), rng)
    structure_tokens = corrupt_structure_with_paired_unknown(side, rng)
    amino_acid_tokens, phase_shift, _codon_starts, _revealed_codon_indices = sample_random_frame_amino_acid_tokens(base_ids, rng)

    base_width = len(BASE_CONDITION_TOKENS)
    rho_width = len(RHO_TOKENS)
    amino_acid_width = len(AMINO_ACID_TOKENS)
    encoded: list[int] = []
    for index, structure_token in enumerate(structure_tokens):
        base_token = UNKNOWN_SIDE_TOKEN if index in base_hidden else BASES[base_ids[index]]
        rho_token = str((index + phase_shift) % 3)
        amino_acid_token = amino_acid_tokens[index]
        encoded.append(
            (((structure_to_id[structure_token] * base_width + base_to_id[base_token]) * rho_width + rho_to_id[rho_token])
            * amino_acid_width)
            + amino_acid_to_id[amino_acid_token]
        )
    return tuple(encoded)


def encode_structure_with_partial_bases_and_codons_no_rho(
    side: str,
    base_ids: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    structure_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
    base_to_id = {token: index for index, token in enumerate(BASE_CONDITION_TOKENS)}
    amino_acid_to_id = {token: index for index, token in enumerate(AMINO_ACID_TOKENS)}

    base_hidden = sample_hidden_indices(len(base_ids), rng)
    structure_tokens = corrupt_structure_with_paired_unknown(side, rng)
    amino_acid_tokens, _phase_shift, _codon_starts, _revealed_codon_indices = sample_random_frame_amino_acid_tokens(base_ids, rng)

    base_width = len(BASE_CONDITION_TOKENS)
    amino_acid_width = len(AMINO_ACID_TOKENS)
    encoded: list[int] = []
    for index, structure_token in enumerate(structure_tokens):
        base_token = UNKNOWN_SIDE_TOKEN if index in base_hidden else BASES[base_ids[index]]
        amino_acid_token = amino_acid_tokens[index]
        encoded.append(
            (structure_to_id[structure_token] * base_width + base_to_id[base_token]) * amino_acid_width
            + amino_acid_to_id[amino_acid_token]
        )
    return tuple(encoded)


def encode_full_condition(side: str, *, conditioning_mode: str) -> tuple[int, ...]:
    if conditioning_mode == "full_structure":
        token_to_id = {token: index for index, token in enumerate(FULL_STRUCTURE_TOKENS)}
        return tuple(token_to_id[token] for token in side)
    if conditioning_mode == "legacy_masked_structure":
        return tuple(CONDITION_TO_ID[token] for token in side)
    if conditioning_mode == "structure_with_paired_unknown":
        token_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
        return tuple(token_to_id[token] for token in side)
    if conditioning_mode == "structure_with_partial_bases":
        structure_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
        unknown_base_id = BASE_CONDITION_TOKENS.index(UNKNOWN_SIDE_TOKEN)
        width = len(BASE_CONDITION_TOKENS)
        return tuple(structure_to_id[token] * width + unknown_base_id for token in side)
    if conditioning_mode == MODEL4_CONDITIONING_MODE:
        structure_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
        unknown_base_id = BASE_CONDITION_TOKENS.index(UNKNOWN_SIDE_TOKEN)
        unknown_amino_acid_id = AMINO_ACID_TOKENS.index(UNKNOWN_SIDE_TOKEN)
        base_width = len(BASE_CONDITION_TOKENS)
        rho_width = len(RHO_TOKENS)
        amino_acid_width = len(AMINO_ACID_TOKENS)
        return tuple(
            (((structure_to_id[token] * base_width + unknown_base_id) * rho_width + (index % 3)) * amino_acid_width)
            + unknown_amino_acid_id
            for index, token in enumerate(side)
        )
    if conditioning_mode == MODEL4_PRIME_CONDITIONING_MODE:
        structure_to_id = {token: index for index, token in enumerate(STRUCTURE_TOKENS_WITH_PAIRED_UNKNOWN)}
        unknown_base_id = BASE_CONDITION_TOKENS.index(UNKNOWN_SIDE_TOKEN)
        unknown_amino_acid_id = AMINO_ACID_TOKENS.index(UNKNOWN_SIDE_TOKEN)
        base_width = len(BASE_CONDITION_TOKENS)
        amino_acid_width = len(AMINO_ACID_TOKENS)
        return tuple(
            (structure_to_id[token] * base_width + unknown_base_id) * amino_acid_width
            + unknown_amino_acid_id
            for token in side
        )
    raise ValueError(f"unsupported conditioning_mode={conditioning_mode!r}")


@dataclass(frozen=True)
class Seq2SeqConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    max_len: int
    conditioning_mode: str = "legacy_masked_structure"
    d_model: int = 512
    n_head: int = 8
    encoder_layers: int = 8
    decoder_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.05
    position_encoding: str = "fractional"


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class RnaSeq2SeqARTransformer(nn.Module):
    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.src_pad_id = config.src_vocab_size
        self.tgt_bos_id = config.tgt_vocab_size
        self.tgt_pad_id = config.tgt_vocab_size + 1
        self.src_embedding = nn.Embedding(config.src_vocab_size + 1, config.d_model, padding_idx=self.src_pad_id)
        self.tgt_embedding = nn.Embedding(config.tgt_vocab_size + 2, config.d_model, padding_idx=self.tgt_pad_id)
        if config.position_encoding == "learned":
            self.src_position_embedding = nn.Embedding(config.max_len, config.d_model)
            self.tgt_position_embedding = nn.Embedding(config.max_len, config.d_model)
            self.src_fractional_position_embedding = None
            self.tgt_fractional_position_embedding = None
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        elif config.position_encoding == "sinusoidal":
            self.src_position_embedding = None
            self.tgt_position_embedding = None
            self.src_fractional_position_embedding = None
            self.tgt_fractional_position_embedding = None
            self.register_buffer(
                "sinusoidal_position_table",
                sinusoidal_positions(config.max_len, config.d_model),
                persistent=False,
            )
        elif config.position_encoding == "fractional":
            self.src_position_embedding = None
            self.tgt_position_embedding = None
            self.src_fractional_position_embedding = FractionalPositionEmbedding(config.d_model)
            self.tgt_fractional_position_embedding = FractionalPositionEmbedding(config.d_model)
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        elif config.position_encoding == "none":
            self.src_position_embedding = None
            self.tgt_position_embedding = None
            self.src_fractional_position_embedding = None
            self.tgt_fractional_position_embedding = None
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        else:
            raise ValueError(f"unsupported position_encoding={config.position_encoding!r}")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.tgt_vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def position_values(self, valid_mask: torch.Tensor, *, which: str) -> torch.Tensor:
        batch_size, seq_len = valid_mask.shape
        positions = torch.arange(seq_len, device=valid_mask.device).unsqueeze(0).expand(batch_size, seq_len)
        if which == "src" and self.src_position_embedding is not None:
            return self.src_position_embedding(positions)
        if which == "tgt" and self.tgt_position_embedding is not None:
            return self.tgt_position_embedding(positions)
        if which == "src" and self.src_fractional_position_embedding is not None:
            return self.src_fractional_position_embedding(valid_mask)
        if which == "tgt" and self.tgt_fractional_position_embedding is not None:
            return self.tgt_fractional_position_embedding(valid_mask)
        if self.sinusoidal_position_table.numel() > 0:
            return self.sinusoidal_position_table[:seq_len].to(valid_mask.device).unsqueeze(0)
        return torch.zeros(batch_size, seq_len, self.config.d_model, device=valid_mask.device)

    def encode(self, src_ids: torch.Tensor, src_valid: torch.Tensor) -> torch.Tensor:
        if src_ids.shape[1] > self.config.max_len:
            raise ValueError(f"source length {src_ids.shape[1]} exceeds max_len={self.config.max_len}")
        src = self.src_embedding(src_ids) + self.position_values(src_valid, which="src")
        memory = self.encoder(src, src_key_padding_mask=~src_valid)
        return self.encoder_norm(memory)

    def forward(
        self,
        src_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        src_valid: torch.Tensor,
        tgt_valid: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_input_ids.shape[1] > self.config.max_len:
            raise ValueError(f"target length {decoder_input_ids.shape[1]} exceeds max_len={self.config.max_len}")
        memory = self.encode(src_ids, src_valid)
        tgt = self.tgt_embedding(decoder_input_ids) + self.position_values(tgt_valid, which="tgt")
        target_length = decoder_input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(target_length, target_length, dtype=torch.bool, device=decoder_input_ids.device),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~tgt_valid,
            memory_key_padding_mask=~src_valid,
        )
        return self.head(self.decoder_norm(decoded))

    def _split_heads(self, tensor: torch.Tensor, *, attention: nn.MultiheadAttention) -> torch.Tensor:
        batch_size, seq_len, embed_dim = tensor.shape
        head_dim = embed_dim // attention.num_heads
        return tensor.view(batch_size, seq_len, attention.num_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _heads, seq_len, head_dim = tensor.shape
        return tensor.transpose(1, 2).reshape(batch_size, seq_len, _heads * head_dim)

    def _project_qkv(
        self,
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attention.in_proj_weight is None:
            raise NotImplementedError("cached decode expects packed q/k/v projections")
        q_weight, k_weight, v_weight = attention.in_proj_weight.chunk(3, dim=0)
        if attention.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = attention.in_proj_bias.chunk(3, dim=0)
        return (
            F.linear(query, q_weight, q_bias),
            F.linear(key, k_weight, k_bias),
            F.linear(value, v_weight, v_bias),
        )

    def _attention_from_projected(
        self,
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_heads = self._split_heads(query, attention=attention)
        key_heads = self._split_heads(key, attention=attention)
        value_heads = self._split_heads(value, attention=attention)
        dropout_p = attention.dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        return attention.out_proj(self._merge_heads(attended))

    def precompute_cross_attention_cache(
        self,
        memory: torch.Tensor,
        src_valid: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Project encoder memory once per decoder layer for incremental decoding."""
        memory_mask = src_valid[:, None, None, :]
        cached: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for layer in self.decoder.layers:
            attention = layer.multihead_attn
            _q, key, value = self._project_qkv(attention, memory, memory, memory)
            cached.append((self._split_heads(key, attention=attention), self._split_heads(value, attention=attention), memory_mask))
        return cached

    def cached_decode_step(
        self,
        decoder_input_ids: torch.Tensor,
        *,
        position: int,
        tgt_position_values: torch.Tensor,
        memory: torch.Tensor,
        src_valid: torch.Tensor,
        self_cache: list[tuple[torch.Tensor, torch.Tensor] | None],
        cross_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Decode one autoregressive position with decoder self-attention K/V caching.

        This is an inference helper. It mirrors the norm-first TransformerDecoderLayer
        used by this model, but only runs the decoder stack for the current token.
        """
        if len(self_cache) != len(self.decoder.layers):
            raise ValueError("self_cache length must match decoder layer count")
        if cross_cache is not None and len(cross_cache) != len(self.decoder.layers):
            raise ValueError("cross_cache length must match decoder layer count")
        if not all(getattr(layer, "norm_first", False) for layer in self.decoder.layers):
            raise NotImplementedError("cached decode is implemented for norm_first decoder layers")

        current = decoder_input_ids[:, None]
        if current.shape[0] != memory.shape[0]:
            raise ValueError("decoder batch size must match encoder memory batch size")
        x = self.tgt_embedding(current) + tgt_position_values[:, position : position + 1]
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        memory_mask = src_valid[:, None, None, :]

        for layer_index, layer in enumerate(self.decoder.layers):
            self_attention = layer.self_attn
            self_norm = layer.norm1(x)
            self_q, self_k, self_v = self._project_qkv(
                self_attention,
                self_norm,
                self_norm,
                self_norm,
            )
            self_q_heads = self._split_heads(self_q, attention=self_attention)
            self_k_heads = self._split_heads(self_k, attention=self_attention)
            self_v_heads = self._split_heads(self_v, attention=self_attention)
            previous = self_cache[layer_index]
            if previous is not None:
                self_k_heads = torch.cat((previous[0], self_k_heads), dim=2)
                self_v_heads = torch.cat((previous[1], self_v_heads), dim=2)
            self_attended = F.scaled_dot_product_attention(
                self_q_heads,
                self_k_heads,
                self_v_heads,
                dropout_p=self_attention.dropout if self.training else 0.0,
                is_causal=False,
            )
            self_out = self_attention.out_proj(self._merge_heads(self_attended))
            x = x + layer.dropout1(self_out)
            next_cache.append((self_k_heads, self_v_heads))

            cross_attention = layer.multihead_attn
            cross_norm = layer.norm2(x)
            q_weight = cross_attention.in_proj_weight[: cross_attention.embed_dim]
            q_bias = cross_attention.in_proj_bias[: cross_attention.embed_dim] if cross_attention.in_proj_bias is not None else None
            cross_q = F.linear(cross_norm, q_weight, q_bias)
            cross_q_heads = self._split_heads(cross_q, attention=cross_attention)
            if cross_cache is None:
                _unused_q, cross_k, cross_v = self._project_qkv(cross_attention, memory, memory, memory)
                cross_k_heads = self._split_heads(cross_k, attention=cross_attention)
                cross_v_heads = self._split_heads(cross_v, attention=cross_attention)
                cross_mask = memory_mask
            else:
                cross_k_heads, cross_v_heads, cross_mask = cross_cache[layer_index]
            cross_attended = F.scaled_dot_product_attention(
                cross_q_heads,
                cross_k_heads,
                cross_v_heads,
                attn_mask=cross_mask,
                dropout_p=cross_attention.dropout if self.training else 0.0,
                is_causal=False,
            )
            cross_out = cross_attention.out_proj(self._merge_heads(cross_attended))
            x = x + layer.dropout2(cross_out)

            feed_forward = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x)))))
            x = x + layer.dropout3(feed_forward)

        logits = self.head(self.decoder_norm(x)).squeeze(1)
        return logits, next_cache


@dataclass(frozen=True)
class ARExample:
    condition_ids: tuple[int, ...]
    base_ids: tuple[int, ...]
    length: int
    source: str


class ARDataset(Dataset[ARExample]):
    def __init__(self, examples: list[ARExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ARExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train encoder-decoder AR p(x|masked t) on cached Vienna/matrixmodel structures.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache", action="append", required=True, help="Source-tagged cache path, as SOURCE:PATH. May be repeated.")
    parser.add_argument("--validation-cache", action="append", default=[], help="Optional validation cache path, as SOURCE:PATH.")
    parser.add_argument("--cache-split", default=None)
    parser.add_argument("--validation-split", default=None)
    parser.add_argument("--max-structures-per-row", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bucket-width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--hold-fraction", type=float, default=0.55)
    parser.add_argument("--lr-schedule-epochs", type=int, default=0)
    parser.add_argument(
        "--lr-schedule-mode",
        choices=("standard", "piecewise_50_25_25"),
        default="standard",
        help=(
            "standard uses the existing single cosine/warmup schedule. "
            "piecewise_50_25_25 runs epochs 1-50 with the standard schedule, "
            "epochs 51-75 flat at --min-learning-rate, and epochs 76-100 as a "
            "fresh cosine decay from --min-learning-rate to --piecewise-final-min-learning-rate."
        ),
    )
    parser.add_argument("--piecewise-final-min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--reset-lr-schedule-on-resume", action="store_true")
    parser.add_argument(
        "--conditioning-mode",
        choices=(
            "full_structure",
            "legacy_masked_structure",
            "structure_with_paired_unknown",
            "structure_with_partial_bases",
            MODEL4_CONDITIONING_MODE,
            MODEL4_PRIME_CONDITIONING_MODE,
        ),
        default="legacy_masked_structure",
        help=(
            "Source conditioning for AR training. full_structure trains p(x|s) on exact L/R/x inputs; "
            "legacy_masked_structure matches old checkpoints; "
            "structure_with_paired_unknown adds # for paired-orientation-hidden sites; "
            "structure_with_partial_bases adds an independently masked x_tilde base channel; "
            f"{MODEL4_CONDITIONING_MODE} additionally adds random-frame rho_i and randomly revealed amino-acid labels; "
            f"{MODEL4_PRIME_CONDITIONING_MODE} removes rho_i while keeping the same amino-acid reveal policy."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--position-encoding", choices=("learned", "sinusoidal", "fractional", "none"), default="fractional")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--validation-batch-size", type=int, default=0)
    parser.add_argument("--feasibility-every", type=int, default=0)
    parser.add_argument("--feasibility-count", type=int, default=128)
    parser.add_argument("--feasibility-batch-size", type=int, default=64)
    parser.add_argument("--feasibility-temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--protected-checkpoint-epochs",
        default="",
        help="Comma-separated epochs whose checkpoint/model state should be copied to protected filenames.",
    )
    parser.add_argument("--protected-checkpoint-tag", default="", help="Optional suffix tag for protected milestone files.")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=choose_device(args.device),
        )
    if args.device != "cuda":
        raise ValueError("distributed training requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requested but CUDA is unavailable")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> RnaSeq2SeqARTransformer:
    if isinstance(model, DistributedDataParallel):
        return model.module  # type: ignore[return-value]
    return model  # type: ignore[return-value]


def shard_batches_for_rank(batches: list[list[int]], *, rank: int, world_size: int) -> list[list[int]]:
    if world_size <= 1:
        return batches
    usable_count = (len(batches) // world_size) * world_size
    return batches[rank:usable_count:world_size]


def distributed_sum(value: float, *, context: DistributedContext) -> float:
    if not context.enabled:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_epoch_examples(
    rows: list[CachedSequence],
    *,
    epoch_index: int,
    seed: int,
    conditioning_mode: str,
) -> list[ARExample]:
    rng = random.Random(seed + 100_003 * epoch_index)
    examples: list[ARExample] = []
    for row in rows:
        side = row.structure_at(epoch_index)
        if conditioning_mode == "full_structure":
            condition_ids = encode_full_condition(side, conditioning_mode=conditioning_mode)
        elif conditioning_mode == "legacy_masked_structure":
            condition_ids = random_mask_condition(side, rng)
        elif conditioning_mode == "structure_with_paired_unknown":
            condition_ids = encode_structure_with_paired_unknown(side, rng)
        elif conditioning_mode == "structure_with_partial_bases":
            condition_ids = encode_structure_with_partial_bases(side, row.base_ids, rng)
        elif conditioning_mode == MODEL4_CONDITIONING_MODE:
            condition_ids = encode_structure_with_partial_bases_and_codons(side, row.base_ids, rng)
        elif conditioning_mode == MODEL4_PRIME_CONDITIONING_MODE:
            condition_ids = encode_structure_with_partial_bases_and_codons_no_rho(side, row.base_ids, rng)
        else:
            raise ValueError(f"unsupported conditioning_mode={conditioning_mode!r}")
        examples.append(
            ARExample(
                condition_ids=condition_ids,
                base_ids=row.base_ids,
                length=row.length,
                source=row.source,
            )
        )
    return examples


def make_bucket_batches(examples: list[ARExample], *, batch_size: int, bucket_width: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    buckets: dict[int, list[int]] = {}
    for index, example in enumerate(examples):
        buckets.setdefault(example.length // bucket_width, []).append(index)
    batches: list[list[int]] = []
    for indices in buckets.values():
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])
    rng.shuffle(batches)
    return batches


def collate_examples(batch: list[ARExample], *, model: RnaSeq2SeqARTransformer) -> dict[str, torch.Tensor]:
    max_len = max(example.length for example in batch)
    src_ids = torch.full((len(batch), max_len), model.src_pad_id, dtype=torch.long)
    decoder_input_ids = torch.full((len(batch), max_len), model.tgt_pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    src_valid = torch.zeros((len(batch), max_len), dtype=torch.bool)
    tgt_valid = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row_index, example in enumerate(batch):
        length = example.length
        src_ids[row_index, :length] = torch.tensor(example.condition_ids, dtype=torch.long)
        decoder_input_ids[row_index, 0] = model.tgt_bos_id
        if length > 1:
            decoder_input_ids[row_index, 1:length] = torch.tensor(example.base_ids[:-1], dtype=torch.long)
        labels[row_index, :length] = torch.tensor(example.base_ids, dtype=torch.long)
        src_valid[row_index, :length] = True
        tgt_valid[row_index, :length] = True
    return {
        "src_ids": src_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "src_valid": src_valid,
        "tgt_valid": tgt_valid,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loss_fn(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    base_model = unwrap_model(model)
    logits = model(batch["src_ids"], batch["decoder_input_ids"], batch["src_valid"], batch["tgt_valid"])
    loss = F.cross_entropy(
        logits.reshape(-1, base_model.config.tgt_vocab_size),
        batch["labels"].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    with torch.no_grad():
        valid = batch["labels"].ne(IGNORE_INDEX)
        pred = logits.argmax(dim=-1)
        token_acc = (pred[valid] == batch["labels"][valid]).float().mean().item()
        token_count = valid.float().sum().item()
    return loss, {
        "loss": float(loss.item()),
        "token_accuracy": token_acc,
        "token_count": token_count,
    }


def evaluate(
    model: RnaSeq2SeqARTransformer,
    examples: list[ARExample],
    *,
    batch_size: int,
    bucket_width: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    batches = make_bucket_batches(examples, batch_size=batch_size, bucket_width=bucket_width, seed=7)
    loader = DataLoader(
        ARDataset(examples),
        batch_sampler=batches,
        num_workers=0,
        collate_fn=lambda batch: collate_examples(batch, model=model),
    )
    totals: dict[str, float] = {}
    total_count = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            _, metrics = loss_fn(model, batch)
            count = metrics["token_count"]
            total_count += count
            for key, value in metrics.items():
                if key == "token_count":
                    continue
                totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / max(total_count, 1.0) for key, value in totals.items()}


def sample_sequences_for_conditions(
    model: RnaSeq2SeqARTransformer,
    condition_rows: list[tuple[int, ...]],
    *,
    temperature: float,
    seed: int,
    device: torch.device,
    batch_size: int,
    fixed_base_rows: list[tuple[int | None, ...]] | None = None,
    use_cache: bool = True,
) -> list[str]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if fixed_base_rows is not None and len(fixed_base_rows) != len(condition_rows):
        raise ValueError("fixed_base_rows must match condition_rows length")
    results: list[str] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(condition_rows), batch_size):
            rows = condition_rows[start : start + batch_size]
            fixed_rows = fixed_base_rows[start : start + batch_size] if fixed_base_rows is not None else None
            lengths = [len(row) for row in rows]
            max_len = max(lengths)
            src = torch.full((len(rows), max_len), model.src_pad_id, dtype=torch.long, device=device)
            src_valid = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
            decoder_input = torch.full((len(rows), max_len), model.tgt_pad_id, dtype=torch.long, device=device)
            generated = torch.full((len(rows), max_len), model.tgt_pad_id, dtype=torch.long, device=device)
            tgt_valid = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
            for row_index, row in enumerate(rows):
                length = len(row)
                src[row_index, :length] = torch.tensor(row, dtype=torch.long, device=device)
                src_valid[row_index, :length] = True
                decoder_input[row_index, 0] = model.tgt_bos_id
                tgt_valid[row_index, :length] = True
            all_rows = torch.arange(len(rows), device=device)
            if use_cache:
                memory = model.encode(src, src_valid)
                target_positions = model.position_values(tgt_valid, which="tgt")
                cross_cache = model.precompute_cross_attention_cache(memory, src_valid)
                self_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(model.decoder.layers)
                current_input = torch.full((len(rows),), model.tgt_bos_id, dtype=torch.long, device=device)
                for position in range(max_len):
                    active = torch.tensor([length > position for length in lengths], dtype=torch.bool, device=device)
                    if not bool(active.any().item()):
                        break
                    logits, self_cache = model.cached_decode_step(
                        current_input,
                        position=position,
                        tgt_position_values=target_positions,
                        memory=memory,
                        src_valid=src_valid,
                        self_cache=self_cache,
                        cross_cache=cross_cache,
                    )
                    active_rows = all_rows[active]
                    if fixed_rows is None:
                        token = torch.distributions.Categorical(logits=logits[active_rows] / temperature).sample()
                    else:
                        token_values: list[int] = []
                        free_row_indices: list[int] = []
                        for local_row in active_rows.detach().cpu().tolist():
                            fixed_base = fixed_rows[local_row][position]
                            if fixed_base is None:
                                free_row_indices.append(local_row)
                                token_values.append(-1)
                            else:
                                token_values.append(int(fixed_base))
                        token = torch.tensor(token_values, dtype=torch.long, device=device)
                        if free_row_indices:
                            free_rows_tensor = torch.tensor(free_row_indices, dtype=torch.long, device=device)
                            free_tokens = torch.distributions.Categorical(logits=logits[free_rows_tensor] / temperature).sample()
                            free_counter = 0
                            for token_index, value in enumerate(token_values):
                                if value < 0:
                                    token[token_index] = free_tokens[free_counter]
                                    free_counter += 1
                    generated[active_rows, position] = token
                    if position + 1 < max_len:
                        still_active_next = torch.tensor(
                            [length > position + 1 for length in lengths],
                            dtype=torch.bool,
                            device=device,
                        )
                        next_input = torch.full((len(rows),), model.tgt_pad_id, dtype=torch.long, device=device)
                        next_rows = all_rows[active & still_active_next]
                        next_input[next_rows] = token[still_active_next[active]]
                        current_input = next_input
            else:
                for position in range(max_len):
                    active = torch.tensor([length > position for length in lengths], dtype=torch.bool, device=device)
                    if not bool(active.any().item()):
                        break
                    logits = model(src, decoder_input, src_valid, tgt_valid)
                    active_rows = all_rows[active]
                    if fixed_rows is None:
                        token = torch.distributions.Categorical(logits=logits[active_rows, position] / temperature).sample()
                    else:
                        token_values: list[int] = []
                        free_row_indices: list[int] = []
                        for local_row in active_rows.detach().cpu().tolist():
                            fixed_base = fixed_rows[local_row][position]
                            if fixed_base is None:
                                free_row_indices.append(local_row)
                                token_values.append(-1)
                            else:
                                token_values.append(int(fixed_base))
                        token = torch.tensor(token_values, dtype=torch.long, device=device)
                        if free_row_indices:
                            free_rows_tensor = torch.tensor(free_row_indices, dtype=torch.long, device=device)
                            free_tokens = torch.distributions.Categorical(
                                logits=logits[free_rows_tensor, position] / temperature
                            ).sample()
                            free_counter = 0
                            for token_index, value in enumerate(token_values):
                                if value < 0:
                                    token[token_index] = free_tokens[free_counter]
                                    free_counter += 1
                    generated[active_rows, position] = token
                    if position + 1 < max_len:
                        still_active_next = torch.tensor([length > position + 1 for length in lengths], dtype=torch.bool, device=device)
                        next_rows = all_rows[active & still_active_next]
                        decoder_input[next_rows, position + 1] = token[still_active_next[active]]
            for row_index, length in enumerate(lengths):
                ids = generated[row_index, :length].detach().cpu().tolist()
                results.append("".join(BASES[index] for index in ids))
    return results


def evaluate_feasibility(
    model: RnaSeq2SeqARTransformer,
    rows: list[CachedSequence],
    *,
    epoch_index: int,
    count: int,
    temperature: float,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if not rows or count <= 0:
        return {}
    rng = random.Random(seed + 1_000_003 * epoch_index)
    chosen = rng.sample(rows, min(count, len(rows)))
    sides = [row.structure_at(epoch_index) for row in chosen]
    conditioning_mode = getattr(model.config, "conditioning_mode", "legacy_masked_structure")
    conditions = [encode_full_condition(side, conditioning_mode=conditioning_mode) for side in sides]
    sequences = sample_sequences_for_conditions(
        model,
        conditions,
        temperature=temperature,
        seed=seed + 2_000_003 * epoch_index,
        device=device,
        batch_size=batch_size,
    )
    feasible = sum(is_sequence_feasible_for_side(sequence, side) for sequence, side in zip(sequences, sides))
    lengths = [len(side) for side in sides]
    pair_counts = [side.count("L") for side in sides]
    return {
        "count": float(len(sides)),
        "feasible_count": float(feasible),
        "feasibility_rate": feasible / max(len(sides), 1),
        "mean_length": sum(lengths) / max(len(lengths), 1),
        "mean_pair_count": sum(pair_counts) / max(len(pair_counts), 1),
        "temperature": temperature,
    }


def save_checkpoint(
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    config: Seq2SeqConfig,
) -> Path:
    base_model = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "args": vars(args),
        "config": asdict(config),
    }
    tmp_path = output_dir / "checkpoint_latest.tmp.pt"
    final_path = output_dir / "checkpoint_latest.pt"
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)
    return final_path


def parse_epoch_set(raw: str) -> set[int]:
    epochs: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        epoch = int(item)
        if epoch <= 0:
            raise ValueError(f"protected checkpoint epoch must be positive: {epoch}")
        epochs.add(epoch)
    return epochs


def protected_path(output_dir: Path, stem: str, epoch: int, tag: str) -> Path:
    tag_suffix = f"_{tag}" if tag else ""
    return output_dir / f"{stem}_epoch{epoch}_protected{tag_suffix}.pt"


def copy_protected_milestone(
    output_dir: Path,
    *,
    epoch: int,
    model: nn.Module,
    tag: str,
) -> tuple[Path, Path]:
    base_model = unwrap_model(model)
    checkpoint_src = output_dir / "checkpoint_latest.pt"
    checkpoint_dst = protected_path(output_dir, "checkpoint", epoch, tag)
    model_state_dst = protected_path(output_dir, "model_state", epoch, tag)
    if not checkpoint_src.exists():
        raise FileNotFoundError(f"cannot protect missing checkpoint: {checkpoint_src}")
    if not checkpoint_dst.exists():
        tmp_path = checkpoint_dst.with_name(f"{checkpoint_dst.name}.tmp")
        shutil.copy2(checkpoint_src, tmp_path)
        tmp_path.replace(checkpoint_dst)
    if not model_state_dst.exists():
        tmp_path = model_state_dst.with_name(f"{model_state_dst.name}.tmp")
        torch.save(base_model.state_dict(), tmp_path)
        tmp_path.replace(model_state_dst)
    return checkpoint_dst, model_state_dst


def adjusted_hold_steps(*, total_steps: int, warmup_steps: int, hold_steps: int) -> int:
    if warmup_steps + hold_steps >= total_steps:
        return max(total_steps - warmup_steps - 1, 0)
    return hold_steps


def piecewise_50_25_25_learning_rate(
    *,
    step_index: int,
    steps_per_epoch: int,
    peak_lr: float,
    plateau_lr: float,
    final_min_lr: float,
    warmup_fraction: float,
    hold_fraction: float,
) -> float:
    stage1_steps = 50 * steps_per_epoch
    stage2_steps = 25 * steps_per_epoch
    stage3_steps = 25 * steps_per_epoch
    if step_index < stage1_steps:
        warmup_steps = int(round(warmup_fraction * stage1_steps))
        hold_steps = adjusted_hold_steps(
            total_steps=stage1_steps,
            warmup_steps=warmup_steps,
            hold_steps=int(round(hold_fraction * stage1_steps)),
        )
        return scheduled_learning_rate(
            step_index=step_index,
            total_steps=stage1_steps,
            min_lr=plateau_lr,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps,
        )
    if step_index < stage1_steps + stage2_steps:
        return plateau_lr
    return scheduled_learning_rate(
        step_index=max(step_index - stage1_steps - stage2_steps, 0),
        total_steps=stage3_steps,
        min_lr=final_min_lr,
        peak_lr=plateau_lr,
        warmup_steps=0,
        hold_steps=0,
    )


def main() -> None:
    args = parse_args()
    distributed = setup_distributed(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.resume_from is not None else "w"
    log_path = output_dir / ("train.log" if distributed.is_main else f"train.rank{distributed.rank}.log")
    try:
        with log_path.open(log_mode, encoding="utf-8") as log_file:
            run_training(args=args, distributed=distributed, output_dir=output_dir, log_file=log_file)
    finally:
        cleanup_distributed(distributed)


def run_training(
    *,
    args: argparse.Namespace,
    distributed: DistributedContext,
    output_dir: Path,
    log_file: Any,
) -> None:
        log = make_logger(log_file)
        rows = load_caches(
            args.cache,
            split_filter=args.cache_split,
            max_structures_per_row=args.max_structures_per_row,
            progress_log=log,
            progress_label="train_cache",
        )
        validation_rows = (
            load_caches(
                args.validation_cache,
                split_filter=args.validation_split,
                max_structures_per_row=args.max_structures_per_row,
                progress_log=log,
                progress_label="validation_cache",
            )
            if args.validation_cache
            else []
        )
        if not rows:
            raise ValueError("no training rows loaded; check --cache and --cache-split")
        if args.validation_cache and not validation_rows:
            raise ValueError("no validation rows loaded; check --validation-cache and --validation-split")
        max_len = max([row.length for row in rows] + [row.length for row in validation_rows])
        device = distributed.device
        log(f"starting AR output_dir={output_dir}")
        log(f"loaded rows={len(rows)} source_counts={source_counts(rows)} length_summary={length_summary(rows)}")
        if validation_rows:
            log(
                f"loaded validation_rows={len(validation_rows)} "
                f"validation_source_counts={source_counts(validation_rows)} "
                f"validation_length_summary={length_summary(validation_rows)}"
            )
        log(
            f"device={device} distributed={distributed.enabled} "
            f"rank={distributed.rank} local_rank={distributed.local_rank} world_size={distributed.world_size}"
        )
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.set_float32_matmul_precision("high")
        src_tokens = source_vocab_tokens(args.conditioning_mode)
        config = Seq2SeqConfig(
            src_vocab_size=len(src_tokens),
            tgt_vocab_size=len(BASES),
            max_len=max_len,
            conditioning_mode=args.conditioning_mode,
            d_model=args.d_model,
            n_head=args.heads,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            dim_feedforward=args.ffn_size,
            dropout=args.dropout,
            position_encoding=args.position_encoding,
        )
        model = RnaSeq2SeqARTransformer(config).to(device)
        log(f"model_parameters={count_parameters(model)} config={asdict(config)}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        start_epoch = 1
        global_step = 0
        history: list[dict[str, Any]] = []
        if args.resume_from is not None:
            checkpoint = torch.load(args.resume_from, map_location=device)
            checkpoint_config = checkpoint.get("config", {})
            checkpoint_src_vocab_size = checkpoint_config.get("src_vocab_size")
            if checkpoint_src_vocab_size is not None and int(checkpoint_src_vocab_size) != config.src_vocab_size:
                raise ValueError(
                    "resume checkpoint src_vocab_size does not match current conditioning mode: "
                    f"checkpoint={checkpoint_src_vocab_size} current={config.src_vocab_size}"
                )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint.get("global_step", 0))
            history = list(checkpoint.get("history", []))
            log(f"resumed checkpoint={args.resume_from} start_epoch={start_epoch} global_step={global_step}")
        train_model: nn.Module
        if distributed.enabled:
            train_model = DistributedDataParallel(
                model,
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
                gradient_as_bucket_view=True,
            )
        else:
            train_model = model
        torch.manual_seed(args.seed + distributed.rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + distributed.rank)
        initial_examples = build_epoch_examples(
            rows,
            epoch_index=0,
            seed=args.seed,
            conditioning_mode=args.conditioning_mode,
        )
        initial_batches = make_bucket_batches(
            initial_examples,
            batch_size=args.batch_size,
            bucket_width=args.bucket_width,
            seed=args.seed + 1,
        )
        steps_per_epoch = len(
            shard_batches_for_rank(
                initial_batches,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )
        )
        if steps_per_epoch <= 0:
            raise ValueError("no training batches remain after distributed sharding")
        if distributed.enabled:
            usable_initial_batches = steps_per_epoch * distributed.world_size
            log(
                "distributed_batching="
                f"per_gpu_batch_size={args.batch_size} world_size={distributed.world_size} "
                f"global_batch_size={args.batch_size * distributed.world_size} "
                f"raw_batches={len(initial_batches)} usable_batches={usable_initial_batches} "
                f"dropped_batches={len(initial_batches) - usable_initial_batches} "
                f"optimizer_steps_per_epoch={steps_per_epoch}"
            )
        schedule_epochs = args.lr_schedule_epochs if args.lr_schedule_epochs > 0 else args.epochs
        total_steps = schedule_epochs * steps_per_epoch
        warmup_steps = int(round(args.warmup_fraction * total_steps))
        hold_steps = adjusted_hold_steps(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            hold_steps=int(round(args.hold_fraction * total_steps)),
        )
        schedule_start_step = global_step if args.reset_lr_schedule_on_resume else 0
        if args.lr_schedule_mode == "standard":
            log(
                "lr_schedule_mode=standard "
                "lr_schedule_steps="
                f"{total_steps} schedule_epochs={schedule_epochs} schedule_start_step={schedule_start_step} "
                f"warmup_steps={warmup_steps} hold_steps={hold_steps}"
            )
        elif args.lr_schedule_mode == "piecewise_50_25_25":
            if args.epochs < 100:
                raise ValueError("piecewise_50_25_25 requires --epochs >= 100")
            stage1_steps = 50 * steps_per_epoch
            stage2_steps = 25 * steps_per_epoch
            stage3_steps = 25 * steps_per_epoch
            stage1_warmup_steps = int(round(args.warmup_fraction * stage1_steps))
            stage1_hold_steps = adjusted_hold_steps(
                total_steps=stage1_steps,
                warmup_steps=stage1_warmup_steps,
                hold_steps=int(round(args.hold_fraction * stage1_steps)),
            )
            log(
                "lr_schedule_mode=piecewise_50_25_25 "
                f"steps_per_epoch={steps_per_epoch} "
                f"stage1_steps={stage1_steps} stage1_peak_lr={args.learning_rate:.6g} "
                f"stage1_min_lr={args.min_learning_rate:.6g} "
                f"stage1_warmup_steps={stage1_warmup_steps} stage1_hold_steps={stage1_hold_steps} "
                f"stage2_steps={stage2_steps} stage2_lr={args.min_learning_rate:.6g} "
                f"stage3_steps={stage3_steps} stage3_peak_lr={args.min_learning_rate:.6g} "
                f"stage3_min_lr={args.piecewise_final_min_learning_rate:.6g}"
            )
        else:
            raise ValueError(f"unsupported lr_schedule_mode={args.lr_schedule_mode!r}")
        protected_epochs = parse_epoch_set(args.protected_checkpoint_epochs)
        if protected_epochs:
            log(
                "protected_checkpoint_epochs="
                f"{','.join(str(epoch) for epoch in sorted(protected_epochs))} "
                f"protected_checkpoint_tag={args.protected_checkpoint_tag!r}"
            )
        validation_batch_size = args.validation_batch_size if args.validation_batch_size > 0 else args.batch_size
        start = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_examples = build_epoch_examples(
                rows,
                epoch_index=epoch - 1,
                seed=args.seed,
                conditioning_mode=args.conditioning_mode,
            )
            batches = make_bucket_batches(
                epoch_examples,
                batch_size=args.batch_size,
                bucket_width=args.bucket_width,
                seed=args.seed + epoch,
            )
            rank_batches = shard_batches_for_rank(
                batches,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )
            loader = DataLoader(
                ARDataset(epoch_examples),
                batch_sampler=rank_batches,
                num_workers=args.num_workers,
                collate_fn=lambda batch: collate_examples(batch, model=model),
                pin_memory=device.type == "cuda",
            )
            train_model.train()
            totals: dict[str, float] = {}
            total_count = 0.0
            last_lr = args.min_learning_rate
            for batch in loader:
                batch = move_batch(batch, device)
                schedule_step = max(global_step - schedule_start_step, 0)
                if args.lr_schedule_mode == "piecewise_50_25_25":
                    last_lr = piecewise_50_25_25_learning_rate(
                        step_index=schedule_step,
                        steps_per_epoch=steps_per_epoch,
                        peak_lr=args.learning_rate,
                        plateau_lr=args.min_learning_rate,
                        final_min_lr=args.piecewise_final_min_learning_rate,
                        warmup_fraction=args.warmup_fraction,
                        hold_fraction=args.hold_fraction,
                    )
                else:
                    last_lr = scheduled_learning_rate(
                        step_index=schedule_step,
                        total_steps=total_steps,
                        min_lr=args.min_learning_rate,
                        peak_lr=args.learning_rate,
                        warmup_steps=warmup_steps,
                        hold_steps=hold_steps,
                    )
                for group in optimizer.param_groups:
                    group["lr"] = last_lr
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = loss_fn(train_model, batch)
                loss.backward()
                if args.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                count = metrics["token_count"]
                total_count += count
                for key, value in metrics.items():
                    if key == "token_count":
                        continue
                    totals[key] = totals.get(key, 0.0) + value * count
                global_step += 1
            global_total_count = distributed_sum(total_count, context=distributed)
            global_totals = {
                key: distributed_sum(value, context=distributed)
                for key, value in totals.items()
            }
            train_metrics = {key: value / max(global_total_count, 1.0) for key, value in global_totals.items()}
            if distributed.is_main:
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "learning_rate": last_lr,
                    "train": train_metrics,
                    "batch_count": len(rank_batches),
                    "raw_batch_count": len(batches),
                    "world_size": distributed.world_size,
                    "per_gpu_batch_size": args.batch_size,
                    "global_batch_size": args.batch_size * distributed.world_size,
                }
                if validation_rows and args.validation_every > 0 and (epoch % args.validation_every == 0 or epoch == args.epochs):
                    validation_examples = build_epoch_examples(
                        validation_rows,
                        epoch_index=epoch - 1,
                        seed=args.seed + 7_000_001,
                        conditioning_mode=args.conditioning_mode,
                    )
                    record["validation"] = evaluate(
                        model,
                        validation_examples,
                        batch_size=validation_batch_size,
                        bucket_width=args.bucket_width,
                        device=device,
                    )
                if validation_rows and args.feasibility_every > 0 and (epoch % args.feasibility_every == 0 or epoch == args.epochs):
                    record["validation_feasibility"] = evaluate_feasibility(
                        model,
                        validation_rows,
                        epoch_index=epoch - 1,
                        count=args.feasibility_count,
                        temperature=args.feasibility_temperature,
                        seed=args.seed + 8_000_003,
                        device=device,
                        batch_size=args.feasibility_batch_size,
                    )
                history.append(record)
                if (args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0 or epoch == args.epochs)) or epoch in protected_epochs:
                    checkpoint_path = save_checkpoint(
                        output_dir,
                        epoch=epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        history=history,
                        args=args,
                        config=config,
                    )
                    log(f"checkpoint epoch={epoch} path={checkpoint_path}")
                    if epoch in protected_epochs:
                        checkpoint_dst, model_state_dst = copy_protected_milestone(
                            output_dir,
                            epoch=epoch,
                            model=model,
                            tag=args.protected_checkpoint_tag,
                        )
                        log(f"protected_checkpoint epoch={epoch} path={checkpoint_dst}")
                        log(f"protected_model_state epoch={epoch} path={model_state_dst}")
                validation_msg = ""
                if "validation" in record:
                    validation = record["validation"]
                    validation_msg = f" val_loss={validation['loss']:.4f} val_tok_acc={validation['token_accuracy']:.3f}"
                feasibility_msg = ""
                if "validation_feasibility" in record:
                    feasibility = record["validation_feasibility"]
                    feasibility_msg = (
                        f" val_feasible={feasibility['feasibility_rate']:.3f}"
                        f"({int(feasibility['feasible_count'])}/{int(feasibility['count'])})"
                    )
                log(
                    f"epoch={epoch} batches={len(rank_batches)} raw_batches={len(batches)} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"tok_acc={train_metrics['token_accuracy']:.3f} lr={last_lr:.2e}"
                    f"{validation_msg}{feasibility_msg}"
                )
            if distributed.enabled:
                dist.barrier()
        if distributed.is_main:
            torch.save(model.state_dict(), output_dir / "model_final.pt")
            final_checkpoint = save_checkpoint(
                output_dir,
                epoch=args.epochs,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                history=history,
                args=args,
                config=config,
            )
            summary = {
                "args": json_sanitize(vars(args)),
                "source_counts": source_counts(rows),
                "length_summary": length_summary(rows),
                "validation_source_counts": source_counts(validation_rows) if validation_rows else {},
                "validation_length_summary": length_summary(validation_rows) if validation_rows else {},
                "condition_tokens": list(CONDITION_TOKENS),
                "source_tokens": list(src_tokens),
                "conditioning_mode": args.conditioning_mode,
                "target_base_tokens": list(BASES),
                "config": asdict(config),
                "history": json_sanitize(history),
                "distributed": {
                    "enabled": distributed.enabled,
                    "world_size": distributed.world_size,
                    "per_gpu_batch_size": args.batch_size,
                    "global_batch_size": args.batch_size * distributed.world_size,
                    "optimizer_steps_per_epoch": steps_per_epoch,
                },
                "elapsed_sec": time.time() - start,
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log(f"wrote summary path={output_dir / 'summary.json'}")
            log(f"wrote final checkpoint path={final_checkpoint}")
        if distributed.enabled:
            dist.barrier()


if __name__ == "__main__":
    main()
