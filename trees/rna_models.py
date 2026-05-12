from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RnaARConfig:
    base_vocab_size: int
    struct_vocab_size: int
    max_len: int
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    position_encoding: str = "learned"


@dataclass(frozen=True)
class RnaMaskedConfig:
    base_vocab_size: int
    struct_vocab_size: int
    max_len: int
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    position_encoding: str = "learned"


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    table = torch.zeros(max_len, d_model, dtype=torch.float32)
    table[:, 0::2] = torch.sin(positions * div_term)
    table[:, 1::2] = torch.cos(positions * div_term[: table[:, 1::2].shape[1]])
    return table


def alibi_slopes(num_heads: int) -> torch.Tensor:
    return 2.0 ** (-torch.arange(1, num_heads + 1, dtype=torch.float32) / num_heads)


def alibi_mask(
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    causal: bool,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    slopes = alibi_slopes(num_heads).to(device).view(num_heads, 1, 1)
    bias = -slopes * distance.unsqueeze(0)
    if causal:
        future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        bias = bias.masked_fill(future.unsqueeze(0), -torch.inf)
    return bias.unsqueeze(0).expand(batch_size, num_heads, seq_len, seq_len).reshape(
        batch_size * num_heads, seq_len, seq_len
    )


class FractionalPositionEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = valid_mask.shape
        device = valid_mask.device
        lengths = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        denom = (lengths - 1.0).clamp(min=1.0)
        positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, seq_len)
        left_frac = positions / denom
        right_frac = (lengths - 1.0 - positions).clamp(min=0.0) / denom
        centered = 2.0 * left_frac - 1.0
        inv_length = (1.0 / lengths).expand(batch_size, seq_len)
        features = torch.stack(
            [left_frac, right_frac, centered, inv_length, valid_mask.float()],
            dim=-1,
        )
        return self.net(features)


class RnaARTransformer(nn.Module):
    def __init__(self, config: RnaARConfig) -> None:
        super().__init__()
        self.config = config
        self.bos_id = config.struct_vocab_size
        self.pad_id = config.struct_vocab_size + 1
        self.base_pad_id = config.base_vocab_size
        self.struct_embedding = nn.Embedding(config.struct_vocab_size + 2, config.d_model, padding_idx=self.pad_id)
        self.base_embedding = nn.Embedding(config.base_vocab_size + 1, config.d_model, padding_idx=self.base_pad_id)
        self.use_alibi = config.position_encoding == "alibi"
        if config.position_encoding == "learned":
            self.position_embedding = nn.Embedding(config.max_len, config.d_model)
            self.fractional_position_embedding = None
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        elif config.position_encoding == "sinusoidal":
            self.position_embedding = None
            self.fractional_position_embedding = None
            self.register_buffer(
                "sinusoidal_position_table",
                sinusoidal_positions(config.max_len, config.d_model),
                persistent=False,
            )
        elif config.position_encoding == "fractional":
            self.position_embedding = None
            self.fractional_position_embedding = FractionalPositionEmbedding(config.d_model)
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        elif config.position_encoding == "alibi":
            self.position_embedding = None
            self.fractional_position_embedding = None
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        elif config.position_encoding == "none":
            self.position_embedding = None
            self.fractional_position_embedding = None
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        else:
            raise ValueError(f"unknown position_encoding={config.position_encoding}")
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layer, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.struct_vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, prev_struct_ids: torch.Tensor, base_ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = prev_struct_ids.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len={self.config.max_len}")
        positions = torch.arange(seq_len, device=prev_struct_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        if self.position_embedding is not None:
            position_values = self.position_embedding(positions)
        elif self.fractional_position_embedding is not None:
            position_values = self.fractional_position_embedding(valid_mask)
        elif self.sinusoidal_position_table.numel() > 0:
            position_values = self.sinusoidal_position_table[:seq_len].to(prev_struct_ids.device).unsqueeze(0)
        else:
            position_values = torch.zeros_like(self.struct_embedding(prev_struct_ids))
        x = self.struct_embedding(prev_struct_ids) + self.base_embedding(base_ids) + position_values
        if self.use_alibi:
            causal_mask = alibi_mask(
                batch_size=batch_size,
                num_heads=self.config.n_head,
                seq_len=seq_len,
                causal=True,
                device=prev_struct_ids.device,
            )
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=prev_struct_ids.device),
                diagonal=1,
            )
        key_padding_mask = ~valid_mask
        if self.use_alibi:
            key_padding_mask = torch.zeros_like(valid_mask, dtype=x.dtype).masked_fill(~valid_mask, -torch.inf)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.head(self.final_norm(x))


class RnaMaskedTransformer(nn.Module):
    def __init__(self, config: RnaMaskedConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_id = config.struct_vocab_size
        self.pad_id = config.struct_vocab_size + 1
        self.base_pad_id = config.base_vocab_size
        self.struct_embedding = nn.Embedding(config.struct_vocab_size + 2, config.d_model, padding_idx=self.pad_id)
        self.base_embedding = nn.Embedding(config.base_vocab_size + 1, config.d_model, padding_idx=self.base_pad_id)
        self.use_alibi = config.position_encoding == "alibi"
        if config.position_encoding == "learned":
            self.position_embedding = nn.Embedding(config.max_len, config.d_model)
            self.fractional_position_embedding = None
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        elif config.position_encoding == "sinusoidal":
            self.position_embedding = None
            self.fractional_position_embedding = None
            self.register_buffer(
                "sinusoidal_position_table",
                sinusoidal_positions(config.max_len, config.d_model),
                persistent=False,
            )
        elif config.position_encoding == "fractional":
            self.position_embedding = None
            self.fractional_position_embedding = FractionalPositionEmbedding(config.d_model)
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        elif config.position_encoding == "alibi":
            self.position_embedding = None
            self.fractional_position_embedding = None
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        elif config.position_encoding == "none":
            self.position_embedding = None
            self.fractional_position_embedding = None
            self.register_buffer("sinusoidal_position_table", torch.empty(0), persistent=False)
        else:
            raise ValueError(f"unknown position_encoding={config.position_encoding}")
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layer, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.position_head = nn.Linear(config.d_model, 1)
        self.token_head = nn.Linear(config.d_model, config.struct_vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        partial_struct_ids: torch.Tensor,
        base_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = partial_struct_ids.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len={self.config.max_len}")
        positions = torch.arange(seq_len, device=partial_struct_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        if self.position_embedding is not None:
            position_values = self.position_embedding(positions)
        elif self.fractional_position_embedding is not None:
            position_values = self.fractional_position_embedding(valid_mask)
        elif self.sinusoidal_position_table.numel() > 0:
            position_values = self.sinusoidal_position_table[:seq_len].to(partial_struct_ids.device).unsqueeze(0)
        else:
            position_values = torch.zeros_like(self.struct_embedding(partial_struct_ids))
        x = self.struct_embedding(partial_struct_ids) + self.base_embedding(base_ids) + position_values
        attn_mask = None
        if self.use_alibi:
            attn_mask = alibi_mask(
                batch_size=batch_size,
                num_heads=self.config.n_head,
                seq_len=seq_len,
                causal=False,
                device=partial_struct_ids.device,
            )
        key_padding_mask = ~valid_mask
        if self.use_alibi:
            key_padding_mask = torch.zeros_like(valid_mask, dtype=x.dtype).masked_fill(~valid_mask, -torch.inf)
        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        return self.position_head(x).squeeze(-1), self.token_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
