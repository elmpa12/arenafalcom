"""Common deep-learning head definitions shared across selector tooling.

This module centralises the creation and validation of the sequence models
used by :mod:`dl_heads_v8` and :mod:`selector21`.  It keeps the list of heads in
one place so that CLIs and runtime configuration files remain consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class HeadDefinition:
    """Describe a supported deep-learning head."""

    name: str
    display_name: str
    builder: Callable[[int, int], nn.Module]
    description: str = ""


# ---------------------------------------------------------------------------
# Head implementations
# ---------------------------------------------------------------------------


class GRUDeep(nn.Module):
    """Stacked GRU followed by a linear classifier."""

    def __init__(self, n_features: int, hidden: int = 512, layers: int = 3, drop: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=drop if layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.out(last).squeeze(-1)


class LSTMDeep(nn.Module):
    """Stacked LSTM analogue of :class:`GRUDeep`."""

    def __init__(self, n_features: int, hidden: int = 512, layers: int = 3, drop: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=drop if layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.out(last).squeeze(-1)


class CNNResidualBlock(nn.Module):
    """Dilated residual block inspired by temporal convolutional networks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = self.residual(x)
        y = self.conv1(x)
        y = self.act1(self.bn1(y))
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(self.bn2(y))
        return self.out_act(y + residual)


class CNNDeep(nn.Module):
    """Temporal CNN with dilated residual blocks and global pooling."""

    def __init__(
        self,
        n_features: int,
        channels: Sequence[int] = (256, 256, 128),
        kernel_size: int = 3,
        dropout: float = 0.15,
        dilations: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("'channels' deve conter pelo menos um valor")
        if dilations is None:
            dilations = tuple(2 ** idx for idx in range(len(channels)))
        elif len(dilations) != len(channels):
            raise ValueError("'dilations' deve ter o mesmo tamanho de 'channels'")

        stem_width = channels[0]
        self.stem = nn.Conv1d(n_features, stem_width, kernel_size=1)

        blocks: List[nn.Module] = []
        in_ch = stem_width
        for ch, dil in zip(channels, dilations):
            blocks.append(CNNResidualBlock(in_ch, ch, kernel_size, dil, dropout))
            in_ch = ch
        self.body = nn.Sequential(*blocks)
        self.post_norm = nn.BatchNorm1d(in_ch)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = x.transpose(1, 2)
        h = self.body(self.stem(z))
        h = self.post_norm(h)
        pooled = h.mean(dim=-1)
        pooled = self.dropout(pooled)
        return self.out(pooled).squeeze(-1)


class DenseDeep(nn.Module):
    """Dense head that flattens the sequence and feeds it through MLP layers."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_sizes: Sequence[int] = (1024, 512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = n_features * seq_len
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = size
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        flat = x.reshape(x.shape[0], -1)
        return self.net(flat).squeeze(-1)


class SinePositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding usable for any sequence length."""

    def __init__(self, d_model: int, *, max_len: int = 2048) -> None:
        super().__init__()
        self.d_model = d_model
        initial = self._build_table(max(1, max_len))
        # ``persistent=False`` evita salvar um tensor possivelmente grande em checkpoints.
        self.register_buffer("_pe", initial, persistent=False)
        # cacheia versões convertidas por device/dtype para evitar ``.to`` a cada forward
        self._device_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _build_table(self, length: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        base = torch.arange(0, self.d_model, 2, dtype=torch.float32)
        div_term = torch.exp(base * (-math.log(10000.0) / max(1, self.d_model)))
        pe = torch.zeros(length, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def _positional_slice(self, length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._pe.size(1) < length:
            self._pe = self._build_table(length)
            self._device_cache.clear()

        key = (device, dtype)
        cached = self._device_cache.get(key)
        if cached is None or cached.size(1) < length:
            cached = self._pe.to(device=device, dtype=dtype)
            self._device_cache[key] = cached
        return cached[:, :length, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        pe = self._positional_slice(x.size(1), device=x.device, dtype=x.dtype)
        return x + pe


class TransformerDeep(nn.Module):
    """Transformer encoder head with sinusoidal positional encodings."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 384,
        nhead: int = 8,
        layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.positional = SinePositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.input_proj(x)
        z = self.positional(z)
        h = self.encoder(z)
        h_last = self.norm(h[:, -1, :])
        h_last = self.dropout(h_last)
        return self.out(h_last).squeeze(-1)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _build_gru(n_features: int, seq_len: int) -> nn.Module:
    return GRUDeep(n_features)


def _build_lstm(n_features: int, seq_len: int) -> nn.Module:
    return LSTMDeep(n_features)


def _build_cnn(n_features: int, seq_len: int) -> nn.Module:
    return CNNDeep(n_features)


def _build_dense(n_features: int, seq_len: int) -> nn.Module:
    return DenseDeep(n_features, seq_len)


def _build_transformer(n_features: int, seq_len: int) -> nn.Module:
    return TransformerDeep(n_features, seq_len)


HEAD_REGISTRY: Dict[str, HeadDefinition] = {
    "gru": HeadDefinition(
        name="gru",
        display_name="GRU",
        builder=_build_gru,
        description="Stacked GRU with dropout and linear classifier.",
    ),
    "lstm": HeadDefinition(
        name="lstm",
        display_name="LSTM",
        builder=_build_lstm,
        description="Stacked LSTM mirroring the GRU topology.",
    ),
    "cnn": HeadDefinition(
        name="cnn",
        display_name="CNN",
        builder=_build_cnn,
        description="Dilated residual CNN capturando padrões temporais multi-escala.",
    ),
    "transformer": HeadDefinition(
        name="transformer",
        display_name="Transformer",
        builder=_build_transformer,
        description="Encoder de atenção para dependências de longo alcance.",
    ),
    "dense": HeadDefinition(
        name="dense",
        display_name="DENSE",
        builder=_build_dense,
        description="Feed-forward network on flattened sequences.",
    ),
}


def available_head_names() -> List[str]:
    """Return the list of supported head identifiers."""

    return list(HEAD_REGISTRY.keys())


def describe_heads() -> Dict[str, str]:
    """Return short descriptions for each registered head."""

    return {name: definition.description for name, definition in HEAD_REGISTRY.items()}


def parse_head_list(raw: Iterable[str] | str) -> List[str]:
    """Normalise a comma-separated or iterable list of heads."""

    if isinstance(raw, str):
        tokens = [part.strip().lower() for part in raw.split(",")]
    else:
        tokens = [str(part).strip().lower() for part in raw]
    return [token for token in tokens if token]


def resolve_requested_heads(requested: Iterable[str] | str) -> List[HeadDefinition]:
    """Validate and resolve the requested heads.

    ``requested`` may contain the magic value ``"all"`` to select every
    registered head.  Raises :class:`ValueError` when an unknown name is
    supplied.
    """

    tokens = parse_head_list(requested)
    if not tokens:
        return []
    if "all" in tokens:
        return list(HEAD_REGISTRY.values())

    invalid = [name for name in tokens if name not in HEAD_REGISTRY]
    if invalid:
        allowed = ", ".join(sorted(available_head_names()))
        raise ValueError(f"Heads inválidos: {', '.join(sorted(set(invalid)))}. Disponíveis: {allowed}.")

    unique: List[str] = []
    for name in tokens:
        if name not in unique:
            unique.append(name)
    return [HEAD_REGISTRY[name] for name in unique]


def instantiate_head(name: str, feature_dim: int, seq_len: int) -> nn.Module:
    """Instantiate a head by name."""

    key = str(name).strip().lower()
    if key not in HEAD_REGISTRY:
        allowed = ", ".join(sorted(available_head_names()))
        raise ValueError(f"Head desconhecido '{name}'. Use um de: {allowed}")
    return HEAD_REGISTRY[key].builder(feature_dim, seq_len)


__all__ = [
    "HeadDefinition",
    "HEAD_REGISTRY",
    "available_head_names",
    "describe_heads",
    "instantiate_head",
    "parse_head_list",
    "resolve_requested_heads",
    "CNNResidualBlock",
    "CNNDeep",
    "DenseDeep",
    "GRUDeep",
    "LSTMDeep",
    "SinePositionalEncoding",
    "TransformerDeep",
]
