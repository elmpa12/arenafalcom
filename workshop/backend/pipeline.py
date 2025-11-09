"""Shared backend contracts between selector21 and the deep-learning heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class TimeSpan:
    """Representa uma janela de tempo inclusiva usada para buscar dados."""

    start: datetime
    end: datetime

    def as_millis(self) -> tuple[int, int]:
        """Retorna ``(start, end)`` em timestamps UTC (ms)."""

        return (int(self.start.timestamp() * 1000), int(self.end.timestamp() * 1000))


@dataclass
class FeatureRequest:
    """Informações mínimas para carregar/enriquecer um timeframe."""

    symbol: str
    timeframe: str
    data_root: str
    agg_dir: Optional[str] = None
    depth_dir: Optional[str] = None


@dataclass
class FeatureSnapshot:
    """Container simples com os dados enriquecidos de um timeframe."""

    timeframe: str
    frame: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalSet:
    """Agrupa sinais base e combinados produzidos pelo selector."""

    timeframe: str
    base: pd.DataFrame = field(default_factory=pd.DataFrame)
    combos: pd.DataFrame = field(default_factory=pd.DataFrame)

    def copy(self) -> "SignalSet":
        return SignalSet(
            timeframe=self.timeframe,
            base=self.base.copy(deep=True),
            combos=self.combos.copy(deep=True),
        )


@dataclass
class DeepLearningDataset:
    """Payload estruturado consumido pelo ``dl_head``."""

    timeframe: str
    train: pd.DataFrame
    validation: pd.DataFrame
    horizon: int
    lags: int
    label_column: str = "y"
    return_column: str = "future_ret"
    weight_column: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        drop = {self.label_column, self.return_column, "time"}
        if self.weight_column:
            drop.add(self.weight_column)
        return [c for c in self.train.columns if c not in drop]

    def iter_frames(self) -> Iterable[pd.DataFrame]:
        yield self.train
        yield self.validation


@dataclass
class DeepLearningResult:
    """Resumo após treinar um head de deep-learning."""

    timeframe: str
    feature_columns: Sequence[str]
    metrics: Dict[str, Dict[str, float]]
    predictions: pd.DataFrame
    model_state: Dict[str, Any]
    best_threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
