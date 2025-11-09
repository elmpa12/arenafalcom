"""
Core helpers for selector21 signal processing.

Focused on pure transformations that are easy to test and reason about:
    * feature gates (ATR-Z, VHF, CVD, imbalance)
    * probabilistic combinations (AND / MAJ / SEQ)
    * recency weighting utilities
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .selector21_config import GateConfig


def _ensure_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} deve ser um pandas.Series, recebido {type(series)}")
    return series.copy()


def _threshold_mask(
    df: pd.DataFrame,
    column: str,
    threshold: float,
    *,
    absolute: bool,
    fallback: bool,
) -> pd.Series:
    if threshold <= 0:
        return pd.Series(True, index=df.index)
    if column not in df.columns:
        return pd.Series(fallback, index=df.index)

    series = pd.to_numeric(df[column], errors="coerce")
    if absolute:
        mask = series.abs() >= threshold
    else:
        mask = series >= threshold
    return mask.fillna(False)


def gate_signal_cvd(signal: pd.Series, df: pd.DataFrame, threshold: float) -> pd.Series:
    sig = _ensure_series(signal, "signal")
    mask = _threshold_mask(df, "cvd_slope_agg", threshold, absolute=True, fallback=True)
    gated = sig.where(mask, 0)
    return gated.astype(sig.dtype, copy=False)


def gate_signal_imbalance(signal: pd.Series, df: pd.DataFrame, threshold: float) -> pd.Series:
    sig = _ensure_series(signal, "signal")
    mask = _threshold_mask(df, "imb_net_depth", threshold, absolute=True, fallback=True)
    gated = sig.where(mask, 0)
    return gated.astype(sig.dtype, copy=False)


def gate_signal_atr_zscore(signal: pd.Series, df: pd.DataFrame, threshold: float) -> pd.Series:
    sig = _ensure_series(signal, "signal")
    mask = _threshold_mask(df, "atr_zscore", threshold, absolute=False, fallback=True)
    gated = sig.where(mask, 0)
    return gated.astype(sig.dtype, copy=False)


def gate_signal_vhf(signal: pd.Series, df: pd.DataFrame, threshold: float) -> pd.Series:
    sig = _ensure_series(signal, "signal")
    mask = _threshold_mask(df, "vhf", threshold, absolute=False, fallback=True)
    gated = sig.where(mask, 0)
    return gated.astype(sig.dtype, copy=False)


def apply_signal_gates(signal: pd.Series, df: pd.DataFrame, gates: GateConfig) -> pd.Series:
    sig = _ensure_series(signal, "signal")
    out = gate_signal_cvd(sig, df, gates.cvd_slope_min)
    out = gate_signal_imbalance(out, df, gates.imbalance_min)
    out = gate_signal_atr_zscore(out, df, gates.atr_z_min)
    out = gate_signal_vhf(out, df, gates.vhf_min)
    return out.astype(sig.dtype, copy=False)


def combine_probs(
    prob_series: Sequence[pd.Series],
    op: str,
    *,
    window: int = 1,
    k: int | None = None,
    threshold: float = 0.5,
    neutral: float = 0.5,
) -> pd.Series:
    if not prob_series:
        return pd.Series(dtype=float)

    series_list = [_ensure_series(s.astype(float), f"probs[{i}]") for i, s in enumerate(prob_series)]
    df = pd.concat(series_list, axis=1)
    df = df.apply(lambda col: col.clip(0.0, 1.0))
    arr = np.nan_to_num(df.to_numpy(dtype=float), nan=neutral)
    n_rows, n_cols = arr.shape
    op = op.upper().strip()

    if op == "AND":
        out = np.prod(arr, axis=1)
        return pd.Series(out, index=df.index, name="prob_AND")

    bool_mask = arr >= threshold

    if op == "MAJ":
        if k is None:
            k = (n_cols // 2) + 1
        votes = bool_mask.sum(axis=1)
        active_vals = np.where(bool_mask, arr, np.nan)
        mean_active = np.nanmean(active_vals, axis=1)
        mean_active = np.where(np.isnan(mean_active), neutral, mean_active)
        out = np.where(votes >= k, mean_active, neutral)
        return pd.Series(out, index=df.index, name=f"prob_MAJ{k}")

    if op == "SEQ":
        out = np.full(n_rows, neutral, dtype=float)
        stage = 0
        last_i = -10**9
        max_stage = n_cols
        for i in range(n_rows):
            if stage == 0:
                if bool_mask[i, 0]:
                    stage = 1
                    last_i = i
            else:
                if i - last_i > window:
                    stage = 1 if bool_mask[i, 0] else 0
                    last_i = i if stage == 1 else -10**9
                else:
                    need = stage
                    if need < max_stage and bool_mask[i, need]:
                        stage += 1
                        last_i = i
                        if stage == max_stage:
                            out[i] = arr[i, -1]
                            stage = 0
        return pd.Series(out, index=df.index, name=f"prob_SEQ{window}")

    raise ValueError(f"Operação de combinação inválida: {op}")


def half_life_weights(length: int, half_life: float) -> np.ndarray:
    if length <= 0:
        raise ValueError("length deve ser > 0")
    if half_life <= 0:
        raise ValueError("half_life deve ser > 0")

    idx = np.arange(length, dtype=float)
    decay = np.exp(np.log(0.5) * (length - 1 - idx) / half_life)
    total = decay.sum()
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Falha ao normalizar pesos – verifique o half_life fornecido.")
    return decay / total


def apply_recency_weights(values: Sequence[float], half_life: float) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    weights = half_life_weights(len(arr), half_life)
    return arr * weights


__all__ = [
    "apply_recency_weights",
    "apply_signal_gates",
    "combine_probs",
    "gate_signal_atr_zscore",
    "gate_signal_cvd",
    "gate_signal_imbalance",
    "gate_signal_vhf",
    "half_life_weights",
]
