"""Reusable utilities for building sequential datasets for DL heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def make_label(df: pd.DataFrame, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """Return (future_return, binary_label) series for the provided dataframe."""

    ret = (df["close"].shift(-horizon) / df["close"] - 1.0).astype(np.float32)
    y = (ret > 0).astype(np.float32)
    return ret, y


def number_cols(df: pd.DataFrame) -> List[str]:
    """List numeric feature columns (excluding ``time``) preserving order."""

    drop = {"time"}
    return [
        col
        for col in df.columns
        if col not in drop and np.issubdtype(np.asarray(df[col]).dtype, np.number)
    ]


def build_sequences_chunked(
    df: pd.DataFrame,
    feats: Sequence[str],
    lags: int,
    horizon: int,
    *,
    chunk_rows: int = 250_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a dataframe into (X, y, return, index) tensors using sliding windows.

    The implementation mirrors ``dl_heads_v8`` to keep compatibility between
    locally generated sequences and the remote GPU training step.
    """

    if not feats:
        return (
            np.zeros((0, lags, 0), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int64),
        )

    arr = df[feats].astype(np.float32).to_numpy()
    n_rows, n_feats = arr.shape
    close = df["close"].to_numpy()

    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    Rs: list[np.ndarray] = []
    idx_blocks: list[np.ndarray] = []

    start = lags - 1
    end = n_rows - horizon
    if end <= start:
        return (
            np.zeros((0, lags, n_feats), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int64),
        )

    i = start
    while i < end:
        j = min(end, i + chunk_rows)

        ret = (close[i + horizon : j + horizon] / close[i:j] - 1.0).astype(np.float32)
        y = (ret > 0).astype(np.float32)
        seq = np.lib.stride_tricks.sliding_window_view(
            arr[i - lags + 1 : j + 1], window_shape=(lags, n_feats)
        )[:, 0, :, :]
        idx = np.arange(i, j, dtype=np.int64)

        usable = min(len(ret), seq.shape[0])
        if usable <= 0:
            i = j
            continue

        Xs.append(seq[:usable])
        Ys.append(y[:usable])
        Rs.append(ret[:usable])
        idx_blocks.append(idx[:usable])
        i = j

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    R = np.concatenate(Rs, axis=0)
    IDX = np.concatenate(idx_blocks, axis=0)
    return X, Y, R, IDX


def weekly_windows(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    train_weeks: int,
    val_weeks: int,
    step_weeks: int,
) -> list[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate walk-forward windows using weekly steps."""

    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    windows: list[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cur = start_ts
    while True:
        train_end = cur + pd.Timedelta(weeks=train_weeks) - pd.Timedelta(minutes=1)
        val_end = train_end + pd.Timedelta(weeks=val_weeks)
        if train_end >= end_ts or cur >= end_ts:
            break
        windows.append((cur, train_end, train_end + pd.Timedelta(minutes=1), min(val_end, end_ts)))
        cur = cur + pd.Timedelta(weeks=step_weeks)
        if cur >= end_ts:
            break
    return windows


def _month_split_points(month_start: pd.Timestamp, splits: int) -> list[pd.Timestamp]:
    month_end = (month_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(minutes=1)
    days = (month_end - month_start + pd.Timedelta(minutes=1)).days or 1
    points = [month_start]
    for i in range(1, splits):
        delta_days = int(round(days * i / splits))
        points.append(month_start + pd.Timedelta(days=delta_days))
    points.append(month_end + pd.Timedelta(minutes=1))
    return points


def monthly_windows(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    train_months: float = 0.5,
    val_months: float = 0.25,
    step_months: float = 0.25,
) -> list[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate walk-forward windows aligned to calendar months."""

    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    splits = int(round(1 / step_months))
    if splits not in (1, 2, 4):
        splits = 2

    cur = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz="UTC")
    if cur < start_ts:
        cur = cur + pd.offsets.MonthBegin(1)

    edges: list[pd.Timestamp] = []
    while cur < end_ts:
        edges.extend(_month_split_points(cur, splits)[:-1])
        cur = cur + pd.offsets.MonthBegin(1)
    edges.append(
        pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz="UTC")
        + pd.offsets.MonthBegin(1)
    )

    train_steps = max(1, int(round(train_months / step_months)))
    val_steps = max(1, int(round(val_months / step_months)))
    windows: list[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    idx = 0
    while True:
        train_start_idx = idx
        train_end_idx = idx + train_steps
        val_end_idx = train_end_idx + val_steps
        if val_end_idx >= len(edges):
            break
        train_start = edges[train_start_idx]
        train_end = edges[train_end_idx] - pd.Timedelta(minutes=1)
        val_start = edges[train_end_idx]
        val_end = min(edges[val_end_idx] - pd.Timedelta(minutes=1), end_ts)
        if train_start >= end_ts:
            break
        windows.append((train_start, train_end, val_start, val_end))
        idx += 1
    return windows


__all__ = [
    "build_sequences_chunked",
    "make_label",
    "monthly_windows",
    "number_cols",
    "weekly_windows",
]
