"""Falcom BotScalp — paper trading executor (stub).

This module converts selector outputs (signals, probabilities, MERGED_meta CSVs)
into simulated trades with costs, allowing Alfred to plug selectors into paper
trading quickly. All heavy lifting (thresholding, stops, risk) should extend the
`PaperExecutor` class or helper functions below.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import json
import math
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_px: float
    exit_px: float
    side: str  # long | short
    size: float
    pnl: float
    fees: float
    reason: str  # stop | tp | timeout | signal_flip


@dataclass
class ExecutionConfig:
    notional_usd: float
    tick_size: float
    fee_perc: float
    slippage_ticks: int
    neutral_band: float
    prob_thr_long: float
    prob_thr_short: float
    atr_stop_mult: float
    timeout_bars: int
    max_hold_bars: Dict[str, int]

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExecutionConfig":
        account = cfg.get("account", {})
        risk = cfg.get("risk", {})
        execution = cfg.get("execution", {})
        selector = cfg.get("selector", {})
        return cls(
            notional_usd=float(account.get("notional_usd", 100.0)),
            tick_size=float(account.get("tick_size", 0.1)),
            fee_perc=float(account.get("fee_perc", 0.0004)),
            slippage_ticks=int(account.get("slippage_ticks", 1)),
            neutral_band=float(execution.get("neutral_band", 0.02)),
            prob_thr_long=float(selector.get("prob_threshold_long", 0.55)),
            prob_thr_short=float(selector.get("prob_threshold_short", 0.45)),
            atr_stop_mult=float(risk.get("atr_stop_mult", 3.0)),
            timeout_bars=int(risk.get("timeout_bars", 90)),
            max_hold_bars=risk.get("max_hold_bars", {"1m": 180}),
        )


class PaperExecutor:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self.trades: List[Trade] = []

    def run(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Simulate trades from a signals DataFrame.

        Expected columns: time, close, prob_long, prob_short, atr, side_hint, etc.
        Alfred will implement the full state machine; for now we log TODO markers.
        """
        if signals.empty:
            return pd.DataFrame()
        # TODO: implement position management, stops, fees, etc.
        # Placeholder returns aggregated metrics with zero trades.
        metrics = {
            "pnl": 0.0,
            "n_trades": 0,
            "winrate": 0.0,
            "profit_factor": math.nan,
            "sharpe": math.nan,
            "max_drawdown": 0.0,
        }
        return pd.DataFrame([metrics])

    @staticmethod
    def load_signals(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df


def load_config(path: Path) -> ExecutionConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return ExecutionConfig.from_dict(raw)


def run_from_csv(csv_path: Path, cfg_path: Path) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    executor = PaperExecutor(cfg)
    df = executor.load_signals(csv_path)
    return executor.run(df)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Paper executor stub")
    ap.add_argument("--signals", required=True, help="MERGED_meta csv or selector export")
    ap.add_argument("--config", default="configs/trading.json")
    ap.add_argument("--out", default="reports/paper_metrics.csv")
    args = ap.parse_args()

    result = run_from_csv(Path(args.signals), Path(args.config))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"[paper_executor] metrics → {args.out}")
