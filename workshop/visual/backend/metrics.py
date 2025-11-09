from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional, Sequence

try:
    from .models import Frame, Trade
except ImportError:  # script execution fallback
    from models import Frame, Trade


def _safe_float(value: Optional[float]) -> float:
    return float(value) if value is not None else 0.0


def sharpe_ratio(equity_curve: Sequence[float], risk_free: float = 0.0) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = []
    for prev, curr in zip(equity_curve, equity_curve[1:]):
        if prev <= 0:
            continue
        returns.append((curr - prev) / prev - risk_free)
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    if variance <= 0:
        return 0.0
    return mean / sqrt(variance) * sqrt(252)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    max_eq = 0.0
    max_dd = 0.0
    for equity in equity_curve:
        if equity > max_eq:
            max_eq = equity
        drawdown = max_eq - equity
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def winrate(trades: Sequence[Trade]) -> float:
    closed = [t for t in trades if t.pnl is not None]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if _safe_float(t.pnl) > 0)
    return wins / len(closed)


def profit_factor(trades: Sequence[Trade]) -> float:
    gross_profit = sum(_safe_float(t.pnl) for t in trades if _safe_float(t.pnl) > 0)
    gross_loss = sum(_safe_float(t.pnl) for t in trades if _safe_float(t.pnl) < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def expectancy(trades: Sequence[Trade]) -> float:
    if not trades:
        return 0.0
    pnl = [ _safe_float(t.pnl) for t in trades if t.pnl is not None ]
    if not pnl:
        return 0.0
    return sum(pnl) / len(pnl)


def avg_trade(trades: Sequence[Trade]) -> float:
    return expectancy(trades)


def hit_ratio_by_side(trades: Sequence[Trade]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for side in ("LONG", "SHORT"):
        subset = [t for t in trades if t.side == side and t.pnl is not None]
        if not subset:
            result[side] = 0.0
            continue
        wins = sum(1 for t in subset if _safe_float(t.pnl) > 0)
        result[side] = wins / len(subset)
    return result


def compute_kpis(trades: Sequence[Trade], equity_curve: Sequence[float]) -> Dict[str, float]:
    closed = [t for t in trades if t.pnl is not None]
    hits = hit_ratio_by_side(closed)
    return {
        "winrate": winrate(closed),
        "sharpe": sharpe_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve) if equity_curve else 0.0,
        "profit_factor": profit_factor(closed),
        "expectancy": expectancy(closed),
        "avg_trade": avg_trade(closed),
        "n_trades": float(len(closed)),
        "hit_long": hits.get("LONG", 0.0),
        "hit_short": hits.get("SHORT", 0.0),
    }


@dataclass
class KPIAccumulator:
    risk_free: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    _frames: List[Frame] = field(default_factory=list)

    def ingest_frame(self, frame: Frame) -> None:
        self._frames.append(frame)
        if frame.equity is not None:
            self.equity_curve.append(frame.equity)
        if frame.trades_closed:
            self.trades.extend(frame.trades_closed)

    def ingest_batch(self, frames: Sequence[Frame]) -> None:
        for frame in frames:
            self.ingest_frame(frame)

    def snapshot(self) -> Dict[str, float]:
        return compute_kpis(self.trades, self.equity_curve)

    def snapshot_until(self, timestamp: str) -> Dict[str, float]:
        eq: List[float] = []
        trades: List[Trade] = []
        for frame in self._frames:
            if frame.bar.t > timestamp:
                break
            if frame.equity is not None:
                eq.append(frame.equity)
            trades.extend(frame.trades_closed)
        return compute_kpis(trades, eq)
