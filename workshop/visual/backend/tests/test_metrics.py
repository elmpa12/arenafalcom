from __future__ import annotations

import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from metrics import KPIAccumulator, compute_kpis, max_drawdown, sharpe_ratio
from models import Frame, PriceBar, Trade


def _trade(**kwargs):
    return Trade(**kwargs)


def _frame(eq: float, trades: list[Trade]):
    return Frame(bar=PriceBar(t="2025-01-01T00:00:00Z", o=1, h=1, l=1, c=1, v=0), equity=eq, trades_closed=trades)


def test_basic_metrics():
    trades = [
        _trade(entry_t="t0", entry_px=1.0, exit_t="t1", exit_px=1.1, side="LONG", pnl=10.0),
        _trade(entry_t="t2", entry_px=1.0, exit_t="t3", exit_px=0.9, side="SHORT", pnl=-5.0),
    ]
    equity = [100.0, 105.0, 110.0]
    kpis = compute_kpis(trades, equity)
    assert kpis["winrate"] == pytest.approx(0.5)
    assert kpis["profit_factor"] == pytest.approx(10.0 / 5.0)
    assert kpis["expectancy"] == pytest.approx(2.5)
    assert kpis["avg_trade"] == pytest.approx(2.5)
    assert kpis["n_trades"] == 2.0


def test_drawdown():
    equity = [100.0, 110.0, 90.0, 95.0]
    assert max_drawdown(equity) == pytest.approx(20.0)


def test_sharpe():
    equity = [100.0, 101.0, 103.0, 102.0]
    val = sharpe_ratio(equity)
    assert isinstance(val, float)


def test_accumulator_snapshot():
    trades = [
        _trade(entry_t="t0", entry_px=1.0, exit_t="t1", exit_px=1.1, side="LONG", pnl=5.0),
    ]
    frames = [
        _frame(100.0, []),
        _frame(101.0, trades),
    ]
    acc = KPIAccumulator()
    acc.ingest_batch(frames)
    snap = acc.snapshot()
    assert snap["n_trades"] == 1.0
    assert snap["winrate"] == 1.0


def test_frame_schema_roundtrip():
    payload = {
        "bar": {"t": "2025-01-01T00:00:00Z", "o": 100.0, "h": 110.0, "l": 95.0, "c": 105.0, "v": 1234},
        "signals": [
            {"t": "2025-01-01T00:00:00Z", "side": "BUY", "reason": "selector", "conf": 0.7},
        ],
        "trades_open": [],
        "trades_closed": [
            {
                "entry_t": "2025-01-01T00:00:00Z",
                "entry_px": 100.0,
                "exit_t": "2025-01-01T00:05:00Z",
                "exit_px": 104.0,
                "side": "LONG",
                "pnl": 4.0,
            }
        ],
        "equity": 100005.0,
    }
    frame = Frame.model_validate(payload)
    assert frame.bar.c == 105.0
    assert frame.signals[0].side == "BUY"
    dump = frame.model_dump()
    assert dump["bar"]["o"] == 100.0
