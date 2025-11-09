from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PriceBar(BaseModel):
    model_config = ConfigDict(frozen=True)
    t: str
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True)
    t: str
    side: Literal["BUY", "SELL"]
    reason: Optional[str] = None
    conf: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pipeline: Optional[str] = None


class Trade(BaseModel):
    model_config = ConfigDict(frozen=True)
    entry_t: str
    entry_px: float
    exit_t: Optional[str] = None
    exit_px: Optional[float] = None
    side: Literal["LONG", "SHORT"]
    size: Optional[float] = None
    pnl: Optional[float] = None
    pnl_abs: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    pipeline: Optional[str] = None
    meta: Optional[Dict[str, object]] = None


class Frame(BaseModel):
    bar: PriceBar
    signals: List[Signal] = Field(default_factory=list)
    trades_open: List[Trade] = Field(default_factory=list)
    trades_closed: List[Trade] = Field(default_factory=list)
    equity: Optional[float] = None


class PipelineMeta(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = None
    edge: Optional[float] = None
    kpis: Dict[str, float] = Field(default_factory=dict)


class BacktestMeta(BaseModel):
    id: str
    symbol: str
    timeframe: str
    start: str
    end: str
    n_frames: int
    n_trades: int
    params: Dict[str, object] = Field(default_factory=dict)
    kpis: Dict[str, float] = Field(default_factory=dict)
    pipelines: List[PipelineMeta] = Field(default_factory=list)


class BacktestSummary(BaseModel):
    id: str
    symbol: str
    timeframe: str
    start: str
    end: str
    n_frames: int
    size_bytes: int


class FramePage(BaseModel):
    frames: List[Frame]
    offset: int
    limit: int
    total: int
    kpis: Dict[str, float]


class TradeTable(BaseModel):
    backtest_id: str
    trades: List[Trade]


class KPIRequest(BaseModel):
    at: Optional[str] = None
    index: Optional[int] = Field(default=None, ge=0)
