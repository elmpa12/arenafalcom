from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # optional dependency
    pd = None  # type: ignore[assignment]

try:
    from .models import BacktestMeta, BacktestSummary, Frame, PipelineMeta, Trade
except ImportError:  # script execution fallback
    from models import BacktestMeta, BacktestSummary, Frame, PipelineMeta, Trade

SUPPORTED_FRAME_FILES = ("frames.jsonl", "frames.ndjson", "frames.csv", "frames.parquet")
SUPPORTED_META_FILES = ("meta.json", "meta.yaml", "meta.yml")
SUPPORTED_TRADE_FILES = ("trades.jsonl", "trades.ndjson", "trades.csv", "trades.parquet")


@dataclass(frozen=True)
class BacktestHandle:
    backtest_id: str
    base_path: Path
    frames_path: Path
    trades_path: Optional[Path]
    meta_path: Optional[Path]

    @property
    def size_bytes(self) -> int:
        total = 0
        for candidate in [self.frames_path, self.trades_path, self.meta_path]:
            if candidate and candidate.exists():
                total += candidate.stat().st_size
        return total


def discover_backtests(root: Path) -> Dict[str, BacktestHandle]:
    if not root.exists():
        raise FileNotFoundError(f"VISUAL_DATA_ROOT missing: {root}")
    handles: Dict[str, BacktestHandle] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        frames = _find_first(entry, SUPPORTED_FRAME_FILES)
        if not frames:
            continue
        trades = _find_first(entry, SUPPORTED_TRADE_FILES)
        meta = _find_first(entry, SUPPORTED_META_FILES)
        handle = BacktestHandle(
            backtest_id=entry.name,
            base_path=entry,
            frames_path=frames,
            trades_path=trades,
            meta_path=meta,
        )
        handles[entry.name] = handle
    return handles


def _find_first(base: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=128)
def load_meta(handle: BacktestHandle) -> BacktestMeta:
    meta = _load_meta_raw(handle)
    frames_count = meta.get("n_frames") or _count_frame_rows(handle)
    trades_count = meta.get("n_trades") or _count_trade_rows(handle)
    pipelines_raw = meta.get("pipelines", [])
    pipelines: List[PipelineMeta] = []
    for item in pipelines_raw:
        if not isinstance(item, dict):
            continue
        try:
            pipelines.append(PipelineMeta.model_validate(item))
        except Exception:
            continue
    return BacktestMeta(
        id=meta.get("id", handle.backtest_id),
        symbol=meta.get("symbol", meta.get("instrument", "UNKNOWN")),
        timeframe=meta.get("timeframe", meta.get("tf", "UNKNOWN")),
        start=meta.get("start"),
        end=meta.get("end"),
        n_frames=int(frames_count or 0),
        n_trades=int(trades_count or 0),
        params=meta.get("params", {}),
        kpis=meta.get("kpis", {}),
        pipelines=pipelines,
    )


def _load_meta_raw(handle: BacktestHandle) -> Dict[str, object]:
    if handle.meta_path and handle.meta_path.exists():
        with handle.meta_path.open("r", encoding="utf-8") as f:
            if handle.meta_path.suffix in {".yml", ".yaml"}:
                import yaml

                return yaml.safe_load(f) or {}
            return json.load(f)
    return {"id": handle.backtest_id}


def list_backtests(root: Path) -> List[BacktestSummary]:
    handles = discover_backtests(root)
    summaries: List[BacktestSummary] = []
    for handle in handles.values():
        meta = load_meta(handle)
        summaries.append(
            BacktestSummary(
                id=meta.id,
                symbol=meta.symbol,
                timeframe=meta.timeframe,
                start=meta.start or "",
                end=meta.end or "",
                n_frames=meta.n_frames,
                size_bytes=handle.size_bytes,
            )
        )
    summaries.sort(key=lambda item: item.start or item.id)
    return summaries


def stream_frames(handle: BacktestHandle, offset: int, limit: int) -> Iterator[Frame]:
    if handle.frames_path.suffix in {".jsonl", ".ndjson"}:
        yield from _stream_jsonl(handle.frames_path, offset, limit)
    elif handle.frames_path.suffix == ".csv":
        yield from _stream_csv(handle.frames_path, offset, limit)
    elif handle.frames_path.suffix == ".parquet":
        yield from _stream_parquet(handle.frames_path, offset, limit)
    else:
        raise ValueError(f"Unsupported frame format: {handle.frames_path.suffix}")


def load_trades(handle: BacktestHandle) -> List[Trade]:
    if not handle.trades_path or not handle.trades_path.exists():
        return []
    suffix = handle.trades_path.suffix
    if suffix in {".jsonl", ".ndjson"}:
        trades: List[Trade] = []
        with handle.trades_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trades.append(Trade.model_validate(json.loads(line)))
        return trades
    if suffix == ".csv":
        if pd is None:
            with handle.trades_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                return _trades_from_records(reader)
        df = pd.read_csv(handle.trades_path)
        return _trades_from_records(df.to_dict(orient="records"))
    if suffix == ".parquet":
        if pd is None:
            raise RuntimeError("Parquet support requires pandas. Install optional deps via `pip install -r visual/backend/requirements.txt`.")
        df = pd.read_parquet(handle.trades_path)
        return _trades_from_records(df.to_dict(orient="records"))
    raise ValueError(f"Unsupported trade format: {suffix}")


def _stream_jsonl(path: Path, offset: int, limit: int) -> Iterator[Frame]:
    with path.open("r", encoding="utf-8") as f:
        lines = islice((line for line in f if line.strip()), offset, offset + limit)
        for line in lines:
            payload = json.loads(line)
            yield Frame.model_validate(payload)


def _stream_csv(path: Path, offset: int, limit: int) -> Iterator[Frame]:
    if pd is None:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for idx, record in enumerate(reader):
                if idx < offset:
                    continue
                if idx >= offset + limit:
                    break
                yield Frame.model_validate(_normalize_flat_frame(record))
        return
    df = pd.read_csv(path, skiprows=range(1, offset + 1), nrows=limit)
    for record in df.to_dict(orient="records"):
        yield Frame.model_validate(_normalize_flat_frame(record))


def _stream_parquet(path: Path, offset: int, limit: int) -> Iterator[Frame]:
    if pd is None:
        raise RuntimeError("Parquet support requires pandas. Install optional deps via `pip install -r visual/backend/requirements.txt`.")
    df = pd.read_parquet(path)
    chunk = df.iloc[offset : offset + limit]
    for record in chunk.to_dict(orient="records"):
        yield Frame.model_validate(_normalize_flat_frame(record))


def _normalize_flat_frame(record: Dict[str, object]) -> Dict[str, object]:
    if "bar" in record:
        return record
    bar_keys = ["t", "o", "h", "l", "c", "v"]
    bar = {key: record.get(f"bar_{key}") or record.get(key) for key in bar_keys}
    return {
        "bar": bar,
        "signals": record.get("signals", []),
        "trades_open": record.get("trades_open", []),
        "trades_closed": record.get("trades_closed", []),
        "equity": record.get("equity"),
    }


def _trades_from_records(records: Iterable[Dict[str, object]]) -> List[Trade]:
    return [Trade.model_validate(record) for record in records]


def _count_frame_rows(handle: BacktestHandle) -> int:
    if handle.frames_path.suffix in {".jsonl", ".ndjson"}:
        with handle.frames_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if handle.frames_path.suffix == ".csv":
        with handle.frames_path.open("r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    if handle.frames_path.suffix == ".parquet":
        if pd is None:
            raise RuntimeError("Parquet support requires pandas. Install optional deps via `pip install -r visual/backend/requirements.txt`.")
        df = pd.read_parquet(handle.frames_path, columns=["bar_t"])
        return len(df)
    return 0


def _count_trade_rows(handle: BacktestHandle) -> int:
    if not handle.trades_path or not handle.trades_path.exists():
        return 0
    suffix = handle.trades_path.suffix
    if suffix in {".jsonl", ".ndjson"}:
        with handle.trades_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if suffix == ".csv":
        with handle.trades_path.open("r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    if suffix == ".parquet":
        if pd is None:
            raise RuntimeError("Parquet support requires pandas. Install optional deps via `pip install -r visual/backend/requirements.txt`.")
        df = pd.read_parquet(handle.trades_path, columns=["entry_t"])
        return len(df)
    return 0
