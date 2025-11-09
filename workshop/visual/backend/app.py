from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from .loaders import (
        BacktestHandle,
        discover_backtests,
        list_backtests,
        load_meta,
        load_trades,
        stream_frames,
    )
    from .metrics import KPIAccumulator
    from .models import BacktestMeta, BacktestSummary, Frame, FramePage, TradeTable
except ImportError:  # allow running as a script from this folder
    from loaders import (
        BacktestHandle,
        discover_backtests,
        list_backtests,
        load_meta,
        load_trades,
        stream_frames,
    )
    from metrics import KPIAccumulator
    from models import BacktestMeta, BacktestSummary, Frame, FramePage, TradeTable

load_dotenv()

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "dist"
DATA_ROOT = Path(os.getenv("VISUAL_DATA_ROOT", DEFAULT_ROOT)).resolve()
API_PORT = int(os.getenv("API_PORT", "8888"))
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "")
USE_WS = os.getenv("USE_WS", "false").lower() == "true"
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", DEFAULT_FRONTEND)).resolve()

app = FastAPI(title="botscalpv3-visual", default_response_class=ORJSONResponse)

origins: List[str] = []
if CORS_ORIGIN:
    origins = [item.strip() for item in CORS_ORIGIN.split(",") if item.strip()]

if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if FRONTEND_DIST.exists():
    # Mount frontend under /app to avoid shadowing API routes under /
    app.mount("/app", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


class BacktestRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._handles: Dict[str, BacktestHandle] = {}
        self.refresh()

    def refresh(self) -> None:
        self._handles = discover_backtests(self.root)

    def get_handle(self, backtest_id: str) -> BacktestHandle:
        handle = self._handles.get(backtest_id)
        if handle:
            return handle
        self.refresh()
        handle = self._handles.get(backtest_id)
        if not handle:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return handle

    def summaries(self) -> List[BacktestSummary]:
        return list_backtests(self.root)


repository = BacktestRepository(DATA_ROOT)


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/")


@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/backtests", response_model=List[BacktestSummary])
def get_backtests() -> List[BacktestSummary]:
    summaries = repository.summaries()
    if len(summaries) < 2:
        # Allow running with fewer, but highlight expectation.
        return summaries
    return summaries


@app.get("/api/backtests/{backtest_id}/meta", response_model=BacktestMeta)
def get_meta(backtest_id: str) -> BacktestMeta:
    handle = repository.get_handle(backtest_id)
    return load_meta(handle)


@app.get("/api/backtests/{backtest_id}/frames", response_model=FramePage)
def get_frames(backtest_id: str, offset: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=5000)) -> FramePage:
    handle = repository.get_handle(backtest_id)
    meta = load_meta(handle)
    frames_iter = stream_frames(handle, offset, limit)
    frames = list(frames_iter)
    accumulator = KPIAccumulator()
    accumulator.ingest_batch(frames)
    return FramePage(
        frames=frames,
        offset=offset,
        limit=limit,
        total=meta.n_frames,
        kpis=accumulator.snapshot(),
    )


@app.get("/api/backtests/{backtest_id}/trades", response_model=TradeTable)
def get_trades(backtest_id: str) -> TradeTable:
    handle = repository.get_handle(backtest_id)
    trades = load_trades(handle)
    return TradeTable(backtest_id=backtest_id, trades=trades)


if USE_WS:

    @app.websocket("/ws/replay/{backtest_id}")
    async def replay_stream(websocket: WebSocket, backtest_id: str) -> None:
        await websocket.accept()
        try:
            handle = repository.get_handle(backtest_id)
            chunk = 250
            offset = 0
            while True:
                frames = list(stream_frames(handle, offset, chunk))
                if not frames:
                    await websocket.send_json({"event": "end"})
                    break
                payload = {
                    "offset": offset,
                    "frames": [frame.model_dump() for frame in frames],
                }
                await websocket.send_json(payload)
                offset += len(frames)
        except HTTPException:
            await websocket.send_json({"event": "error", "message": "backtest not found"})
        except WebSocketDisconnect:
            return
        finally:
            await websocket.close()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    run()
