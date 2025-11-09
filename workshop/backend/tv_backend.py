#!/usr/bin/env python3
"""
TV Backend - FastAPI server para interface TradingView do BotScalp V3
Serve candles, sinais e métodos validados para o frontend app.js
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("/opt/botscalpv3/data_monthly")
FRONTEND_DIR = Path("/opt/botscalpv3/frontend")
API_PORT = int(os.getenv("API_PORT", "8080"))

app = FastAPI(
    title="BotScalp V3 - TV Backend",
    default_response_class=ORJSONResponse
)

# CORS para permitir requests do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve frontend estático
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# ============================================================
# MÉTODOS VALIDADOS (hardcoded dos resultados)
# ============================================================
VALIDATED_METHODS = {
    "5m": [
        {
            "id": "ema_crossover_5m",
            "method": "ema_crossover",
            "label": "EMA Crossover (5m)",
            "config": {
                "use_atr_stop": False,
                "use_atr_tp": False,
                "sl_fixed_perc": 0.003,
                "timeout_mode": "bars",
                "max_hold": 100,
                "intrabar_priority": "tp-first"
            },
            "metrics": {
                "pnl": 231408,
                "sharpe": 0.46,
                "hit_rate": 0.2744,
                "trades": 156,
                "win_rate": 0.75
            },
            "params": {
                "fast_ema": 9,
                "slow_ema": 21
            }
        }
    ],
    "15m": [
        {
            "id": "ema_crossover_15m",
            "method": "ema_crossover",
            "label": "EMA Crossover (15m) ⭐⭐ MELHOR",
            "config": {
                "use_atr_stop": False,
                "use_atr_tp": False,
                "sl_fixed_perc": 0.003,
                "timeout_mode": "bars",
                "max_hold": 50,
                "intrabar_priority": "tp-first"
            },
            "metrics": {
                "pnl": 297408,
                "sharpe": 0.52,
                "hit_rate": 0.2744,
                "trades": 78,
                "win_rate": 0.75
            },
            "params": {
                "fast_ema": 9,
                "slow_ema": 21
            }
        },
        {
            "id": "macd_trend_15m",
            "method": "macd_trend",
            "label": "MACD Trend (15m)",
            "config": {
                "use_atr_stop": False,
                "sl_fixed_perc": 0.003,
                "timeout_mode": "bars",
                "max_hold": 50,
                "intrabar_priority": "tp-first"
            },
            "metrics": {
                "pnl": 217408,
                "sharpe": 0.57,
                "hit_rate": 0.2811,
                "trades": 103,
                "win_rate": 0.75
            },
            "params": {
                "fast": 12,
                "slow": 26,
                "signal": 9
            }
        },
        {
            "id": "keltner_breakout_15m",
            "method": "keltner_breakout",
            "label": "Keltner Breakout (15m)",
            "config": {
                "use_atr_stop": True,
                "atr_stop_len": 14,
                "atr_stop_mult": 1.0,
                "timeout_mode": "bars",
                "max_hold": 50,
                "intrabar_priority": "tp-first"
            },
            "metrics": {
                "pnl": 57408,
                "sharpe": 0.09,
                "hit_rate": 0.2123,
                "trades": 65,
                "win_rate": 0.75
            },
            "params": {
                "length": 20,
                "mult": 2.0
            }
        }
    ],
    "1m": []
}

# ============================================================
# HELPERS: LOAD PARQUET DATA
# ============================================================
def load_candles_from_parquet(
    symbol: str,
    tf: str,
    limit: int = 1000,
    since_sec: Optional[int] = None
) -> List[dict]:
    """
    Carrega candles históricos dos arquivos parquet mensais.
    """
    # Mapeamento de timeframe para regra do pandas
    tf_map = {"1m": "1min", "5m": "5min", "15m": "15min"}
    rule = tf_map.get(tf, "5min")

    # Busca arquivos parquet do símbolo
    pattern = str(DATA_ROOT / f"*{symbol}*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        return []

    # Lê últimos 2-3 arquivos (últimos meses)
    recent_files = files[-3:] if len(files) >= 3 else files

    dfs = []
    for fpath in recent_files:
        try:
            df = pd.read_parquet(fpath)
            if "timestamp" in df.columns:
                df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            elif "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            else:
                continue
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Erro ao ler {fpath}: {e}")
            continue

    if not dfs:
        return []

    # Concatena e resample
    df = pd.concat(dfs, ignore_index=True)
    df = df.set_index("time").sort_index()

    # Garante colunas OHLCV
    if "open" not in df.columns:
        return []

    # Resample para o timeframe desejado
    ohlc = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    # Filtra desde timestamp se fornecido
    if since_sec:
        cutoff = pd.Timestamp.fromtimestamp(since_sec, tz="UTC")
        ohlc = ohlc[ohlc.index >= cutoff]

    # Limita quantidade
    ohlc = ohlc.tail(limit)

    # Converte para formato esperado pelo frontend
    candles = []
    for ts, row in ohlc.iterrows():
        candles.append({
            "time": int(ts.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"])
        })

    return candles

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/api/candles")
async def get_candles(
    symbol: str = "BTCUSDT",
    tf: str = "5m",
    limit: int = 1000,
    since: Optional[int] = None  # timestamp em ms
):
    """
    Retorna candles OHLCV para o símbolo e timeframe.

    Query params:
    - symbol: BTCUSDT, ETHUSDT, etc
    - tf: 1m, 5m, 15m
    - limit: quantidade máxima de candles
    - since: timestamp em ms (retorna apenas candles >= since)
    """
    since_sec = int(since / 1000) if since else None
    candles = load_candles_from_parquet(symbol, tf, limit, since_sec)
    return {"candles": candles}

@app.get("/api/fx")
async def get_fx(pair: str = "USDTBRL"):
    """
    Retorna taxa de câmbio USDT/BRL.
    Por enquanto hardcoded, pode integrar com API externa depois.
    """
    # TODO: Integrar com API real (Binance, AwesomeAPI, etc)
    return {"price": 5.67}

@app.get("/api/methods")
async def get_methods(
    type: str = "base",  # base | combo | ml
    tf: str = "5m",
    with_: str = Query("metrics,config,params", alias="with")
):
    """
    Retorna métodos/setups disponíveis para o timeframe.

    Query params:
    - type: base (métodos individuais), combo (combinações), ml (ML filter)
    - tf: 1m, 5m, 15m
    - with: campos extras (metrics, config, params)
    """
    if type == "base":
        methods = VALIDATED_METHODS.get(tf, [])
        return {"timeframes": {tf: methods}}
    elif type == "combo":
        # TODO: Implementar combos
        return {"timeframes": {tf: []}}
    elif type == "ml":
        # TODO: Implementar ML filtering
        return {"timeframes": {tf: []}}
    else:
        return {"timeframes": {tf: []}}

@app.get("/api/signals")
async def get_signals(
    symbol: str = "BTCUSDT",
    tf: str = "5m",
    type: str = "base",
    hours: int = 720,  # últimas 720h (30 dias)
    id: Optional[str] = None  # method_id
):
    """
    Retorna sinais do método selecionado.

    Query params:
    - symbol: BTCUSDT
    - tf: 1m, 5m, 15m
    - type: base, combo, ml
    - hours: janela de lookback em horas
    - id: ID do método (ex: ema_crossover_5m)

    Retorna:
    - signals: lista vazia (para compatibilidade)
    - executions: lista de sinais com entry, stop, target, exit

    Formato esperado pelo frontend:
    {
      "executions": [
        {
          "time": <timestamp ms>,
          "dir": 1 ou -1,
          "entry": <preço>,
          "stop": <preço>,
          "target": <preço>,
          "exit_time": <timestamp ms> (opcional),
          "exit_price": <preço> (opcional),
          "exit_reason": "target"|"stop"|"timeout"|"flip" (opcional),
          "pnl": <USDT> (opcional)
        }
      ]
    }
    """
    # TODO: Implementar carregamento de sinais reais
    # Por enquanto retorna vazio, frontend vai simular localmente
    #
    # Opções de implementação:
    # 1. Ler de CSV do selector21 (ex: leaderboard_base.csv)
    # 2. Rodar selector21 em background e cachear resultados
    # 3. Conectar com exchange e gerar sinais em tempo real

    # Placeholder: retorna sinais vazios
    return {
        "signals": [],
        "executions": []
    }

# ============================================================
# MAIN
# ============================================================
def run():
    """Inicia o servidor FastAPI"""
    print(f"🚀 TV Backend iniciando na porta {API_PORT}")
    print(f"📁 Data root: {DATA_ROOT}")
    print(f"🌐 Frontend: http://localhost:{API_PORT}")
    print(f"📊 API: http://localhost:{API_PORT}/api")

    uvicorn.run(
        "tv_backend:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )

if __name__ == "__main__":
    run()
