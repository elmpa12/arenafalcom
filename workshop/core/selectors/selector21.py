#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selectorv2ml.py — v2.1 (2025-10-11)

Uma versão reescrita e endurecida do selectorbeta_ml.py, focada em:
- ✅ Bugfixes críticos: chamadas ao backtest, otimização de threshold no ML, salvamento opcional do modelo.
- ✅ Execução mais realista: stop/TP em USD calculados por PnL líquido (com fees), intrabar STOP antes de TP,
     slippage na entrada/saída, flip correto e arredondamento ao tick.
- ✅ WFO robusto (mensal) para métodos base e combos, com leaderboard e consolidação de parâmetros.
- ✅ Pipeline de ML (XGB→RF→LogReg→NP-LogReg) com scaler, features ricas e threshold ótimo por janela.
- ✅ Gating por CVD/Depth com tolerância a ausência de features e métricas com fees/notional.

Uso básico (exemplos):
  python selectorv2ml.py --symbol BTCUSDT --data_dir ./data --start 2023-01-01 --end 2025-09-30 \
      --run_base --run_combos --exec_rules "1m,5m,15m" --walkforward \
      --wf_train_months 3 --wf_val_months 1 --wf_step_months 1 \
      --use_atr_stop --atr_stop_len 14 --atr_stop_mult 2.0 \
      --use_atr_tp --atr_tp_len "14,14,14" --atr_tp_mult "2.5,2.5,3.0" \
      --timeout_mode both --atr_timeout_len "14,14,14" --atr_timeout_mult "8,6,4" \
      --round_to_tick --tick_size 0.10 --contracts 1 --contract_value 100 \
      --hard_stop_usd "60,80,100" --hard_tp_usd "300,360,400" \
      --agg_dir ./agg --depth_dir ./depth --depth_field bd_imb_50bps \
      --run_ml --ml_model_kind auto --ml_use_agg --ml_use_depth --ml_add_base_feats --ml_opt_thr \
      --print_top10

Requisitos opcionais (auto‑fallback se ausentes):
  - xgboost, scikit‑learn, joblib

Nada aqui é recomendação financeira. Este script foca engenharia de backtest e execução.
"""
# === v2.1 CHANGELOG (2025-10-11) ===
# - Corrigido: fallback de hard_stop_usd agora usa hard_stop_usd_default (antes usava hard_tp_usd_default).
# - ML: respeita --ml_opt_thr (threshold default=0.5 quando desligado).
# - ML: adicionada opção --ml_neutral_band para zona neutra (substitui hardcode ±0.10).
# - ML: filtro de regime 'vol_bin' agora é aplicado com base em df_tr/df_va (não em X_tr_df).
# - Features: _make_ml_features_v2 aceita 'mid/spread' como fallback para 'bbo_mid/bbo_spread'.
# - main(): import de psutil agora é opcional (fallback seguro se ausente).


from __future__ import annotations
import ast

# ========== Limitar threads internas das libs numéricas (antes de importar numpy/pandas) ==========
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

# ========== Imports ==========
import os
import sys
import re
import glob
import json
import math
import argparse
GLOBAL_DATA_MAP = {}
BASE_KLINES_DIR = None
from dataclasses import dataclass
from itertools import combinations, product
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.dl.heads import available_head_names
from .selector21_config import GateConfig, gate_config_from_args
from .selector21_core import (
    apply_signal_gates,
    combine_probs,
    gate_signal_atr_zscore,
    gate_signal_cvd,
    gate_signal_imbalance,
    gate_signal_vhf,
    half_life_weights,
)
from .selector21_run import iter_task_results
from core.selectors.pipeline import (
    DeepLearningDataset,
    FeatureRequest,
    FeatureSnapshot,
    SignalSet,
    TimeSpan,
)
from core.selectors.data_pipeline import compute_feature_presence, enrich_with_all_features

# ------------------ Globais ------------------
TF_FRAMES: Dict[str, pd.DataFrame] = {}        # df enriquecido por timeframe (com AGG/DEPTH)
FEATURES_META: Dict[str, Dict[str, float|bool]] = {}  # presença/nnz das features por TF


class SelectorSession:
    """Coordena o fluxo Selector → Deep Learning de forma explícita."""

    def __init__(self, gate_config: Optional[GateConfig] = None) -> None:
        self.gate_config = gate_config
        self._snapshots: Dict[str, FeatureSnapshot] = {}
        self._signals: Dict[str, SignalSet] = {}

    def load_features(self, request: FeatureRequest, window: TimeSpan) -> FeatureSnapshot:
        """Carrega e armazena um timeframe enriquecido no cache local."""

        start_ms, end_ms = window.as_millis()
        frame = enrich_with_all_features(
            request.symbol,
            request.timeframe,
            start_ms,
            end_ms,
            request.data_root,
        )
        if frame is None:
            frame = pd.DataFrame()

        frame = frame.sort_values("time").reset_index(drop=True)
        TF_FRAMES[request.timeframe] = frame

        snapshot = FeatureSnapshot(
            timeframe=request.timeframe,
            frame=frame,
            metadata=dict(FEATURES_META.get(request.timeframe, {})),
        )
        self._snapshots[request.timeframe] = snapshot
        return snapshot

    def get_snapshot(self, timeframe: str) -> FeatureSnapshot:
        try:
            return self._snapshots[timeframe]
        except KeyError as exc:  # pragma: no cover - defesa contra mau uso
            raise KeyError(f"Snapshot não encontrado para timeframe '{timeframe}'") from exc

    def register_signals(
        self,
        timeframe: str,
        *,
        base: Optional[pd.DataFrame] = None,
        combos: Optional[pd.DataFrame] = None,
    ) -> None:
        """Armazena sinais para reutilização posterior (DL/UI)."""

        self._signals[timeframe] = SignalSet(
            timeframe=timeframe,
            base=base.copy(deep=True) if base is not None else pd.DataFrame(),
            combos=combos.copy(deep=True) if combos is not None else pd.DataFrame(),
        )

    def signals_for(self, timeframe: str) -> SignalSet:
        return self._signals.get(timeframe, SignalSet(timeframe=timeframe))

    def build_deep_learning_dataset(
        self,
        timeframe: str,
        *,
        horizon: int,
        lags: int,
        train_split: float = 0.8,
        include_agg: bool = True,
        include_depth: bool = True,
        add_signal_features: bool = True,
        weight_half_life: Optional[int] = None,
    ) -> DeepLearningDataset:
        """Monta um payload estruturado para o ``dl_head``."""

        snapshot = self.get_snapshot(timeframe)
        frame = snapshot.frame.copy()
        if frame.empty:
            raise ValueError(
                f"Snapshot vazio para '{timeframe}'. Carregue dados com load_features() antes de montar o dataset."
            )

        signals = self.signals_for(timeframe) if add_signal_features else SignalSet(timeframe=timeframe)
        base_df = signals.base if not signals.base.empty else None
        combo_df = signals.combos if not signals.combos.empty else None

        features = _make_ml_features_v2(
            frame,
            add_lags=lags,
            include_agg=include_agg,
            include_depth=include_depth,
            base_signals_df=base_df,
            combo_signals_df=combo_df,
        )

        selected_index = features.index
        features = features.reset_index(drop=True)

        close = pd.to_numeric(frame.loc[selected_index, "close"], errors="coerce").reset_index(drop=True)
        times = pd.to_datetime(frame.loc[selected_index, "time"], utc=True).reset_index(drop=True)
        atr_series = atr(frame).reindex(selected_index).reset_index(drop=True)
        labels = _make_ml_labels(close, atr_series=atr_series, horizon=horizon).reset_index(drop=True)
        future_ret = (close.shift(-horizon) / close - 1.0).reset_index(drop=True)

        mask = labels.notna() & future_ret.notna()
        dataset_frame = features.loc[mask].reset_index(drop=True)
        dataset_frame["y"] = labels.loc[mask].astype(np.float32).reset_index(drop=True)
        dataset_frame["future_ret"] = future_ret.loc[mask].astype(np.float32).reset_index(drop=True)
        dataset_frame["time"] = times.loc[mask].reset_index(drop=True)

        weight_column = None
        if weight_half_life:
            weights = half_life_weights(len(dataset_frame), weight_half_life).astype(np.float32)
            dataset_frame["sample_weight"] = weights
            weight_column = "sample_weight"

        split_idx = int(max(1, min(len(dataset_frame) - 1, round(len(dataset_frame) * train_split))))
        train_df = dataset_frame.iloc[:split_idx].reset_index(drop=True)
        val_df = dataset_frame.iloc[split_idx:].reset_index(drop=True)

        metadata = dict(snapshot.metadata)
        metadata.update(
            {
                "rows_total": int(len(dataset_frame)),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "horizon": int(horizon),
                "lags": int(lags),
                "train_split": float(train_split),
                "available_heads": available_head_names(),
            }
        )

        return DeepLearningDataset(
            timeframe=timeframe,
            train=train_df,
            validation=val_df,
            horizon=horizon,
            lags=lags,
            label_column="y",
            return_column="future_ret",
            weight_column=weight_column,
            metadata=metadata,
        )
def enrich_with_all_features(
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
    root_dir: str,
) -> pd.DataFrame:
    """
    Enriquecimento completo para backtest:
      - OHLCV base reamostrado
      - AggTrades: cvd_slope_agg, flow_buy, flow_sell
      - BookDepth: imb_net_depth (+ alias imb_ratio_depth)
      - BookTicker: mid, spread, microprice_imb
      - FundingRate, PremiumIndex, MarkPrice
      - Normalizações, lags, dedupe e retorno compacto
    """
    import os, glob, zipfile
    import numpy as np
    import pandas as pd

    # --- helpers locais (para evitar colisão com globais) ---
    def _tf_to_rule(tf_: str) -> str:
        """Converte 5m → 5min, 1h → 1H etc."""
        tf_ = str(tf_).lower().strip()
        if tf_.endswith("m"):
            return f"{int(tf_[:-1])}min"
        if tf_.endswith("h"):
            return f"{int(tf_[:-1])}H"
        if tf_.endswith("d"):
            return f"{int(tf_[:-1])}D"
        return "5min"

    def _to_utc(s, unit=None):
        try:
            return pd.to_datetime(s, utc=True, unit=unit, errors="coerce")
        except Exception:
            return pd.to_datetime(s, utc=True, errors="coerce")

    def _clip_std(s: pd.Series, k: float = 8.0) -> pd.Series:
        m = s.mean(); sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd): return s
        lo, hi = m - k*sd, m + k*sd
        return s.clip(lower=lo, upper=hi)

    def _downcast_inplace_local(df: pd.DataFrame) -> pd.DataFrame:
        for c in df.select_dtypes(include=[np.floating]).columns:
            df[c] = pd.to_numeric(df[c], downcast="float")
        for c in df.select_dtypes(include=[np.integer]).columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        return df

    def _ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()
    def _rsi(s, n=14):
        d = s.diff(); up = d.clip(lower=0); down = -d.clip(upper=0)
        rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
        return 100 - (100 / (1 + rs))
    def _boll_pos(s, n=20, k=2): 
        m = s.rolling(n).mean(); sd = s.rolling(n).std(ddof=0)
        return (s - m) / (k*sd + 1e-9)
    def _atr(df_, n=14):
        tr = pd.concat([
            df_["high"] - df_["low"],
            (df_["high"] - df_["close"].shift()).abs(),
            (df_["low"]  - df_["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    # ---------- setup ----------
    rule = _tf_to_rule(tf)
    sym  = symbol.upper().strip()
    t0   = _to_utc(start_ms, unit="ms")
    t1   = _to_utc(end_ms,   unit="ms")
    ym   = t0.strftime("%Y-%m")

    # =============== OHLCV base (usa GLOBAL_DATA_MAP se existir) ===============
    _gdm = globals().get("GLOBAL_DATA_MAP", {})
    base_df = _gdm.get(tf) if isinstance(_gdm, dict) else None
    dfs = []

    if base_df is not None and hasattr(base_df, "empty") and not base_df.empty:
        d = base_df.copy()
        if "time" not in d.columns and "open_time" in d.columns:
            d["time"] = d["open_time"]
        d["time"] = _to_utc(pd.to_numeric(d["time"], errors="coerce"), unit="ms")
        for c in ("open","high","low","close","volume"):
            if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
        dfs.append(d[[c for c in ("time","open","high","low","close","volume") if c in d.columns]])
    else:
        kl_dir_candidates = []
        _args = globals().get("args", None)
        if _args is not None and getattr(_args, "data_dir", ""):
            kl_dir_candidates.append(os.path.join(_args.data_dir, "klines", tf))
        kl_dir_candidates.append(os.path.join(root_dir, "klines", tf))
        kl_dir_candidates.append(os.path.join("/opt/botscalp/datafull/klines", tf))

        kl_files = []
        for ddir in kl_dir_candidates:
            if os.path.isdir(ddir):
                found = sorted(glob.glob(os.path.join(ddir, f"{sym}-{tf}-*.parquet")))
                if found:
                    print(f"[ENRICH][INFO] usando klines em: {ddir} ({len(found)} arquivos)")
                    kl_files = found
                    break
        if not kl_files:
            raise FileNotFoundError(f"[ENRICH] Nenhum kline encontrado em {kl_dir_candidates}")

        for f in kl_files:
            d = pd.read_parquet(f)
            if d.empty: 
                continue
            d.columns = [str(c).strip() for c in d.columns]
            if all(c.isdigit() for c in d.columns):
                cols = ['time','open','high','low','close','volume',
                        'close_time','quote_vol','n_trades',
                        'taker_buy_vol','taker_buy_quote_vol','ignore']
                d.columns = cols[:len(d.columns)]
            if "time" not in d.columns:
                if "open_time" in d.columns: d["time"] = pd.to_numeric(d["open_time"], errors="coerce")
                else: raise ValueError("[ENRICH] Klines sem coluna de tempo reconhecível")
            d["time"] = _to_utc(pd.to_numeric(d["time"], errors="coerce"), unit="ms")
            for c in ("open","high","low","close","volume"):
                if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
            dfs.append(d[[c for c in ("time","open","high","low","close","volume") if c in d.columns]])

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    df = df[(df["time"] >= t0) & (df["time"] <= t1)]

    g = df.set_index("time").resample(rule, label="right", closed="right")
    df = pd.DataFrame({
        "time":   g["close"].last().index,
        "open":   g["open"].first(),
        "high":   g["high"].max(),
        "low":    g["low"].min(),
        "close":  g["close"].last(),
        "volume": g["volume"].sum(),
    }).dropna().reset_index(drop=True)


    # === PATCH: Feature de regime de volatilidade ===
    import numpy as np
    import pandas as pd

    try:
        import talib as ta
    except ImportError:
        print("[WARN] TA-Lib não disponível, ATR_zscore será calculado manualmente.")
        ta = None

    if all(col in df.columns for col in ["high", "low", "close"]):
        if ta:
            df["atr14"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)
            df["atr100"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=100)
        else:
            hl = df["high"] - df["low"]
            hc = (df["high"] - df["close"].shift()).abs()
            lc = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df["atr14"] = tr.rolling(14, min_periods=1).mean()
            df["atr100"] = tr.rolling(100, min_periods=1).mean()

        df["atr_zscore"] = df["atr14"] / (df["atr100"] + 1e-9)
        df["atr_zscore"] = df["atr_zscore"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        df["vol_bin"] = (df["atr_zscore"] > 1.3).astype(np.int8)

        mean_z = df["atr_zscore"].mean()
        high_vol_frac = (df["vol_bin"].sum() / len(df)) * 100.0
        print(f"[ENRICH][ATR_Z] Média z={mean_z:.2f} | Vol. alta em {high_vol_frac:.1f}% das barras.")
    else:
        print("[ENRICH][ATR_Z] Colunas high/low/close ausentes — ignorando ATR regime.")


    # =============== Indicadores base ===============
    close = df["close"].astype(float)
    df["ema_fast"] = _ema(close, 12)
    df["ema_slow"] = _ema(close, 26)
    df["macd"]     = df["ema_fast"] - df["ema_slow"]
    df["rsi14"]    = _rsi(close, 14)
    df["boll_pos"] = _boll_pos(close, 20, 2.0)
    df["atr_pct"]  = _atr(df, 14) / (close.abs() + 1e-9)
    df["vwap20"]   = ((df["close"] * df["volume"]).rolling(20).sum() /
                      (df["volume"].rolling(20).sum() + 1e-9))

    # =============== AggTrades → cvd_slope_agg (multi-formato) ===============
    agg_dir_candidates = []
    _args = globals().get("args", None)
    if _args is not None and getattr(_args, "agg_dir", ""):
        agg_dir_candidates.append(_args.agg_dir)
    agg_dir_candidates.append(os.path.join(root_dir, "aggTrades"))
    agg_dir_candidates.append("/opt/botscalp/datafull/aggTrades")
    agg_dir_use = next((d for d in agg_dir_candidates if d and os.path.isdir(d)), None)

    if not agg_dir_use:
        print("[ENRICH][WARN] Nenhum diretório válido para aggTrades entre:", agg_dir_candidates)
        df["cvd_slope_agg"] = 0.0
    else:
        print(f"[ENRICH][INFO] Usando aggTrades em: {agg_dir_use}")
        agg_files = sorted([
            os.path.join(agg_dir_use, f) for f in os.listdir(agg_dir_use)
            if f.startswith(f"{sym}-aggTrades-") and f.lower().endswith((".parquet", ".csv", ".zip"))
        ])
        dfa = []
        for p in agg_files:
            try:
                if p.lower().endswith(".parquet"):
                    a = pd.read_parquet(p)
                elif p.lower().endswith(".zip"):
                    with zipfile.ZipFile(p) as z:
                        inner = z.namelist()[0]
                        with z.open(inner) as fh:
                            a = pd.read_csv(fh, header=None)
                else:
                    a = pd.read_csv(p, header=None)

                a = a.loc[:, ~a.columns.duplicated()]
                if all(str(c).isdigit() for c in a.columns) and a.shape[1] >= 7:
                    a = a.iloc[:, :7]
                    a.columns = ["trade_id","price","qty","first_id","last_id","time","is_buyer_maker"]

                if "time" not in a.columns:
                    cand = [c for c in a.columns if "time" in str(c).lower() or str(c).lower() in ("timestamp","open_time")]
                    if cand: a = a.rename(columns={cand[0]:"time"})
                    else:    continue

                a["time"] = pd.to_numeric(a["time"], errors="coerce")
                if a["time"].max() < 1e12: a["time"] *= 1000
                a["time"] = _to_utc(a["time"].astype("int64"), unit="ms")

                a = a[(a["time"] >= t0) & (a["time"] <= t1)]
                if a.empty: 
                    continue

                a["price"] = pd.to_numeric(a.get("price"), errors="coerce")
                a["qty"]   = pd.to_numeric(a.get("qty"),   errors="coerce").fillna(0.0)
                a["is_buyer_maker"] = a.get("is_buyer_maker", False)
                a["is_buyer_maker"] = a["is_buyer_maker"].astype(str).str.lower().isin(["true","1","t","yes"])

                idx = a.set_index("time").sort_index()
                idx["_notional"]        = idx["price"] * idx["qty"]
                idx["_signed_notional"] = idx["_notional"] * np.where(idx["is_buyer_maker"], -1.0, +1.0)

                net  = idx["_signed_notional"].resample(rule, label="right", closed="right").sum()
                buy  = idx.loc[~idx["is_buyer_maker"], "_notional"].resample(rule, label="right", closed="right").sum()
                sell = idx.loc[ idx["is_buyer_maker"], "_notional"].resample(rule, label="right", closed="right").sum()

                out = pd.DataFrame({
                    "time":          net.index,
                    "cvd_slope_agg": net.values,
                    "flow_buy":      buy.reindex(net.index).values,
                    "flow_sell":     sell.reindex(net.index).values,
                }).reset_index(drop=True)
                dfa.append(out)
            except Exception as e:
                print(f"[ENRICH][SKIP AGG] {os.path.basename(p)} -> {e}")

        if dfa:
            agg_feat = pd.concat(dfa, ignore_index=True).dropna(how="all")
            df = df.merge(agg_feat, on="time", how="left")

    # compat: coluna acumulada 'cvd'
    if "cvd_slope_agg" not in df.columns:
        df["cvd_slope_agg"] = 0.0
    # correção robusta para cvd_slope_agg que pode vir como float
    cvd_src = df.get("cvd_slope_agg", 0.0)
    if not isinstance(cvd_src, pd.Series):
        cvd_src = pd.Series(cvd_src, index=df.index)
    df["cvd"] = cvd_src.fillna(0.0).cumsum()

    # =============== BookDepth → imb_net_depth (mensal/diário, multi-formato) ===============
    depth_dir_candidates = []
    if _args is not None and getattr(_args, "depth_dir", ""):
        depth_dir_candidates.append(_args.depth_dir)
    depth_dir_candidates.append(os.path.join(root_dir, "bookDepth"))
    depth_dir_candidates.append("/opt/botscalp/datafull/bookDepth")
    depth_dir_use = next((d for d in depth_dir_candidates if d and os.path.isdir(d)), None)

    if not depth_dir_use:
        print("[ENRICH][WARN] Nenhum diretório válido para bookDepth entre:", depth_dir_candidates)
        df["imb_net_depth"] = 0.0
    else:
        depth_files = sorted([
            os.path.join(depth_dir_use, f) for f in os.listdir(depth_dir_use)
            if f.startswith(f"{sym}-bookDepth-") and f.lower().endswith((".parquet",".csv",".zip"))
        ])
        dd_list=[]
        for p in depth_files:
            try:
                if p.lower().endswith(".parquet"):
                    dd = pd.read_parquet(p)
                elif p.lower().endswith(".zip"):
                    with zipfile.ZipFile(p) as z:
                        inner = z.namelist()[0]
                        with z.open(inner) as fh:
                            dd = pd.read_csv(fh)
                else:
                    dd = pd.read_csv(p)

                dd = dd.loc[:, ~dd.columns.duplicated()]

                if {"timestamp","percentage","depth"} <= set(dd.columns):
                    ts = pd.to_numeric(dd["timestamp"], errors="coerce")
                    if ts.max() < 1e12: ts = ts * 1000
                    dd["_time"] = _to_utc(ts.astype("int64"), unit="ms")
                    dd = dd[(dd["_time"] >= t0) & (dd["_time"] <= t1)]
                    if dd.empty: 
                        continue

                    mask_pos = dd["percentage"] > 0
                    mask_neg = dd["percentage"] < 0
                    pos = pd.to_numeric(dd.loc[mask_pos, "depth"], errors="coerce").groupby(dd.loc[mask_pos, "_time"]).sum()
                    neg = pd.to_numeric(dd.loc[mask_neg, "depth"], errors="coerce").groupby(dd.loc[mask_neg, "_time"]).sum()
                    ix = pos.index.union(neg.index).sort_values()
                    pos = pos.reindex(ix, fill_value=0.0)
                    neg = neg.reindex(ix, fill_value=0.0)
                    tmp = pd.DataFrame({"time": ix, "imb_net_depth": (pos - neg) / (pos + neg + 1e-9)})

                else:
                    col = "imb_net_depth"
                    if col not in dd.columns:
                        cand = [c for c in dd.columns if str(c).startswith("bd_imb_")]
                        if cand: col = cand[0]

                    if "time" not in dd.columns:
                        candt = [c for c in dd.columns if "time" in c.lower() or c.lower()=="timestamp"]
                        if candt: dd = dd.rename(columns={candt[0]:"time"})
                    ts = pd.to_numeric(dd.get("time"), errors="coerce")
                    if ts.notna().any():
                        if ts.max() < 1e12: ts = ts * 1000
                        dd["_time"] = _to_utc(ts.astype("int64"), unit="ms")
                        dd = dd[(dd["_time"] >= t0) & (dd["_time"] <= t1)]
                        tmp = pd.DataFrame({"time": dd["_time"], "imb_net_depth": pd.to_numeric(dd[col], errors="coerce")})
                    else:
                        tmp = pd.DataFrame(columns=["time","imb_net_depth"])

                if not tmp.empty:
                    gdep = tmp.set_index("time").resample(rule, label="right", closed="right")["imb_net_depth"].mean()
                    dd_list.append(gdep.reset_index())
            except Exception as e:
                print(f"[ENRICH][SKIP DEPTH] {os.path.basename(p)} -> {e}")

        if dd_list:
            depth_feat = pd.concat(dd_list, ignore_index=True)
            df = df.merge(depth_feat, on="time", how="left")

    if "imb_net_depth" not in df.columns:
        df["imb_net_depth"] = 0.0
    df["imb_ratio_depth"] = df["imb_net_depth"]

    # =============== BookTicker (BBO) ===============
    bbo_dir = os.path.join(root_dir, "bookTicker")
    bbo_paths = sorted(glob.glob(os.path.join(bbo_dir, f"{sym}*.parquet")))
    if bbo_paths:
        bbos=[]
        for p in bbo_paths:
            try:
                b = pd.read_parquet(p)
                if b.empty: continue
                for c in ("bestBidPrice","bestAskPrice","bestBidQty","bestAskQty","time"):
                    if c in b.columns: b[c] = pd.to_numeric(b[c], errors="coerce")
                b["time"] = _to_utc(b["time"], unit="ms")
                b = b[(b["time"] >= t0) & (b["time"] <= t1)]
                b = b.set_index("time").resample(rule, label="right", closed="right").last()
                b["mid"]    = (b.get("bestBidPrice") + b.get("bestAskPrice")) / 2.0
                b["spread"] = (b.get("bestAskPrice") - b.get("bestBidPrice")).abs()
                if {"bestAskQty","bestBidQty"}.issubset(b.columns):
                    b["microprice_imb"] = (b["bestAskQty"] - b["bestBidQty"]) / (b["bestAskQty"] + b["bestBidQty"] + 1e-9)
                else:
                    b["microprice_imb"] = np.nan
                bbos.append(b[["mid","spread","microprice_imb"]])
            except Exception as e:
                print("[ENRICH][SKIP BBO]", os.path.basename(p), "->", e)
        if bbos:
            bbo = pd.concat(bbos).sort_index()
            df = df.merge(bbo.reset_index(), on="time", how="left")

    # =============== FundingRate / PremiumIndex / MarkPrice ===============
    fr_dir   = os.path.join(root_dir, "fundingRate")
    prem_dir = os.path.join(root_dir, "premiumIndexKlines")
    mark_dir = os.path.join(root_dir, "markPriceKlines")

    fr_paths   = sorted(glob.glob(os.path.join(fr_dir,   f"{sym}*.parquet"))) if os.path.isdir(fr_dir)   else []
    prem_paths = sorted(glob.glob(os.path.join(prem_dir, f"{sym}*.parquet"))) if os.path.isdir(prem_dir) else []
    mark_paths = sorted(glob.glob(os.path.join(mark_dir, f"{sym}*.parquet"))) if os.path.isdir(mark_dir) else []

    if fr_paths:
        funds=[]
        for p in fr_paths:
            try:
                f = pd.read_parquet(p)
                if len(f): funds.append(f)
            except Exception as e:
                print("[ENRICH][SKIP FR]", os.path.basename(p), "->", e)
        if funds:
            fr = pd.concat(funds, ignore_index=True)
            fr["time"] = _to_utc(pd.to_numeric(fr["time"], errors="coerce"), unit="ms")
            fr = fr[(fr["time"] >= t0) & (fr["time"] <= t1)]
            fr = fr.set_index("time").resample(rule, label="right", closed="right").last()
            if "fundingRate" in fr.columns:
                fr["funding_chg"] = fr["fundingRate"].diff().fillna(0)
                fr["funding_z"]   = (fr["fundingRate"] - fr["fundingRate"].rolling(96).mean()) / (fr["fundingRate"].rolling(96).std() + 1e-9)
                df = df.merge(fr[["fundingRate","funding_chg","funding_z"]].reset_index(), on="time", how="left")

    if prem_paths:
        prems=[]
        for p in prem_paths:
            try:
                pr = pd.read_parquet(p); 
                if len(pr): prems.append(pr)
            except Exception as e:
                print("[ENRICH][SKIP PREM]", os.path.basename(p), "->", e)
        if prems:
            pr = pd.concat(prems, ignore_index=True)
            pr["time"] = _to_utc(pd.to_numeric(pr["time"], errors="coerce"), unit="ms")
            pr = pr[(pr["time"] >= t0) & (pr["time"] <= t1)]
            pr = pr.set_index("time").resample(rule, label="right", closed="right").last()
            if "premiumIndex" in pr.columns:
                pr["premium_z"] = (pr["premiumIndex"] - pr["premiumIndex"].rolling(96).mean()) / (pr["premiumIndex"].rolling(96).std() + 1e-9)
                df = df.merge(pr[["premiumIndex","premium_z"]].reset_index(), on="time", how="left")

    if mark_paths:
        marks=[]
        for p in mark_paths:
            try:
                mk = pd.read_parquet(p)
                if len(mk): marks.append(mk)
            except Exception as e:
                print("[ENRICH][SKIP MARK]", os.path.basename(p), "->", e)
        if marks:
            mk = pd.concat(marks, ignore_index=True)
            mk["time"] = _to_utc(pd.to_numeric(mk["time"], errors="coerce"), unit="ms")
            mk = mk[(mk["time"] >= t0) & (mk["time"] <= t1)]
            mk = mk.set_index("time").resample(rule, label="right", closed="right").last()
            if "markPrice" in mk.columns:
                tmp = mk[["markPrice"]].reset_index()
                df = df.merge(tmp, on="time", how="left")
                df["mark_close_ratio"] = (df["markPrice"] / df["close"].replace(0, np.nan)) - 1.0
    # =============== Metrics (parquet/csv/zip) ===============
    metrics_dir = os.path.join(root_dir, "metrics")
    if os.path.isdir(metrics_dir):
        import re, io
        def _pick_time_col(df):
            cands = ["create_time","time","timestamp","datetime","time_ms","timestamp_ms"]
            cols = [str(c) for c in df.columns]
            low  = {re.sub(r'[^a-z0-9]','', c.lower()): c for c in cols}
            for cand in cands:
                nc = re.sub(r'[^a-z0-9]','', cand.lower())
                if nc in low: return low[nc]
            for cand in cands:
                nc = re.sub(r'[^a-z0-9]','', cand.lower())
                for k,orig in low.items():
                    if k.startswith(nc) or nc in k: return orig
            return None

        def _read_zip_any_csv(path, nrows=None):
            try:
                with zipfile.ZipFile(path) as z:
                    names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
                    name  = names[0] if names else next((n for n in z.namelist() if not n.endswith("/")), None)
                    if not name: return pd.DataFrame()
                    with z.open(name) as fh: raw = fh.read()
                for kw in (dict(nrows=nrows), dict(sep=None, engine="python", nrows=nrows),
                           dict(sep=";", engine="python", nrows=nrows), dict(sep="|", engine="python", nrows=nrows),
                           dict(sep=None, engine="python", encoding="utf-8-sig", nrows=nrows),
                           dict(sep=None, engine="python", encoding="latin1", nrows=nrows)):
                    bio = io.BytesIO(raw)
                    try:
                        t = pd.read_csv(bio, **kw)
                        if not t.empty: return t
                    except: pass
            except: pass
            return pd.DataFrame()

        files = sorted([os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir)
                        if f.startswith(f"{sym}-metrics-") and f.lower().endswith((".parquet",".csv",".zip"))])

        mx_list = []
        for p in files:
            try:
                if p.lower().endswith(".parquet"):
                    t = pd.read_parquet(p)
                elif p.lower().endswith(".zip"):
                    t = _read_zip_any_csv(p)
                else:
                    # csv direto
                    for kw in (dict(), dict(sep=None, engine="python"),
                               dict(sep=";", engine="python"), dict(sep="|", engine="python"),
                               dict(sep=None, engine="python", encoding="utf-8-sig"),
                               dict(sep=None, engine="python", encoding="latin1")):
                        try:
                            t = pd.read_csv(p, **kw)
                            if not t.empty: break
                        except: t = pd.DataFrame()
                if t is None or t.empty: 
                    print("[ENRICH][SKIP METRICS] vazio:", os.path.basename(p)); 
                    continue

                t = t.loc[:, ~t.columns.duplicated()]
                tcol = _pick_time_col(t)
                if not tcol:
                    print("[ENRICH][SKIP METRICS] sem coluna de tempo:", os.path.basename(p),
                          "cols=", list(t.columns)[:12]); 
                    continue

                # tempo → datetime UTC
                ts = pd.to_numeric(t[tcol], errors="coerce")
                if ts.notna().sum() >= max(1, int(0.5*len(t))):
                    if ts.max() < 1e12: ts = ts * 1000
                    t["_time"] = pd.to_datetime(ts.astype("int64"), unit="ms", utc=True)
                else:
                    t["_time"] = pd.to_datetime(t[tcol], errors="coerce", utc=True)

                t = t[(t["_time"] >= t0) & (t["_time"] <= t1)]
                if t.empty: 
                    continue

                cols = set(t.columns)
                def safe_div(num, den):
                    num = pd.to_numeric(t.get(num), errors="coerce")
                    den = pd.to_numeric(t.get(den), errors="coerce").replace(0, pd.NA)
                    return (num / den).astype("float64")

                feat = pd.DataFrame({"_time": t["_time"]})

                if "sum_open_interest" in cols:       feat["mx_oi"]        = pd.to_numeric(t["sum_open_interest"], errors="coerce")
                if "sum_open_interest_value" in cols:  feat["mx_oi_value"]  = pd.to_numeric(t["sum_open_interest_value"], errors="coerce")
                if {"sum_toptrader_long_short_ratio","count_toptrader_long_short_ratio"} <= cols:
                    feat["mx_toptrader_lsr"] = safe_div("sum_toptrader_long_short_ratio","count_toptrader_long_short_ratio")
                if {"sum_long_short_ratio","count_long_short_ratio"} <= cols:
                    feat["mx_crowd_lsr"]     = safe_div("sum_long_short_ratio","count_long_short_ratio")
                if "sum_taker_long_short_vol_ratio" in cols:
                    feat["mx_taker_lsr_vol"] = pd.to_numeric(t["sum_taker_long_short_vol_ratio"], errors="coerce")

                # Se nada específico encontrado, agrega tudo numérico
                if feat.shape[1] == 1:
                    drops = {"_time", "symbol","pair","contract","asset"}
                    for c in t.columns:
                        if c not in drops and pd.api.types.is_numeric_dtype(t[c]):
                            feat[f"mx_{c}"] = pd.to_numeric(t[c], errors="coerce")

                # resample por barra (média)
                rs = feat.set_index("_time").resample(rule, label="right", closed="right").mean()
                rs = rs.reset_index().rename(columns={"_time":"time"})
                mx_list.append(rs)
            except Exception as e:
                print(f"[ENRICH][SKIP METRICS] {os.path.basename(p)} -> {e}")

        if mx_list:
            mx_feat = pd.concat(mx_list, ignore_index=True)
            df = df.merge(mx_feat, on="time", how="left")


    # =============== Normalizações & lags ===============
    if "cvd_slope_agg" in df.columns:
        base = (df["atr_pct"]*df["close"]).rolling(20).mean()
        df["cvd_norm"] = df["cvd_slope_agg"] / (base + 1e-9)
        df["cvd_slope_agg"] = _clip_std(df["cvd_slope_agg"])
    if "imb_net_depth" in df.columns:
        df["imb_zscore"] = (df["imb_net_depth"] - df["imb_net_depth"].rolling(50).mean()) / (df["imb_net_depth"].rolling(50).std() + 1e-9)

    lag_cols = ["cvd_slope_agg","cvd_norm","imb_net_depth","imb_ratio_depth","imb_zscore",
                "fundingRate","premiumIndex","rsi14","macd","boll_pos","mark_close_ratio"]
    for L in (1,3,5):
        for c in lag_cols:
            if c in df.columns:
                df[f"{c}_lag{L}"] = df[c].shift(L)

    # === garantir unicidade de barras por 'time' antes do retorno ===
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    # limpeza final e tipos compactos
    df = df.replace([np.inf, -np.inf], np.nan)
    df["time_int"] = (df["time"].astype("int64") // 10**6).astype("int64")
    df = _downcast_inplace_local(df)

    if "cvd_slope_agg" not in df.columns: df["cvd_slope_agg"] = 0.0
    if "imb_net_depth" not in df.columns: df["imb_net_depth"] = 0.0
    return df


def parse_when(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts

def _parquet_list(path_or_dir: str):
    if not path_or_dir:
        return []
    if os.path.isdir(path_or_dir):
        return sorted(glob.glob(os.path.join(path_or_dir, "*.parquet")))
    return [path_or_dir] if path_or_dir.endswith(".parquet") and os.path.exists(path_or_dir) else []

def _tf_rule(tf: str) -> str:
    tf = tf.lower().strip()
    if tf.endswith("m"): return f"{int(tf[:-1])}min"
    if tf.endswith("h"): return f"{int(tf[:-1])}H"   # padroniza maiúsculo
    if tf.endswith("d"): return f"{int(tf[:-1])}D"   # padroniza maiúsculo
    return "1min"

def _infer_bar_minutes_from_df(df: pd.DataFrame) -> int:
    """Infere duração de barra em minutos (mediana do delta de 'time')."""
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    d = t.diff().dropna()
    if d.empty:
        return 1
    mins = int(round(d.dt.total_seconds().median() / 60.0))
    return max(1, mins)

def _downcast_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numéricas para reduzir RAM (mantém 'time')."""
    df = df.copy()
    for col in df.columns:
        if col == "time":
            continue
        s = df[col]
        try:
            if pd.api.types.is_float_dtype(s):
                df.loc[:, col] = s.astype(np.float32, copy=False)
            elif pd.api.types.is_integer_dtype(s):
                df.loc[:, col] = s.astype(np.int32, copy=False)
            elif pd.api.types.is_bool_dtype(s):
                pass
        except Exception:
            pass
    return df

def _fix_time_col(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    """
    Corrige timestamps em micros/nanos para milissegundos.
    Se max(time) < 1e12, multiplica por 1000.
    """
    if col not in df.columns:
        return df
    try:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].max() < 1e12:
            df[col] *= 1000
        df[col] = df[col].astype("int64")
    except Exception:
        pass
    return df

def _round_to_tick(price: float, tick: float) -> float:
    """Arredonda preço ao múltiplo mais próximo de tick_size."""
    try:
        if tick is None or tick <= 0:
            return float(price)
        return float(round(price / tick) * tick)
    except Exception:
        return float(price)

# ============================ Leitura/normalização OHLCV ============================

def _read_table_auto(path: str, usecols=None, parse_dates=None):
    pl = path.lower()
    if pl.endswith(".parquet"):
        df = pd.read_parquet(path, columns=usecols)
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return df
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)
    except Exception:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, engine="python", on_bad_lines="skip")

def _normalize_ohlcv(df_in: pd.DataFrame, symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Normaliza dados OHLCV, mesmo que venham com colunas numéricas (0,1,2,...).
    Retorna (DataFrame normalizado, motivo_erro se None).
    """
    df = df_in.copy()
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    lower = {c.lower(): c for c in df.columns}

    # Detectar formato numérico 0..11 (Binance padrão sem header)
    if all(str(c).isdigit() for c in df.columns) and len(df.columns) >= 6:
        exp = ['time','open','high','low','close','volume',
               'close_time','quote_vol','n_trades','taker_buy_vol','taker_buy_quote_vol','ignore']
        df.columns = exp[:len(df.columns)]
        lower = {c.lower(): c for c in df.columns}

    # coluna time
    time_candidates = ["time","timestamp","date","datetime","open_time","close_time","ts"]
    time_col = next((lower[c] for c in time_candidates if c in lower), None)
    if time_col is None:
        return None, "sem_coluna_time"

    s = pd.to_numeric(df[time_col], errors="coerce")
    med = float(s.dropna().median()) if s.notna().any() else 0
    if med > 1e14:
        t = pd.to_datetime(s, unit="ns", utc=True, errors="coerce")
    elif med > 1e11:
        t = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
    else:
        t = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    df["time"] = t

    # OHLCV
    def pick(*cands): return next((lower[c] for c in cands if c in lower), None)
    o = pick("open","o","1"); h = pick("high","h","2"); l = pick("low","l","3"); c = pick("close","c","price","4")
    if not all([o,h,l,c]):
        return None, "sem_ohlc"

    # filtro por símbolo, se existir
    sym_col = pick("symbol","sym","ticker","asset")
    if sym_col:
        df = df[df[sym_col].astype(str).str.upper() == str(symbol).upper()]

    vol_col = pick("volume","vol","qty","5")
    out = pd.DataFrame({
        "time": df["time"],
        "open":   pd.to_numeric(df[o], errors="coerce"),
        "high":   pd.to_numeric(df[h], errors="coerce"),
        "low":    pd.to_numeric(df[l], errors="coerce"),
        "close":  pd.to_numeric(df[c], errors="coerce"),
        "volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0
    }).dropna()

    out = out.sort_values("time").reset_index(drop=True)
    return _downcast_inplace(out), ""

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV com label/closed à direita, usando sintaxe de `_tf_rule`."""
    rule_real = _tf_rule(rule)  # 'Xmin', 'XH', 'XD'
    g = df.set_index("time").resample(rule_real, label="right", closed="right")
    out = pd.DataFrame({
        "time":   g["close"].last().index,
        "open":   g["open"].first(),
        "high":   g["high"].max(),
        "low":    g["low"].min(),
        "close":  g["close"].last(),
        "volume": g["volume"].sum(),
    }).dropna().reset_index(drop=True)
    return _downcast_inplace(out)

def read_local_data(symbol: str, requested_tfs: List[str] | None = None, base_dir: str = ".", data_glob: Optional[str]=None) -> Dict[str, pd.DataFrame]:
    # Seleção de paths
    if data_glob:
        globs = [g.strip() for g in str(data_glob).split(";") if g.strip()]
        paths = []
        for pat in globs:
            if os.path.isabs(pat):
                paths += glob.glob(pat, recursive=True)
            else:
                paths += glob.glob(os.path.join(base_dir, pat), recursive=True)
    else:
        if os.path.isfile(base_dir):
            paths = [base_dir]
        else:
            paths = sorted(set(
                list(glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)) +
                list(glob.glob(os.path.join(base_dir, "**", "*.parquet"), recursive=True)) +
                list(glob.glob(os.path.join(base_dir, "**", "*.zip"), recursive=True))
            ))

    # Filtro por símbolo (se o glob não trancar o símbolo)
    if not (os.path.isfile(base_dir) or (data_glob and str(symbol).upper() in str(data_glob).upper())):
        paths = [p for p in paths if str(symbol).upper() in os.path.basename(p).upper()]

    if not paths:
        return {}

    frames = []
    for p in paths:
        try:
            df = _read_table_auto(p)
        except Exception:
            continue
        norm, _why = _normalize_ohlcv(df, symbol)
        if norm is None or norm.empty:
            continue
        frames.append(norm)

    if not frames:
        return {}

    df_all = pd.concat(frames, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    if len(df_all) < 3:
        return {}

    # Infere TF base
    deltas = df_all["time"].diff().dropna().dt.total_seconds()/60.0
    median_min = max(1, int(np.median(deltas))) if deltas.notna().any() else 1
    tf_map = {1:"1m",3:"3m",5:"5m",10:"10m",15:"15m",30:"30m",60:"1h"}
    base_tf = tf_map.get(median_min, f"{median_min}m")
    out = {base_tf: df_all}

    # Resample TFs solicitadas
    if requested_tfs:
        for tf in requested_tfs:
            tf = tf.strip()
            if not tf or tf == base_tf:
                continue
            try:
                out[tf] = resample_ohlcv(df_all, tf)
            except Exception:
                pass

    return out

# ========================== Loaders AGG/DEPTH ==========================

import re
from datetime import datetime

def _file_in_range(fname: str, start_ms: int, end_ms: int) -> bool:
    """Retorna True se o nome do arquivo contém uma data dentro do range."""
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", fname)
    if not match:
        return True  # se não houver data no nome, tenta ler
    y, m, d = map(int, match.groups())
    ts = int(datetime(y, m, d).timestamp() * 1000)
    return start_ms <= ts <= end_ms

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"]  - prev).abs(),
    ], axis=1).max(axis=1)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).rolling(length, min_periods=length).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    gain = up.rolling(length, min_periods=length).mean()
    loss = dn.rolling(length, min_periods=length).mean()
    rs = (gain / loss.replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)

def vwap_win(df: pd.DataFrame, length: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df.get("volume", pd.Series(0.0, index=df.index)).astype(float).replace(0, np.nan)
    pv = (tp*vol).rolling(length, min_periods=length).sum()
    vv = vol.rolling(length, min_periods=length).sum()
    return (pv/vv)

def macd_line(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ema(series, fast) - ema(series, slow)

def keltner_channels(df: pd.DataFrame, ema_len: int = 20, atr_len: int = 20, mult: float = 1.5):
    basis = ema(df["close"], ema_len)
    rng = atr(df, atr_len)
    upper = basis + mult * rng
    lower = basis - mult * rng
    return basis, upper, lower

def bollinger_bands(series: pd.Series, length: int = 20, k: float = 2.0):
    m = series.rolling(length, min_periods=length).mean()
    sd = series.rolling(length, min_periods=length).std(ddof=0)
    upper = m + k*sd
    lower = m - k*sd
    return m, upper, lower

def avwap_anchored(df: pd.DataFrame, anchor: str = "D") -> pd.Series:
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    vol = df.get("volume", pd.Series(0.0, index=df.index)).astype(float)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    if str(anchor).upper().startswith("W"):
        grp = t.dt.to_period("W-MON").dt.start_time
    else:
        grp = t.dt.floor("D")
    pv = (tp * vol)
    df_aux = pd.DataFrame({"grp": grp, "pv": pv, "vol": vol}, index=df.index)
    pv_cum = df_aux.groupby("grp")["pv"].cumsum()
    vol_cum = df_aux.groupby("grp")["vol"].cumsum().replace(0, np.nan)
    return (pv_cum / vol_cum).ffill().fillna(df["close"])

# ===================== Estratégias BASE (subset robusto) =====================
def signals_trend_breakout(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    don = int(params.get("donchian", 20))
    ef = int(params.get("ema_fast", 12)); es = int(params.get("ema_slow", 50))
    hh = df["high"].rolling(don, min_periods=don).max()
    ll = df["low"].rolling(don, min_periods=don).min()
    trend_up = ema(df["close"], ef) > ema(df["close"], es)
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[(df["close"] > hh.shift(1)) & trend_up] = 1
    if futures:
        sig[(df["close"] < ll.shift(1)) & (~trend_up)] = -1
    return sig.fillna(0).astype(np.int8)

def signals_keltner_breakout(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    e = int(params.get("ema_len", 20)); a = int(params.get("atr_len", 20)); m = float(params.get("mult", 1.5))
    basis, up, lo = keltner_channels(df, e, a, m)
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[df["close"] > up] = 1
    if futures:
        sig[df["close"] < lo] = -1
    return sig.fillna(0).astype(np.int8)

def signals_rsi_reversion(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    rl = int(params.get("rsi_len", 14)); rs = float(params.get("rsi_short", 30)); rlng = float(params.get("rsi_long",70))
    r = rsi(df["close"], rl)
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[r < rs] = 1
    if futures:
        sig[r > rlng] = -1
    return sig.fillna(0).astype(np.int8)

def signals_ema_crossover(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    f = int(params.get("ema_fast",12)); s = int(params.get("ema_slow",26))
    ef = ema(df["close"], f); es = ema(df["close"], s)
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[ef > es] = 1
    if futures:
        sig[ef < es] = -1
    return sig.fillna(0).astype(np.int8)

def signals_macd_trend(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    fast = int(params.get("macd_fast",12)); slow = int(params.get("macd_slow",26)); siglen = int(params.get("macd_signal",9))
    mline = macd_line(df["close"], fast, slow); sigl = ema(mline, siglen)
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[mline > sigl] = 1
    if futures:
        sig[mline < sigl] = -1
    return sig.fillna(0).astype(np.int8)

def signals_vwap_trend(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    L = int(params.get("len",20)); v = vwap_win(df, L)
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[df["close"] > v] = 1
    if futures:
        sig[df["close"] < v] = -1
    return sig.fillna(0).astype(np.int8)

def signals_boll_breakout(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    bb_len = int(params.get("bb_len", 20)); bb_k = float(params.get("bb_k", 2.0))
    m, up, lo = bollinger_bands(df["close"], bb_len, bb_k)
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[df["close"] > up] = 1
    if futures:
        sig[df["close"] < lo] = -1
    return sig.fillna(0).astype(np.int8)

def signals_orb_breakout(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    init_min = int(params.get("init_minutes", 5))
    vol_q = float(params.get("vol_q", 0.6))
    t = pd.to_datetime(df["time"], utc=True)
    day = t.dt.floor("D")
    bar_in_day = (t.groupby(day).cumcount()).astype(int)
    tf_minutes = _infer_bar_minutes_from_df(df)
    init_bars = max(1, int(round(init_min / max(1, tf_minutes))))
    in_open = (bar_in_day < init_bars)

    hh = df["high"].where(in_open).groupby(day).transform(lambda s: s.max())
    ll = df["low"].where(in_open).groupby(day).transform(lambda s: s.min())

    vol = df.get("vol_per_bar", df.get("volume", pd.Series(0.0, index=df.index))).astype(float)
    if vol.eq(0).all():
        vol_ok = pd.Series(True, index=df.index)
    else:
        thr = vol.groupby(day).transform(lambda s: s.quantile(vol_q))
        vol_ok = vol >= thr.ffill().fillna(vol.median())

    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[(df["close"] > hh.shift(1)) & vol_ok] = 1
    if futures:
        sig[(df["close"] < ll.shift(1)) & vol_ok] = -1
    return sig.fillna(0).astype(np.int8)

def signals_orr_reversal(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    init_min = int(params.get("init_minutes", 5))
    t = pd.to_datetime(df["time"], utc=True)
    day = t.dt.floor("D")
    bar_in_day = (t.groupby(day).cumcount()).astype(int)
    tf_minutes = _infer_bar_minutes_from_df(df)
    init_bars = max(1, int(round(init_min / max(1, tf_minutes))))
    in_open = (bar_in_day < init_bars)

    hh = df["high"].where(in_open).groupby(day).transform(lambda s: s.max())
    ll = df["low"].where(in_open).groupby(day).transform(lambda s: s.min())

    sig = pd.Series(0, index=df.index, dtype=np.int8)
    fail_long = (df["close"].shift(1) > hh.shift(1)) & (df["close"] < hh)
    fail_short= (df["close"].shift(1) < ll.shift(1)) & (df["close"] > ll)
    sig[fail_short] = 1
    if futures:
        sig[fail_long] = -1
    return sig.fillna(0).astype(np.int8)

def signals_ema_pullback(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    f = int(params.get("ema_fast", 9)); s=int(params.get("ema_slow",20))
    ef = ema(df["close"], f); es = ema(df["close"], s)
    trend_up = ef > es
    close = df["close"]; open_ = df["open"]
    touch_f = (close >= ef) & (open_ <= ef)
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[touch_f & trend_up] = 1
    if futures:
        touch_f_s = (close <= ef) & (open_ >= ef)
        sig[touch_f_s & (~trend_up)] = -1
    return sig.fillna(0).astype(np.int8)

def signals_donchian_breakout(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    n = int(params.get("n", 20))
    hh = df["high"].rolling(n, min_periods=n).max()
    ll = df["low"].rolling(n, min_periods=n).min()
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[df["close"] > hh.shift(1)] = 1
    if futures:
        sig[df["close"] < ll.shift(1)] = -1
    return sig.fillna(0).astype(np.int8)

def signals_vwap_poc_reject(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    anchor = params.get("anchor", "D")
    wick_frac = float(params.get("wick_frac", 0.5))
    vol_q = float(params.get("vol_q", 0.6))
    v = avwap_anchored(df, anchor=anchor)
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    rng = (h - l).replace(0, np.nan)
    lower_wick = (o.combine(c, min) - l) / (rng + 1e-9)
    upper_wick = (h - o.combine(c, max)) / (rng + 1e-9)

    t = pd.to_datetime(df["time"], utc=True)
    day = t.dt.floor("D")
    vol = df.get("vol_per_bar", df.get("volume", pd.Series(0.0, index=df.index))).astype(float)
    if vol.eq(0).all():
        vol_ok = pd.Series(True, index=df.index)
    else:
        thr = vol.groupby(day).transform(lambda s: s.quantile(vol_q))
        vol_ok = vol >= thr.ffill().fillna(vol.median())

    bull = (l <= v) & (c > v) & (lower_wick >= wick_frac) & vol_ok
    bear = (h >= v) & (c < v) & (upper_wick >= wick_frac) & vol_ok

    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[bull] = 1
    if futures:
        sig[bear] = -1
    return sig.fillna(0).astype(np.int8)

def signals_ob_imbalance_break(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    if "imb_net_depth" not in df.columns:
        return pd.Series(0, index=df.index, dtype=np.int8)
    imb = df["imb_net_depth"].astype(float)
    k = float(params.get("imb_thr_mult", 1.5))
    thr = k * (imb.abs().rolling(100, min_periods=20).median().fillna(imb.abs().median()))
    long_cond  = (imb > thr) & (df["close"] > df["high"].shift(1))
    short_cond = (imb < -thr) & (df["close"] < df["low"].shift(1))
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[long_cond] = 1
    if futures:
        sig[short_cond] = -1
    return sig.fillna(0).astype(np.int8)

def signals_cvd_divergence_reversal(df: pd.DataFrame, params: dict, futures: bool) -> pd.Series:
    if "cvd" not in df.columns:
        return pd.Series(0, index=df.index, dtype=np.int8)
    N = int(params.get("div_lookback", 20))
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    cvd = df["cvd"].astype(float)
    price_dd_long  = l < l.shift(N)
    price_dd_short = h > h.shift(N)
    cvd_up  = (cvd - cvd.shift(N)) > 0
    cvd_dn  = (cvd - cvd.shift(N)) < 0
    bull_conf = c > o
    bear_conf = c < o
    sig = pd.Series(0, index=df.index, dtype=np.int8)
    sig[price_dd_long & cvd_up & bull_conf] = 1
    if futures:
        sig[price_dd_short & cvd_dn & bear_conf] = -1
    return sig.fillna(0).astype(np.int8)

# registro de estratégias
STRATEGY_FUNCS = {
    "trend_breakout":                 signals_trend_breakout,
    "keltner_breakout":               signals_keltner_breakout,
    "rsi_reversion":                  signals_rsi_reversion,
    "ema_crossover":                  signals_ema_crossover,
    "macd_trend":                     signals_macd_trend,
    "vwap_trend":                     signals_vwap_trend,
    "boll_breakout":                  signals_boll_breakout,
    "orb_breakout":                   signals_orb_breakout,
    "orr_reversal":                   signals_orr_reversal,
    "ema_pullback":                   signals_ema_pullback,
    "donchian_breakout":              signals_donchian_breakout,
    "vwap_poc_reject":                signals_vwap_poc_reject,
    "ob_imbalance_break":             signals_ob_imbalance_break,
    "cvd_divergence_reversal":        signals_cvd_divergence_reversal,
}

# ============================ Backtest ============================

def _build_combo_signal_feats(df: pd.DataFrame, combo_specs: List[str], futures: bool,
                              default_window: int = 2, default_maj_votes: int = 2) -> pd.DataFrame:
    """
    Constrói colunas de sinal (shifted) para cada combo em combo_specs.
    Requer: parse_combo_spec, _combine_AND/MAJ/SEQ e STRATEGY_FUNCS definidos no arquivo.
    """
    out = {}
    for spec in combo_specs:
        try:
            cs = parse_combo_spec(spec, default_window=default_window, default_maj_votes=default_maj_votes)
            sigs = []
            for mname in cs.items:
                if mname not in STRATEGY_FUNCS:
                    raise ValueError(f"método inválido: {mname}")
                s = STRATEGY_FUNCS[mname](df, {}, futures).shift(1).fillna(0).astype(np.int8)
                sigs.append(s)
            if cs.op == "AND":
                sc = _combine_AND(sigs, window=cs.window, futures=futures)
            elif cs.op == "MAJ":
                sc = _combine_MAJ(sigs, k=cs.k, window=cs.window, futures=futures)
            else:
                sc = _combine_SEQ(sigs, window=cs.window, futures=futures)
            cname = re.sub(r'[^A-Za-z0-9_]+', '_', f"{cs.op}_" + "_".join(cs.items))[:64]
            out[cname] = sc.shift(1).fillna(0).astype(np.int8)   # shift p/ evitar look-ahead
        except Exception as e:
            print(f"[ML][combo_feat][skip] {spec} -> {e}")
    return pd.DataFrame(out, index=df.index) if out else pd.DataFrame(index=df.index)
# ==== HARD STOP/TP helper: preço de saída para atingir PnL alvo (líquido) ====
def _exit_px_for_target_pnl(entry_px: float, target_pnl_usd: float, side: int,
                            fee_perc: float, contracts: float, contract_value: float) -> float:
    """
    Resolve o preço de saída (sem slippage) que gera PnL líquido = target_pnl_usd,
    considerando fees na entrada e saída: fee * (entry + exit) * (contracts * contract_value).
    side: +1 long, -1 short
    """
    fee = float(fee_perc)
    N = float(contracts) * float(contract_value)
    T = float(target_pnl_usd)  # +TP ou -STOP
    if N <= 0:
        return float(entry_px)

    if side > 0:  # long
        # Net = N*(exit - entry) - fee*N*(entry + exit) = T
        # exit = (entry*(1+fee) + T/N) / (1 - fee)
        denom = max(1e-12, (1.0 - fee))
        return (entry_px * (1.0 + fee) + (T / N)) / denom
    else:         # short
        # Net = N*(entry - exit) - fee*N*(entry + exit) = T
        # entry*(1-fee) - exit*(1+fee) = T/N  ->  exit = (entry*(1-fee) - T/N) / (1+fee)
        return (entry_px * (1.0 - fee) - (T / N)) / (1.0 + fee)

def backtest_from_signals(
    df: pd.DataFrame,
    sig: pd.Series,
    *,
    max_hold: int = 480,
    fee_perc: float = 0.0002,
    slippage_ticks: float = 0,
    tick_size: float = 0.01,
    contracts: float = 1.0,
    contract_value: float = 1.0,
    futures: bool = True,
    # Stops
    use_atr_stop: bool = False,
    atr_stop_len: int = 14,
    atr_stop_mult: float = 1.5,
    # TP por ATR
    use_atr_tp: bool = False,
    atr_tp_len: int = 14,
    atr_tp_mult: float = 0.0,
    # Stop por Candle
    use_candle_stop: bool = False,
    candle_stop_lookback: int = 1,
    # Trailing & timeouts
    trailing: bool = False,
    timeout_mode: str = "bars",          # bars | atr | both
    atr_timeout_len: int = 14,
    atr_timeout_mult: float = 0.0,       # 0 => desliga timeout ATR
    # Execução
    round_to_tick: bool = False,
    # HARD Stop/TP em USD
    hard_stop_usd: float = 0.0,
    hard_tp_usd: float = 0.0
) -> pd.DataFrame:
    """
    Execução realista com:
      - slippage aplicado entrada/saída
      - stop intrabar (ATR/candle/HARD) com prioridade correta
      - TP intrabar (ATR/HARD)
      - flip fecha posição anterior antes de abrir na direção oposta
      - timeouts por ATR acumulado e por barras
    """
    df = df.reset_index(drop=True).copy()
    sig = sig.reindex(df.index).fillna(0).astype(np.int8)

    slip = float(slippage_ticks) * float(tick_size)
    notional_mult = float(contracts) * float(contract_value)

    # Séries auxiliares
    atr_stop_series = atr(df, atr_stop_len) if use_atr_stop else pd.Series(np.nan, index=df.index)
    atr_to_series   = atr(df, atr_timeout_len) if (timeout_mode in ("atr","both") and atr_timeout_mult>0) else pd.Series(np.nan, index=df.index)
    atr_tp_series   = atr(df, atr_tp_len) if use_atr_tp else pd.Series(np.nan, index=df.index)

    trades = []
    pos_dir = 0  # +1 long, -1 short, 0 flat
    entry_idx = None
    entry_px  = None
    atr_stop_px = None
    candle_stop_px = None
    hard_stop_px = None
    tp_px = None
    hard_tp_px = None
    bars_held = 0
    atr_budget = 0.0
    atr_accum = 0.0

    # helpers candle stop (mín/máx últimas N)
    def _candle_stop_long(i: int):
        j0 = max(0, i - candle_stop_lookback + 1)
        return float(df["low"].iloc[j0:i+1].min()) if i>=0 else None
    def _candle_stop_short(i: int):
        j0 = max(0, i - candle_stop_lookback + 1)
        return float(df["high"].iloc[j0:i+1].max()) if i>=0 else None

    def _effective_stop(side: int):
        stops = []
        if atr_stop_px is not None:    stops.append(float(atr_stop_px))
        if candle_stop_px is not None: stops.append(float(candle_stop_px))
        if hard_stop_px is not None:   stops.append(float(hard_stop_px))
        if not stops:
            return None
        if side > 0:   # long: piores são stops mais altos (mais próximos)
            return max(stops)
        else:          # short: piores são stops mais baixos (mais próximos)
            return min(stops)

    def _effective_tp(side: int):
        tps = []
        if tp_px is not None:      tps.append(float(tp_px))
        if hard_tp_px is not None: tps.append(float(hard_tp_px))
        if not tps:
            return None
        if side > 0:   # long: alvo mais próximo é o menor acima
            return min(tps)
        else:          # short: alvo mais próximo é o maior abaixo
            return max(tps)

    for i in range(len(df)):
        px = float(df.at[i, "close"])
        bar_low  = float(df.at[i, "low"])
        bar_high = float(df.at[i, "high"])
        s = int(sig.iat[i])

        # ========= Trailing / atualização de stops =========
        if pos_dir != 0:
            # ATR stop trailing (chandelier)
            if use_atr_stop and not np.isnan(atr_stop_series.iat[i]):
                dist = float(atr_stop_series.iat[i]) * float(atr_stop_mult)
                if pos_dir > 0:
                    atr_stop_px = max(atr_stop_px if atr_stop_px is not None else -1e18, px - dist)
                else:
                    atr_stop_px = min(atr_stop_px if atr_stop_px is not None else 1e18, px + dist)
            # Candle stop trailing
            if use_candle_stop and trailing:
                candle_stop_px = _candle_stop_long(i) if pos_dir>0 else _candle_stop_short(i)

        # ========= 1) STOP intrabar tem precedência =========
        eff_stop = _effective_stop(pos_dir)
        if pos_dir > 0 and eff_stop is not None and bar_low <= eff_stop:
            exit_px = eff_stop - slip
            if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
            trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "LONG", bars_held))
            pos_dir=0; entry_idx=None; entry_px=None
            atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
            tp_px=None; hard_tp_px=None
            bars_held=0; atr_budget=0.0; atr_accum=0.0
            continue
        if pos_dir < 0 and eff_stop is not None and bar_high >= eff_stop:
            exit_px = eff_stop + slip
            if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
            trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "SHORT", bars_held))
            pos_dir=0; entry_idx=None; entry_px=None
            atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
            tp_px=None; hard_tp_px=None
            bars_held=0; atr_budget=0.0; atr_accum=0.0
            continue

        # ========= 2) TP intrabar =========
        eff_tp = _effective_tp(pos_dir)
        if pos_dir > 0 and eff_tp is not None and bar_high >= eff_tp:
            exit_px = eff_tp - slip
            if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
            trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "LONG", bars_held))
            pos_dir=0; entry_idx=None; entry_px=None
            atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
            tp_px=None; hard_tp_px=None
            bars_held=0; atr_budget=0.0; atr_accum=0.0
            continue
        if pos_dir < 0 and eff_tp is not None and bar_low <= eff_tp:
            exit_px = eff_tp + slip
            if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
            trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "SHORT", bars_held))
            pos_dir=0; entry_idx=None; entry_px=None
            atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
            tp_px=None; hard_tp_px=None
            bars_held=0; atr_budget=0.0; atr_accum=0.0
            continue

        # ========= 3) Timeout por ATR acumulado =========
        if pos_dir != 0 and (timeout_mode in ("atr","both")) and atr_timeout_mult > 0 and not np.isnan(atr_to_series.iat[i]):
            atr_accum += float(atr_to_series.iat[i])
            if atr_budget > 0 and atr_accum >= atr_budget:
                exit_px = (px - slip) if pos_dir>0 else (px + slip)
                if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
                trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "LONG" if pos_dir>0 else "SHORT", bars_held))
                pos_dir=0; entry_idx=None; entry_px=None
                atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
                tp_px=None; hard_tp_px=None
                bars_held=0; atr_budget=0.0; atr_accum=0.0
                continue

        # ========= 4) Processar sinais (com flip correto) =========
        if s == 0:
            if pos_dir != 0:
                exit_px = (px - slip) if pos_dir>0 else (px + slip)
                if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
                trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "LONG" if pos_dir>0 else "SHORT", bars_held))
                pos_dir=0; entry_idx=None; entry_px=None
                atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
                tp_px=None; hard_tp_px=None
                bars_held=0; atr_budget=0.0; atr_accum=0.0

        elif s > 0:
            # flip short->long se necessário
            if pos_dir < 0:
                exit_px = px + slip
                if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
                trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "SHORT", bars_held))
                pos_dir=0; entry_idx=None; entry_px=None
                atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
                tp_px=None; hard_tp_px=None
                bars_held=0; atr_budget=0.0; atr_accum=0.0
            if pos_dir == 0:
                pos_dir=1; entry_idx=i; entry_px=px+slip
                if round_to_tick: entry_px = _round_to_tick(entry_px, tick_size)
                bars_held=0; atr_accum=0.0
                # ATR stop inicial
                if use_atr_stop and not np.isnan(atr_stop_series.iat[i]):
                    dist = float(atr_stop_series.iat[i]) * float(atr_stop_mult)
                    atr_stop_px = entry_px - dist
                else:
                    atr_stop_px = None
                # Candle stop
                candle_stop_px = _candle_stop_long(i) if use_candle_stop else None
                # HARD stop/TP por USD (preço alvo líquido com fees)
                if hard_stop_usd and notional_mult>0:
                    hard_stop_px = _exit_px_for_target_pnl(entry_px, -abs(hard_stop_usd), +1, fee_perc, contracts, contract_value)
                else:
                    hard_stop_px = None
                if hard_tp_usd and notional_mult>0:
                    hard_tp_px = _exit_px_for_target_pnl(entry_px, +abs(hard_tp_usd), +1, fee_perc, contracts, contract_value)
                else:
                    hard_tp_px = None
                # TP ATR
                if use_atr_tp and not np.isnan(atr_tp_series.iat[i]) and atr_tp_mult > 0:
                    tp_px = entry_px + float(atr_tp_series.iat[i]) * float(atr_tp_mult)
                else:
                    tp_px = None
                # Budget ATR timeout
                if (timeout_mode in ("atr","both")) and atr_timeout_mult>0 and not np.isnan(atr_to_series.iat[i]):
                    atr_budget = float(atr_to_series.iat[i]) * float(atr_timeout_mult)
                else:
                    atr_budget = 0.0

        elif s < 0 and futures:
            # flip long->short se necessário
            if pos_dir > 0:
                exit_px = px - slip
                if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
                trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "LONG", bars_held))
                pos_dir=0; entry_idx=None; entry_px=None
                atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
                tp_px=None; hard_tp_px=None
                bars_held=0; atr_budget=0.0; atr_accum=0.0
            if pos_dir == 0:
                pos_dir=-1; entry_idx=i; entry_px=px-slip
                if round_to_tick: entry_px = _round_to_tick(entry_px, tick_size)
                bars_held=0; atr_accum=0.0
                # ATR stop inicial
                if use_atr_stop and not np.isnan(atr_stop_series.iat[i]):
                    dist = float(atr_stop_series.iat[i]) * float(atr_stop_mult)
                    atr_stop_px = entry_px + dist
                else:
                    atr_stop_px = None
                candle_stop_px = _candle_stop_short(i) if use_candle_stop else None
                # HARD stop/TP por USD (preço alvo líquido com fees)
                if hard_stop_usd and notional_mult>0:
                    hard_stop_px = _exit_px_for_target_pnl(entry_px, -abs(hard_stop_usd), -1, fee_perc, contracts, contract_value)
                else:
                    hard_stop_px = None
                if hard_tp_usd and notional_mult>0:
                    hard_tp_px = _exit_px_for_target_pnl(entry_px, +abs(hard_tp_usd), -1, fee_perc, contracts, contract_value)
                else:
                    hard_tp_px = None
                # TP ATR
                if use_atr_tp and not np.isnan(atr_tp_series.iat[i]) and atr_tp_mult > 0:
                    tp_px = entry_px - float(atr_tp_series.iat[i]) * float(atr_tp_mult)
                else:
                    tp_px = None
                if (timeout_mode in ("atr","both")) and atr_timeout_mult>0 and not np.isnan(atr_to_series.iat[i]):
                    atr_budget = float(atr_to_series.iat[i]) * float(atr_timeout_mult)
                else:
                    atr_budget = 0.0

        # ========= 5) Timeout por barras (ao fim da barra) =========
        if pos_dir != 0:
            bars_held += 1
            if (timeout_mode in ("bars","both")) and max_hold>0 and bars_held >= max_hold:
                exit_px = (px - slip) if pos_dir>0 else (px + slip)
                if round_to_tick: exit_px = _round_to_tick(exit_px, tick_size)
                trades.append((df.at[entry_idx,"time"], df.at[i,"time"], entry_px, exit_px, "LONG" if pos_dir>0 else "SHORT", bars_held))
                pos_dir=0; entry_idx=None; entry_px=None
                atr_stop_px=None; candle_stop_px=None; hard_stop_px=None
                tp_px=None; hard_tp_px=None
                bars_held=0; atr_budget=0.0; atr_accum=0.0

    cols = ["entry_time","exit_time","entry_price","exit_price","side","bars_held"]
    out = pd.DataFrame(trades, columns=cols)
    return _downcast_inplace(out)


def _metrics_with_fees(trades: pd.DataFrame, fee_perc: float,
                       *, contracts: float = 1.0, contract_value: float = 1.0) -> Dict[str, float]:
    """
    Calcula métricas considerando contract_value como QUANTIDADE de ativo (ex.: BTC por contrato).
    """
    if trades is None or trades.empty:
        return {"expectancy":0.0,"hit":0.0,"sharpe":0.0,"n_trades":0,
                "total_pnl":0.0,"maxdd":0.0,"payoff":0.0}

    qty = float(contracts) * float(contract_value)  # quantidade de ativo
    entry = trades["entry_price"].values.astype(float)
    exit_ = trades["exit_price"].values.astype(float)
    side = np.where(trades["side"].values=="LONG", 1.0, -1.0)

    pnl_raw = (exit_ - entry) * side * qty
    fees = fee_perc * (np.abs(entry) + np.abs(exit_)) * qty
    pnl = pd.Series(pnl_raw - fees, index=trades.index)

    hit = float((pnl > 0).mean())
    total = float(pnl.sum())
    sharpe = float(pnl.mean()/(pnl.std(ddof=1)+1e-9)*np.sqrt(len(pnl))) if len(pnl)>1 else 0.0
    maxdd = float((pnl.cumsum().cummax()-pnl.cumsum()).max() if len(pnl)>1 else 0.0)
    payoff = float((pnl[pnl>0].mean()/(-pnl[pnl<0].mean()+1e-9)) if (pnl>0).any() and (pnl<0).any() else 0.0)

    return {"expectancy":float(pnl.mean()),"hit":hit,"sharpe":sharpe,
            "n_trades":int(len(pnl)),"total_pnl":total,"maxdd":maxdd,"payoff":payoff}


# ===================== Combos =====================

@dataclass
class ComboSpec:
    op: str                # "AND" | "MAJ" | "SEQ"
    items: List[str]       # lista de métodos base
    k: int = 0             # mínimo de votos (MAJ) | para AND = len(items)
    window: int = 2        # barras para confirmação
    raw: str = ""          # string original

_COMBO_RE_AND = re.compile(r"^\s*AND\(\s*([^)]+?)\s*\)\s*(?:\[(.*?)\])?\s*$", re.I)
_COMBO_RE_MAJ = re.compile(r"^\s*MAJ(?:(\d+))?\(\s*([^)]+?)\s*\)\s*(?:\[(.*?)\])?\s*$", re.I)
_COMBO_RE_SEQ = re.compile(r"^\s*SEQ\(\s*([^)]+?)\s*\)\s*(?:\[(.*?)\])?\s*$", re.I)

def _parse_opts(optstr: Optional[str]) -> Dict[str,str]:
    out={}
    if not optstr: return out
    parts = [p.strip() for p in optstr.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            k,v = p.split("=",1); out[k.strip().lower()] = v.strip()
    return out

def parse_combo_spec(spec: str, default_window: int, default_maj_votes: int) -> ComboSpec:
    s = (spec or "").strip()
    m = _COMBO_RE_AND.match(s)
    if m:
        items = [x.strip() for x in m.group(1).split(",") if x.strip()]
        opts = _parse_opts(m.group(2))
        w = int(opts.get("w", opts.get("window", default_window)))
        return ComboSpec(op="AND", items=items, k=len(items), window=w, raw=spec)
    m = _COMBO_RE_MAJ.match(s)
    if m:
        k_num = m.group(1)
        items = [x.strip() for x in m.group(2).split(",") if x.strip()]
        opts = _parse_opts(m.group(3))
        w = int(opts.get("w", opts.get("window", default_window)))
        k = int(opts.get("k", k_num or default_maj_votes))
        k = max(1, min(len(items), k))
        return ComboSpec(op="MAJ", items=items, k=k, window=w, raw=spec)
    m = _COMBO_RE_SEQ.match(s)
    if m:
        chain = [x.strip() for x in m.group(1).split("->") if x.strip()]
        opts = _parse_opts(m.group(2))
        w = int(opts.get("w", opts.get("window", default_window)))
        return ComboSpec(op="SEQ", items=chain, k=len(chain), window=w, raw=spec)
    raise ValueError(f"Combo inválido: '{spec}'")

def _combine_AND(signals: List[pd.Series], window: int, futures: bool) -> pd.Series:
    if not signals:
        return pd.Series(0, index=pd.RangeIndex(0), dtype=np.int8)
    idx = signals[0].index
    longs = [ (s.eq(1).rolling(window, min_periods=1).max() > 0).astype(np.int8) for s in signals ]
    shorts= [ (s.eq(-1).rolling(window, min_periods=1).max() > 0).astype(np.int8) for s in signals ] if futures else []
    long_all  = np.minimum.reduce([x.values for x in longs]) if longs else np.zeros(len(idx), dtype=np.int8)
    short_all = np.minimum.reduce([x.values for x in shorts]) if shorts else np.zeros(len(idx), dtype=np.int8)
    out = np.zeros(len(idx), dtype=np.int8)
    out[(long_all==1) & (short_all==0)]  = 1
    out[(short_all==1) & (long_all==0)]  = -1
    return pd.Series(out, index=idx, dtype=np.int8)

def _combine_MAJ(signals: List[pd.Series], k: int, window: int, futures: bool) -> pd.Series:
    if not signals:
        return pd.Series(0, index=pd.RangeIndex(0), dtype=np.int8)
    idx = signals[0].index
    long_votes = np.zeros(len(idx), dtype=np.int16)
    short_votes= np.zeros(len(idx), dtype=np.int16)
    for s in signals:
        long_votes  += (s.eq(1).rolling(window, min_periods=1).max() > 0).astype(np.int16).values
        if futures:
            short_votes += (s.eq(-1).rolling(window, min_periods=1).max() > 0).astype(np.int16).values
    out = np.zeros(len(idx), dtype=np.int8)
    out[(long_votes >= k) & (short_votes < k)] = 1
    out[(short_votes>= k) & (long_votes < k)] = -1
    return pd.Series(out, index=idx, dtype=np.int8)

def _combine_SEQ(signals: List[pd.Series], window: int, futures: bool) -> pd.Series:
    if not signals:
        return pd.Series(0, index=pd.RangeIndex(0), dtype=np.int8)
    n = len(signals[0])
    idx = signals[0].index
    out = np.zeros(n, dtype=np.int8)

    L = [s.eq(1).astype(np.int8).values for s in signals]
    S = [s.eq(-1).astype(np.int8).values if futures else np.zeros(n, dtype=np.int8) for s in signals]

    def run_chain(B):
        res = np.zeros(n, dtype=np.int8)
        stage = 0
        last_i = -10**9
        K = len(B)
        for i in range(n):
            if stage == 0:
                if B[0][i] == 1:
                    stage = 1; last_i = i
            else:
                if i - last_i > window:
                    stage = 1 if B[0][i]==1 else 0
                    last_i = i if stage==1 else -10**9
                else:
                    need = stage
                    if need < K and B[need][i] == 1:
                        stage += 1; last_i = i
                        if stage == K:
                            res[i] = 1
                            stage = 0
        return res

    long_seq  = run_chain(L)
    short_seq = run_chain(S) if futures else np.zeros(n, dtype=np.int8)

    out[(long_seq==1) & (short_seq==0)]  = 1
    out[(short_seq==1) & (long_seq==0)]  = -1
    return pd.Series(out, index=idx, dtype=np.int8)

# === GATES de qualidade de feature (CVD/Depth) ===
def apply_gate_cvd_only(sig: pd.Series, df: pd.DataFrame, cvd_slope_min: float) -> pd.Series:
    return gate_signal_cvd(sig, df, float(cvd_slope_min))


def apply_gate_imb_only(sig: pd.Series, df: pd.DataFrame, imbalance_min: float) -> pd.Series:
    return gate_signal_imbalance(sig, df, float(imbalance_min))


def apply_feature_gate(
    sig: pd.Series,
    df: pd.DataFrame,
    gate_cfg: GateConfig | float,
    imbalance_min: float | None = None,
    *,
    atr_z_min: float | None = None,
    vhf_min: float | None = None,
) -> pd.Series:
    """
    Compat wrapper que aceita tanto GateConfig quanto os thresholds antigos.
    """
    if isinstance(gate_cfg, GateConfig):
        cfg = gate_cfg
    else:
        cfg = GateConfig(
            cvd_slope_min=float(gate_cfg),
            imbalance_min=float(imbalance_min or 0.0),
            atr_z_min=float(atr_z_min or 0.0),
            vhf_min=float(vhf_min or 0.0),
        )
    return apply_signal_gates(sig, df, cfg).astype(sig.dtype, copy=False)

# =============== Workers base/combos ===============

def _worker_base(method: str, tf: str, A: Dict[str, Any]) -> Tuple[Dict, Optional[pd.DataFrame], Dict]:
    df = TF_FRAMES[tf]

    s_raw = STRATEGY_FUNCS[method](df, {}, bool(A["futures"])).shift(1).fillna(0).astype(np.int8)
    gate_cfg: GateConfig = A.get("gate_config") or GateConfig(
        cvd_slope_min=float(A.get("cvd_slope_min", 0.0)),
        imbalance_min=float(A.get("imbalance_min", 0.0)),
        atr_z_min=float(A.get("atr_z_min", 0.0)),
        vhf_min=float(A.get("vhf_min", 0.0)),
    )

    s_cvd = gate_signal_cvd(s_raw, df, gate_cfg.cvd_slope_min)
    s_imb = gate_signal_imbalance(s_raw, df, gate_cfg.imbalance_min)
    s_atr = gate_signal_atr_zscore(s_raw, df, gate_cfg.atr_z_min)
    s_vhf = gate_signal_vhf(s_raw, df, gate_cfg.vhf_min)
    s_all = apply_signal_gates(s_raw, df, gate_cfg).astype(np.int8)

    diff_masks = {
        "cvd": (s_cvd != s_raw),
        "imb": (s_imb != s_raw),
        "atrz": (s_atr != s_raw),
        "vhf": (s_vhf != s_raw),
    }
    stacked = np.vstack([mask.values for mask in diff_masks.values()])
    multi_mask = (stacked.sum(axis=0) > 1)
    total_mask = (s_all != s_raw)

    gstats = {
        "tf": tf,
        "cvd": int(diff_masks["cvd"].sum()),
        "imb": int(diff_masks["imb"].sum()),
        "atrz": int(diff_masks["atrz"].sum()),
        "vhf": int(diff_masks["vhf"].sum()),
        "both": int(multi_mask.sum()),
        "total": int(total_mask.sum()),
    }

    trades = backtest_from_signals(
        df, s_all,
        hard_stop_usd=float(A["hard_stop_usd_map"].get(tf, A["hard_stop_usd_default"])),
        hard_tp_usd=float(A["hard_tp_usd_map"].get(tf, A["hard_tp_usd_default"])),
        max_hold=int(A["max_hold_map"].get(tf, A["max_hold_default"])),
        fee_perc=float(A["fee_perc"]),
        slippage_ticks=int(A["slippage"]), tick_size=float(A["tick_size"]),
        contracts=float(A["contracts"]), contract_value=float(A["contract_value"]),
        futures=bool(A["futures"]), use_atr_stop=bool(A["use_atr_stop"]),
        atr_stop_len=int(A["atr_stop_len"]), atr_stop_mult=float(A["atr_stop_mult"]),
        use_atr_tp=bool(A["use_atr_tp"]),
        atr_tp_len=int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
        atr_tp_mult=float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
        use_candle_stop=bool(A["use_candle_stop"]),
        candle_stop_lookback=int(A["candle_stop_lookback"]),
        trailing=bool(A["trailing"]),
        timeout_mode=str(A["timeout_mode"]),
        atr_timeout_len=int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
        atr_timeout_mult=float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
        round_to_tick=bool(A.get("round_to_tick", False)),
    )
    m = _metrics_with_fees(trades, float(A["fee_perc"]),
                           contracts=float(A["contracts"]), contract_value=float(A["contract_value"]))
    score = (m["total_pnl"]*1.0) + (m["hit"]*100.0) + (m["payoff"]*10.0) - (m["maxdd"]*0.5)
    row = {
        "method":method,"timeframe":tf, **m, "score":score, "is_combo":False, "combo_spec":"",
        "timeout_mode": str(A["timeout_mode"]),
        "max_hold_used": int(A["max_hold_map"].get(tf, A["max_hold_default"])),
        "atr_timeout_mult_used": float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
        "atr_timeout_len_used": int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
        "use_atr_stop": bool(A["use_atr_stop"]),
        "atr_stop_len_used": int(A["atr_stop_len"]),
        "atr_stop_mult_used": float(A["atr_stop_mult"]),
        "use_atr_tp": bool(A["use_atr_tp"]),
        "atr_tp_len_used": int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
        "atr_tp_mult_used": float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
        "use_candle_stop": bool(A["use_candle_stop"]),
        "candle_stop_lookback": int(A["candle_stop_lookback"]),
    }

    if trades.empty:
        return row, None, gstats
    bt = trades.copy()
    bt["entry_time"] = pd.to_datetime(bt["entry_time"], utc=True)
    bt["exit_time"]  = pd.to_datetime(bt["exit_time"],  utc=True)
    return row, bt, gstats

def _worker_combo(raw: str, tf: str, A: Dict[str, Any], default_window: int, default_maj_votes: int) -> Tuple[Dict, Dict]:
    df = TF_FRAMES[tf]
    try:
        spec = parse_combo_spec(raw, default_window=default_window, default_maj_votes=default_maj_votes)
    except Exception as e:
        return {"_skip": True, "_why": f"parse-fail:{e}"}, {"tf": tf, "cvd":0, "imb":0, "both":0, "total":0}

    gate_cfg: GateConfig = A.get("gate_config") or GateConfig(
        cvd_slope_min=float(A.get("cvd_slope_min", 0.0)),
        imbalance_min=float(A.get("imbalance_min", 0.0)),
        atr_z_min=float(A.get("atr_z_min", 0.0)),
        vhf_min=float(A.get("vhf_min", 0.0)),
    )

    sigs_raw, sigs_cvd, sigs_imb, sigs_atr, sigs_vhf, sigs_all = [], [], [], [], [], []
    for mname in spec.items:
        if mname not in STRATEGY_FUNCS:
            return {"_skip": True, "_why": f"invalid-method:{mname}"}, {"tf": tf, "cvd":0, "imb":0, "atrz":0, "vhf":0, "both":0, "total":0}
        s_raw  = STRATEGY_FUNCS[mname](df, {}, bool(A["futures"])).shift(1).fillna(0).astype(np.int8)
        s_cvd  = gate_signal_cvd(s_raw, df, gate_cfg.cvd_slope_min)
        s_imb  = gate_signal_imbalance(s_raw, df, gate_cfg.imbalance_min)
        s_atr  = gate_signal_atr_zscore(s_raw, df, gate_cfg.atr_z_min)
        s_vhf  = gate_signal_vhf(s_raw, df, gate_cfg.vhf_min)
        s_all  = apply_signal_gates(s_raw, df, gate_cfg).astype(np.int8)
        sigs_raw.append(s_raw)
        sigs_cvd.append(s_cvd)
        sigs_imb.append(s_imb)
        sigs_atr.append(s_atr)
        sigs_vhf.append(s_vhf)
        sigs_all.append(s_all)

    if spec.op == "AND":
        sig_raw   = _combine_AND(sigs_raw,  spec.window, bool(A["futures"]))
        sig_cvd   = _combine_AND(sigs_cvd,  spec.window, bool(A["futures"]))
        sig_imb   = _combine_AND(sigs_imb,  spec.window, bool(A["futures"]))
        sig_atr   = _combine_AND(sigs_atr,  spec.window, bool(A["futures"]))
        sig_vhf   = _combine_AND(sigs_vhf,  spec.window, bool(A["futures"]))
        sig_all   = _combine_AND(sigs_all,  spec.window, bool(A["futures"]))
    elif spec.op == "MAJ":
        sig_raw   = _combine_MAJ(sigs_raw,  spec.k, spec.window, bool(A["futures"]))
        sig_cvd   = _combine_MAJ(sigs_cvd,  spec.k, spec.window, bool(A["futures"]))
        sig_imb   = _combine_MAJ(sigs_imb,  spec.k, spec.window, bool(A["futures"]))
        sig_atr   = _combine_MAJ(sigs_atr,  spec.k, spec.window, bool(A["futures"]))
        sig_vhf   = _combine_MAJ(sigs_vhf,  spec.k, spec.window, bool(A["futures"]))
        sig_all   = _combine_MAJ(sigs_all,  spec.k, spec.window, bool(A["futures"]))
    else:
        sig_raw   = _combine_SEQ(sigs_raw,  spec.window, bool(A["futures"]))
        sig_cvd   = _combine_SEQ(sigs_cvd,  spec.window, bool(A["futures"]))
        sig_imb   = _combine_SEQ(sigs_imb,  spec.window, bool(A["futures"]))
        sig_atr   = _combine_SEQ(sigs_atr,  spec.window, bool(A["futures"]))
        sig_vhf   = _combine_SEQ(sigs_vhf,  spec.window, bool(A["futures"]))
        sig_all   = _combine_SEQ(sigs_all,  spec.window, bool(A["futures"]))

    diff_masks = {
        "cvd": (sig_cvd != sig_raw),
        "imb": (sig_imb != sig_raw),
        "atrz": (sig_atr != sig_raw),
        "vhf": (sig_vhf != sig_raw),
    }
    stacked = np.vstack([mask.values for mask in diff_masks.values()])
    multi_mask = (stacked.sum(axis=0) > 1)
    total_mask = (sig_all != sig_raw)

    gstats = {
        "tf": tf,
        "cvd": int(diff_masks["cvd"].sum()),
        "imb": int(diff_masks["imb"].sum()),
        "atrz": int(diff_masks["atrz"].sum()),
        "vhf": int(diff_masks["vhf"].sum()),
        "both": int(multi_mask.sum()),
        "total": int(total_mask.sum()),
    }

    trades = backtest_from_signals(
        df, sig_all,
        hard_stop_usd=float(A["hard_stop_usd_map"].get(tf, A["hard_stop_usd_default"])),
        hard_tp_usd=float(A["hard_tp_usd_map"].get(tf, A["hard_tp_usd_default"])),
        max_hold=int(A["max_hold_map"].get(tf, A["max_hold_default"])),
        fee_perc=float(A["fee_perc"]),
        slippage_ticks=int(A["slippage"]), tick_size=float(A["tick_size"]),
        contracts=float(A["contracts"]), contract_value=float(A["contract_value"]),
        futures=bool(A["futures"]), use_atr_stop=bool(A["use_atr_stop"]),
        atr_stop_len=int(A["atr_stop_len"]), atr_stop_mult=float(A["atr_stop_mult"]),
        use_atr_tp=bool(A["use_atr_tp"]),
        atr_tp_len=int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
        atr_tp_mult=float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
        use_candle_stop=bool(A["use_candle_stop"]),
        candle_stop_lookback=int(A["candle_stop_lookback"]),
        trailing=bool(A["trailing"]),
        timeout_mode=str(A["timeout_mode"]),
        atr_timeout_len=int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
        atr_timeout_mult=float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
        round_to_tick=bool(A.get("round_to_tick", False)),
    )
    m = _metrics_with_fees(trades, float(A["fee_perc"]),
                           contracts=float(A["contracts"]), contract_value=float(A["contract_value"]))
    score = (m["total_pnl"]*1.0) + (m["hit"]*100.0) + (m["payoff"]*10.0) - (m["maxdd"]*0.5)
    row = {
        "method":f"COMBO:{spec.op}","timeframe":tf, **m, "score":score, "is_combo":True, "combo_spec":spec.raw or raw,
        "timeout_mode": str(A["timeout_mode"]),
        "max_hold_used": int(A["max_hold_map"].get(tf, A["max_hold_default"])),
        "atr_timeout_mult_used": float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
        "atr_timeout_len_used": int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
        "use_atr_stop": bool(A["use_atr_stop"]),
        "atr_stop_len_used": int(A["atr_stop_len"]),
        "atr_stop_mult_used": float(A["atr_stop_mult"]),
        "use_atr_tp": bool(A["use_atr_tp"]),
        "atr_tp_len_used": int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
        "atr_tp_mult_used": float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
        "use_candle_stop": bool(A["use_candle_stop"]),
        "candle_stop_lookback": int(A["candle_stop_lookback"]),
    }

    return row, gstats

# ============================ ML ============================
class _BaseModel:
    def fit(self, X, y): ...
    def predict_proba(self, X): ...

class _XGB(_BaseModel):
    def __init__(self, seed: int = None):
        from xgboost import XGBClassifier
        seed = int(seed if seed is not None else globals().get("ML_SEED", 42))
        ncpu = max(1, (os.cpu_count() or 2) - 1)
        self.m = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
            eval_metric="logloss", n_jobs=ncpu, random_state=seed, reg_lambda=5, reg_alpha=1,
        )
        self.m.set_params(n_jobs=1)
    def fit(self, X, y): self.m.fit(X, y)
    def predict_proba(self, X): return self.m.predict_proba(X)[:,1]


class _SKRF(_BaseModel):
    def __init__(self, seed: int = None):
        from sklearn.ensemble import RandomForestClassifier
        seed = int(seed if seed is not None else globals().get("ML_SEED", 42))
        self.m = RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=1,
            class_weight="balanced_subsample", random_state=seed
        )
    def fit(self, X, y): self.m.fit(X, y)
    def predict_proba(self, X):
        p = self.m.predict_proba(X)
        return p[:,1] if p.shape[1]>1 else p.ravel()

class _NPLR(_BaseModel):
    """ Logistic Regression leve (sem sklearn) """
    def __init__(self, lr: float = 0.2, epochs: int = 200, l2: float = 1e-3, seed: int = None):
        seed = int(seed if seed is not None else globals().get("ML_SEED", 42))
        np.random.seed(seed)
        self.lr = lr; self.epochs = epochs; self.l2 = l2
        self.w = None; self.b = 0.0
    def _sig(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    def fit(self, X, y):
        X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
        n, d = X.shape
        self.w = np.zeros(d, np.float64); self.b = 0.0
        for _ in range(self.epochs):
            p = self._sig(X @ self.w + self.b)
            gw = (X.T @ (p - y)) / max(1, n) + self.l2 * self.w
            gb = (p - y).mean()
            self.w -= self.lr * gw
            self.b -= self.lr * gb
    def predict_proba(self, X):
        X = np.asarray(X, np.float64)
        return self._sig(X @ self.w + self.b)

try:
    from sklearn.preprocessing import StandardScaler as SKStandardScaler
    _SKLEARN_OK = True
except Exception:
    SKStandardScaler = None
    _SKLEARN_OK = False

class _FallbackScaler:
    def fit(self, X):
        X = np.asarray(X, np.float64)
        self.mu = np.nanmean(X, axis=0); self.sd = np.nanstd(X, axis=0) + 1e-9
    def transform(self, X):
        X = np.asarray(X, np.float64)
        return (X - self.mu) / self.sd

def _fit_scaler(X):
    sc = SKStandardScaler() if _SKLEARN_OK else _FallbackScaler()
    sc.fit(X)
    return sc

class _SKLR(_BaseModel):
    def __init__(self, seed: int = None):
        from sklearn.linear_model import LogisticRegression
        seed = int(seed if seed is not None else globals().get("ML_SEED", 42))
        self.m = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1, random_state=seed)
    def fit(self, X, y): self.m.fit(X, y)
    def predict_proba(self, X):
        p = self.m.predict_proba(X)
        return p[:,1] if p.shape[1] > 1 else p.ravel()

class _EnsembleModel(_BaseModel):
    def __init__(self, kind_xgb="xgb", kind_rf="rf", kind_lr="logreg"):
        _, self.xgb = _make_model(kind_xgb)
        _, self.rf  = _make_model(kind_rf)
        _, self.lr  = _make_model(kind_lr)
        self.weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    def fit(self, X, y):
        self.xgb.fit(X, y); self.rf.fit(X, y); self.lr.fit(X, y)
    def predict_proba(self, X):
        px = np.clip(self.xgb.predict_proba(X), 0.0, 1.0)
        pr = np.clip(self.rf.predict_proba(X), 0.0, 1.0)
        pl = np.clip(self.lr.predict_proba(X), 0.0, 1.0)
        idx = pd.RangeIndex(len(px))
        series_list = [
            pd.Series(px, index=idx),
            pd.Series(pr, index=idx),
            pd.Series(pl, index=idx),
        ]
        prob_ma = combine_probs(series_list, "MAJ", k=2)
        prob_and = combine_probs(series_list, "AND")
        weighted = (self.weights[0]*px + self.weights[1]*pr + self.weights[2]*pl)
        stacked = np.vstack([weighted, prob_ma.values, prob_and.values])
        return np.clip(stacked.mean(axis=0), 0.0, 1.0)

def _make_model(kind: str, seed: int = None) -> Tuple[str,_BaseModel]:
    seed = int(seed if seed is not None else globals().get("ML_SEED", 42))
    k = (kind or "auto").lower().strip()
    if k in ("auto","xgb"):
        try: return "xgb", _XGB(seed=seed)
        except Exception:
            if k=="xgb": print("[ML] xgboost indisponível; fallback.")
    if k in ("rf","randomforest","auto"):
        try: return "rf", _SKRF(seed=seed)
        except Exception: print("[ML] RF indisponível; fallback.")
    if k in ("lr","logreg","auto"):
        try: return "logreg", _SKLR(seed=seed)
        except Exception: print("[ML] LogReg indisponível; fallback.")
    return "logreg", _NPLR(seed=seed)


def _wf_month_splits(start, end, train_m, val_m, step_m):
    # converte meses fracionários em dias (~30d = 1m)
    train_days = int(round(train_m * 30))
    val_days = int(round(val_m * 30))
    step_days = int(round(step_m * 30))

    cur_start = pd.Timestamp(start)
    final_end = pd.Timestamp(end)
    windows = []

    while True:
        tr_start = cur_start
        tr_end = tr_start + pd.Timedelta(days=train_days) - pd.Timedelta(microseconds=1)
        va_start = tr_end + pd.Timedelta(microseconds=1)
        va_end = va_start + pd.Timedelta(days=val_days) - pd.Timedelta(microseconds=1)
        if va_end > final_end:
            break
        windows.append((tr_start, tr_end, va_start, va_end))
        cur_start = cur_start + pd.Timedelta(days=step_days)
    return windows


def _build_base_signal_feats(df: pd.DataFrame, methods: List[str], futures: bool) -> pd.DataFrame:
    feats = {}
    for m in methods:
        if m in STRATEGY_FUNCS:
            feats[m] = STRATEGY_FUNCS[m](df, {}, futures).shift(1).fillna(0).astype(np.int8)
    return pd.DataFrame(feats, index=df.index) if feats else pd.DataFrame(index=df.index)


def _make_ml_features(df: pd.DataFrame, *,
                      add_lags: int = 3,
                      include_agg: bool = True,
                      include_depth: bool = True,
                      base_signals_df: Optional[pd.DataFrame] = None,
                      combo_signals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Cria o vetor de features X para ML.
    - OHLCV/indicadores + Agg/Depth (opcionais)
    - Sinais base como features (sig_*)
    - Sinais de combos top-N como features (sigc_*)
    """
    x = pd.DataFrame(index=df.index).copy()
    close = df["close"].astype(float)

    # === Preço / Retornos ===
    x["ret_1"]   = close.pct_change(1)
    x["ret_5"]   = close.pct_change(5)
    x["ret_20"]  = close.pct_change(20)
    x["ret_vol_20"] = x["ret_1"].rolling(20, min_periods=5).std()

    # ATR relativo
    _a14 = atr(df, 14)
    x["atr_pct"] = (_a14 / close.abs().clip(lower=1e-9))

    # === Tendência clássica ===
    ef, es = ema(close, 9), ema(close, 21)
    x["ema_f"] = ef
    x["ema_s"] = es
    x["rsi14"] = rsi(close, 14)
    mac = macd_line(close, 12, 26)
    x["macd"] = mac
    x["macd_sig"] = ema(mac, 9)
    m, u, l = bollinger_bands(close, 20, 2.0)
    x["bb_pos"] = (close - m) / (u - l + 1e-9)
    vwap20 = vwap_win(df, 20)
    x["vwap20"] = vwap20
    x["dist_ema_f"] = (close - ef) / (ef + 1e-9)
    x["dist_vwap20"] = (close - vwap20) / (vwap20 + 1e-9)

    # AggTrades
    if include_agg:
        x["cvd"] = df.get("cvd", pd.Series(0.0, index=df.index)).fillna(0.0)
        x["cvd_slope"] = x["cvd"].diff(3)
        x["taker_buy_vol"]  = df.get("taker_buy_vol",  0.0)
        x["taker_sell_vol"] = df.get("taker_sell_vol", 0.0)
        x["vol_per_bar"]    = df.get("vol_per_bar",    0.0)

    # Depth
    if include_depth:
        x["imb_net_depth"] = df.get("imb_net_depth", 0.0)

    # ===== Extras (se existirem no DF) =====
    if "funding_rate" in df.columns:
        x["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce").fillna(0.0)
        x["funding_chg"]  = x["funding_rate"].diff().fillna(0.0)
    if "mark_price" in df.columns:
        mp = pd.to_numeric(df["mark_price"], errors="coerce")
        x["mark_close_ratio"] = (mp / (close.replace(0, np.nan))) - 1.0
    if "premium_index" in df.columns:
        pr = pd.to_numeric(df["premium_index"], errors="coerce")
        x["premium_z"]  = (pr - pr.rolling(96, min_periods=10).mean()) / (pr.rolling(96, min_periods=10).std() + 1e-9)
        x["prem_abs"]   = pr.fillna(0.0)
    if "bbo_spread" in df.columns:
        x["bbo_spread"]    = pd.to_numeric(df["bbo_spread"], errors="coerce")
        x["bbo_mid"]       = pd.to_numeric(df["bbo_mid"],    errors="coerce")
        x["microprice_imb"]= pd.to_numeric(df["microprice_imb"], errors="coerce")
        x["bbo_imb"]       = pd.to_numeric(df["bbo_imb"],    errors="coerce")

    # Sinais base (se ativado)
    if base_signals_df is not None and not base_signals_df.empty:
        for c in base_signals_df.columns:
            x[f"sig_{c}"] = base_signals_df[c].shift(1)

    # Sinais de combos (NOVO)
    if combo_signals_df is not None and not combo_signals_df.empty:
        for c in combo_signals_df.columns:
            x[f"sigc_{c}"] = combo_signals_df[c].shift(1)

    # Lags
    for L in range(1, add_lags+1):
        if "funding_rate" in x:
            x[f"funding_lag{L}"] = x["funding_rate"].shift(L)
        if "bbo_spread" in x:
            x[f"bbo_spread_lag{L}"] = x["bbo_spread"].shift(L)
            x[f"microprice_lag{L}"] = x["microprice_imb"].shift(L)

        for col in ["ret_1", "rsi14", "bb_pos", "dist_ema_f", "dist_vwap20"]:
            if col in x:
                x[f"{col}_lag{L}"] = x[col].shift(L)
        if include_agg:
            x[f"cvd_lag{L}"] = x["cvd"].shift(L)
        if include_depth and "imb_net_depth" in x:
            x[f"imb_lag{L}"] = x["imb_net_depth"].shift(L)

    # Clean
    x["time"] = pd.to_datetime(df["time"], utc=True)
    x = x.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return _downcast_inplace(x)


def _make_ml_labels(close: pd.Series, atr_series: Optional[pd.Series] = None,
                    k_mult: float = 0.8, horizon: int = 5) -> pd.Series:
    """
    Gera labels binários adaptativos: compara retorno futuro com ATR% local.
    horizon: nº de candles à frente.
    k_mult: multiplicador do ATR relativo (ex: 0.8 → exige ±0.8 * ATR como movimento)
    """
    fut = close.shift(-horizon)
    ret = (fut / close - 1.0)

    if atr_series is not None:
        atr_rel = (atr_series / close).clip(lower=1e-6)
        ret_thr_up = +k_mult * atr_rel
        ret_thr_dn = -k_mult * atr_rel
    else:
        ret_thr_up = pd.Series(0.0015, index=close.index)
        ret_thr_dn = -ret_thr_up

    y = pd.Series(np.nan, index=close.index)
    y[ret > ret_thr_up] = 1.0
    y[ret < ret_thr_dn] = 0.0
    return y

def _make_model_and_fit(
    kind: str,
    X_tr_df: pd.DataFrame,
    y_tr: Union[pd.Series, np.ndarray],
    X_va_df: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
) -> Tuple[str, _BaseModel, Any, np.ndarray, Optional[np.ndarray]]:
    """
    Cria scaler, transforma X_tr/X_va, instancia e treina o modelo.
    Retorna: (model_key, model, scaler, Xtr, Xva)
    - kind: "ensemble" | "xgb" | "rf" | "logreg" | "auto"
    - seed: se None, usa ML_SEED global (se existir), senão 42.
    """
    # -------- seed --------
    if seed is None:
        seed = int(globals().get("ML_SEED", 42))

    # -------- features/escala --------
    feat_cols = [c for c in X_tr_df.columns if c != "time"]
    sc = _fit_scaler(X_tr_df[feat_cols].values)
    Xtr = sc.transform(X_tr_df[feat_cols].values)
    Xva = sc.transform(X_va_df[feat_cols].values) if X_va_df is not None else None

    # -------- modelo --------
    k = (kind or "auto").lower().strip()
    if k == "ensemble":
        mk = "ensemble"
        # _EnsembleModel cria os submodelos via _make_model, que já usa ML_SEED nos __init__
        model = _EnsembleModel()
    else:
        mk, model = _make_model(k, seed=seed)

    # garante y como array 1D
    y_arr = y_tr.values if isinstance(y_tr, pd.Series) else np.asarray(y_tr)
    model.fit(Xtr, y_arr)

    return mk, model, sc, Xtr, Xva


# ---------- utils ----------
def _slice_df_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["time"]>=start) & (df["time"]<=end)].reset_index(drop=True)

def _append_csv(path: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", header=header, index=False)

# ===================== WFO: param grid e execuções =====================
def _grid_for_method(method: str, mode: str = "light") -> List[dict]:
    mode = (mode or "light").lower().strip()
    L = []
    if method == "trend_breakout":
        ef = [9,12]; es=[26,50]; don=[20,40] if mode!="light" else [20]
        for a,b,c in product(ef, es, don):
            if a < b: L.append({"ema_fast":a,"ema_slow":b,"donchian":c})
    elif method == "keltner_breakout":
        m=[1.5,2.0] if mode!="light" else [1.5]
        for mult in m: L.append({"ema_len":20,"atr_len":20,"mult":mult})
    elif method == "rsi_reversion":
        rs=[25,30,35] if mode!="light" else [30]
        rl=[14,21] if mode!="light" else [14]
        rl_long=[65,70,75] if mode!="light" else [70]
        for a,b,c in product(rl, rs, rl_long):
            L.append({"rsi_len":a,"rsi_short":b,"rsi_long":c})
    elif method == "ema_crossover":
        ef=[9,12]; es=[26,50]
        for a,b in product(ef,es):
            if a<b: L.append({"ema_fast":a,"ema_slow":b})
    elif method == "macd_trend":
        L.append({"macd_fast":12,"macd_slow":26,"macd_signal":9})
    elif method == "vwap_trend":
        L.append({"len":20}); 
        if mode!="light": L.append({"len":40})
    elif method == "boll_breakout":
        ks = [2.0, 2.5] if mode!="light" else [2.0]
        for k in ks: L.append({"bb_len":20,"bb_k":k})
    elif method == "orb_breakout":
        for im in [3,5,7]:
            L.append({"init_minutes":im,"vol_q":0.6})
    elif method == "orr_reversal":
        for im in [3,5,7]:
            L.append({"init_minutes":im})
    elif method == "ema_pullback":
        L.append({"ema_fast":9,"ema_slow":20})
    elif method == "donchian_breakout":
        L += [{"n":20},{"n":55}]
    elif method == "vwap_poc_reject":
        L.append({"anchor":"D","wick_frac":0.5,"vol_q":0.6})
    elif method == "ob_imbalance_break":
        vals = [1.5] if mode=="light" else [1.0,1.5,2.0]
        for k in vals:
            L.append({"imb_thr_mult":k})
    elif method == "cvd_divergence_reversal":
        vals = [20] if mode=="light" else [10,20]
        for n in vals:
            L.append({"div_lookback":n})
    else:
        L.append({})
    return L
# ========================================
# TESTE DE SANIDADE — BUY & HOLD
# ========================================

def make_buy_hold(df):
    """
    Método de teste para verificar o pipeline.
    Retorna +1 em todas as barras (comprado sempre).
    """
    import numpy as np
    sig = np.ones(len(df))
    return sig

def _eval_method_on_range(tf: str, method: str, params: dict, A: Dict[str, Any],
                          start: pd.Timestamp, end: pd.Timestamp,
                          lookback_bars: Optional[int] = None) -> Tuple[Dict, Optional[pd.DataFrame]]:
    df_all = TF_FRAMES[tf]
    df = _slice_df_time(df_all, start, end)
    if lookback_bars and len(df)>lookback_bars:
        df = df.iloc[-lookback_bars:].reset_index(drop=True)
    if df.empty:
        return {"method":method,"timeframe":tf,"n_trades":0,"total_pnl":0.0,"hit":0.0,
                "sharpe":0.0,"expectancy":0.0,"maxdd":0.0,"payoff":0.0,"score":-1e18,
                "params":json.dumps(params)}, None
    gate_cfg: GateConfig = A.get("gate_config") or GateConfig(
        cvd_slope_min=float(A.get("cvd_slope_min", 0.0)),
        imbalance_min=float(A.get("imbalance_min", 0.0)),
        atr_z_min=float(A.get("atr_z_min", 0.0)),
        vhf_min=float(A.get("vhf_min", 0.0)),
    )
    s = STRATEGY_FUNCS[method](df, params or {}, bool(A["futures"])).shift(1).fillna(0).astype(np.int8)
    s = apply_signal_gates(s, df, gate_cfg).astype(np.int8)
    trades = backtest_from_signals(
        df, s,
        hard_stop_usd=float(A["hard_stop_usd_map"].get(tf, A["hard_stop_usd_default"])),
        hard_tp_usd=float(A["hard_tp_usd_map"].get(tf, A["hard_tp_usd_default"])),
        max_hold=int(A["max_hold_map"].get(tf, A["max_hold_default"])),
        fee_perc=float(A["fee_perc"]),
        slippage_ticks=int(A["slippage"]), tick_size=float(A["tick_size"]),
        contracts=float(A["contracts"]), contract_value=float(A["contract_value"]),
        futures=bool(A["futures"]), use_atr_stop=bool(A["use_atr_stop"]),
        atr_stop_len=int(A["atr_stop_len"]), atr_stop_mult=float(A["atr_stop_mult"]),
        use_atr_tp=bool(A["use_atr_tp"]),
        atr_tp_len=int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
        atr_tp_mult=float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
        use_candle_stop=bool(A["use_candle_stop"]),
        candle_stop_lookback=int(A["candle_stop_lookback"]),
        trailing=bool(A["trailing"]),
        timeout_mode=str(A["timeout_mode"]),
        atr_timeout_len=int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
        atr_timeout_mult=float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
        round_to_tick=bool(A.get("round_to_tick", False)),
    )
    m = _metrics_with_fees(trades, float(A["fee_perc"]),
                           contracts=float(A["contracts"]), contract_value=float(A["contract_value"]))
    score = (m["total_pnl"]*1.0) + (m["hit"]*100.0) + (m["payoff"]*10.0) - (m["maxdd"]*0.5)
    row = {"method":method,"timeframe":tf, **m, "score":score, "params":json.dumps(params),
           "timeout_mode": str(A["timeout_mode"]),
           "max_hold_used": int(A["max_hold_map"].get(tf, A["max_hold_default"])),
           "atr_timeout_mult_used": float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
           "atr_timeout_len_used": int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
           "use_atr_stop": bool(A["use_atr_stop"]),
           "atr_stop_len_used": int(A["atr_stop_len"]),
           "atr_stop_mult_used": float(A["atr_stop_mult"]),
           "use_atr_tp": bool(A["use_atr_tp"]),
           "atr_tp_len_used": int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
           "atr_tp_mult_used": float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
           "use_candle_stop": bool(A["use_candle_stop"]),
           "candle_stop_lookback": int(A["candle_stop_lookback"])}
    return row, trades if not trades.empty else None

def _best_params_for_tf(tf: str, methods: List[str], A: Dict[str,Any],
                        tr_s: pd.Timestamp, tr_e: pd.Timestamp, grid_mode: str,
                        n_jobs: int = 1, top_n: int = 5,
                        par_backend: str = "none",
                        lookback_bars: Optional[int] = None) -> Dict[Tuple[str,str], Dict]:
    tasks = []
    for m in methods:
        if m not in STRATEGY_FUNCS:
            continue
        for p in _grid_for_method(m, grid_mode):
            tasks.append((m, p))
    rows=[]
    max_workers = n_jobs if n_jobs>0 else (_os.cpu_count() or 2)
    exec_tasks = [(tf, m, p, A, tr_s, tr_e, lookback_bars) for (m,p) in tasks]
    for success, payload in iter_task_results(exec_tasks, _eval_method_on_range, backend=par_backend, max_workers=max_workers):
        if success:
            row, _ = payload
            rows.append(row)
        else:
            print(f"[WFO] train worker fail: {payload}")

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["method","timeframe","score","params"])
    best: Dict[Tuple[str,str], Dict] = {}
    if not df.empty:
        df = df.sort_values("score", ascending=False)
        for m in methods:
            best_rows = df[df["method"]==m].head(top_n)
            if not best_rows.empty:
                best[(tf,m)] = {
                    "best": json.loads(best_rows.iloc[0]["params"]),
                    "top_list": [(json.loads(r["params"]), float(r["score"])) for _,r in best_rows.iterrows()]
                }
    return best

def _signals_with_params(df: pd.DataFrame, method: str, params: dict, A: Dict[str,Any]) -> pd.Series:
    gate_cfg: GateConfig = A.get("gate_config") or GateConfig(
        cvd_slope_min=float(A.get("cvd_slope_min", 0.0)),
        imbalance_min=float(A.get("imbalance_min", 0.0)),
        atr_z_min=float(A.get("atr_z_min", 0.0)),
        vhf_min=float(A.get("vhf_min", 0.0)),
    )
    s = STRATEGY_FUNCS[method](df, params or {}, bool(A["futures"])).shift(1).fillna(0).astype(np.int8)
    return apply_signal_gates(s, df, gate_cfg).astype(np.int8)

def run_wfo_base_combos(
    req_tfs: List[str], methods: List[str], A: Dict[str,Any],
    start: pd.Timestamp, end: pd.Timestamp,
    train_m: int, val_m: int, step_m: int,
    combo_ops: List[str], combo_cap: int, combo_window: int, combo_min_votes: int,
    grid_mode: str = "light", n_jobs: int = 1, top_n: int = 5,
    par_backend: str = "none",
    out_wf_base: Optional[str] = None, out_wf_combos: Optional[str] = None, out_wf_all: Optional[str] = None,
    out_wf_trades: Optional[str] = None,
    lookback_bars_train: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    gate_stats_base: Dict[str, Dict[str,int]] = {}
    gate_stats_combo: Dict[str, Dict[str,int]] = {}
    base_rows=[]; combo_rows=[]; all_rows=[]
    oos_trades=[]

    def _acc_gate(acc: Dict[str, Dict[str,int]], tf: str, gs: Dict[str,int]):
        cur = acc.get(tf, {})
        for k, v in gs.items():
            if k == "tf":
                continue
            cur[k] = cur.get(k, 0) + int(v)
        acc[tf] = cur

    windows = _wf_month_splits(start, end, train_m, val_m, step_m)
    if not windows:
        print("[WFO] Sem janelas. Verifique datas/meses.")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), gate_stats_base, gate_stats_combo)

    for (tr_s, tr_e, va_s, va_e) in windows:
        win_id = f"{tr_s.strftime('%Y-%m')}→{va_e.strftime('%Y-%m')}"

        # 1) selecionar melhores params por TF/método no treino
        best_by_tf: Dict[Tuple[str,str], Dict] = {}
        for tf in req_tfs:
            if tf not in TF_FRAMES: 
                continue
            best = _best_params_for_tf(tf, methods, A, tr_s, tr_e, grid_mode, n_jobs=n_jobs, top_n=top_n,
                                       par_backend=par_backend, lookback_bars=lookback_bars_train)
            best_by_tf.update(best)

        # 2) rodar OOS para base
        for tf in req_tfs:
            if tf not in TF_FRAMES:
                continue
            df = _slice_df_time(TF_FRAMES[tf], va_s, va_e)
            if df.empty: 
                continue
            for m in methods:
                if (tf,m) not in best_by_tf:
                    continue
                params = best_by_tf[(tf,m)]["best"]
                gate_cfg: GateConfig = A.get("gate_config") or GateConfig(
                    cvd_slope_min=float(A.get("cvd_slope_min", 0.0)),
                    imbalance_min=float(A.get("imbalance_min", 0.0)),
                    atr_z_min=float(A.get("atr_z_min", 0.0)),
                    vhf_min=float(A.get("vhf_min", 0.0)),
                )
                s_raw = STRATEGY_FUNCS[m](df, params or {}, bool(A["futures"])).shift(1).fillna(0).astype(np.int8)
                s_cvd = gate_signal_cvd(s_raw, df, gate_cfg.cvd_slope_min)
                s_imb = gate_signal_imbalance(s_raw, df, gate_cfg.imbalance_min)
                s_atr = gate_signal_atr_zscore(s_raw, df, gate_cfg.atr_z_min)
                s_vhf = gate_signal_vhf(s_raw, df, gate_cfg.vhf_min)
                s_all = apply_signal_gates(s_raw, df, gate_cfg).astype(np.int8)

                diff_masks = {
                    "cvd": (s_cvd != s_raw),
                    "imb": (s_imb != s_raw),
                    "atrz": (s_atr != s_raw),
                    "vhf": (s_vhf != s_raw),
                }
                stacked = np.vstack([mask.values for mask in diff_masks.values()])
                gstats = {
                    "tf": tf,
                    "cvd": int(diff_masks["cvd"].sum()),
                    "imb": int(diff_masks["imb"].sum()),
                    "atrz": int(diff_masks["atrz"].sum()),
                    "vhf": int(diff_masks["vhf"].sum()),
                    "both": int((stacked.sum(axis=0) > 1).sum()),
                    "total": int((s_all != s_raw).sum()),
                }
                _acc_gate(gate_stats_base, tf, gstats)

                trades = backtest_from_signals(
                    df, s_all,
                    hard_stop_usd=float(A["hard_stop_usd_map"].get(tf, A["hard_stop_usd_default"])),
                    hard_tp_usd=float(A["hard_tp_usd_map"].get(tf, A["hard_tp_usd_default"])),
                    max_hold=int(A["max_hold_map"].get(tf, A["max_hold_default"])),
                    fee_perc=float(A["fee_perc"]),
                    slippage_ticks=int(A["slippage"]), tick_size=float(A["tick_size"]),
                    contracts=float(A["contracts"]), contract_value=float(A["contract_value"]),
                    futures=bool(A["futures"]), use_atr_stop=bool(A["use_atr_stop"]),
                    atr_stop_len=int(A["atr_stop_len"]), atr_stop_mult=float(A["atr_stop_mult"]),
                    use_atr_tp=bool(A["use_atr_tp"]),
                    atr_tp_len=int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
                    atr_tp_mult=float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
                    use_candle_stop=bool(A["use_candle_stop"]),
                    candle_stop_lookback=int(A["candle_stop_lookback"]),
                    trailing=bool(A["trailing"]),
                    timeout_mode=str(A["timeout_mode"]),
                    atr_timeout_len=int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
                    atr_timeout_mult=float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
                    round_to_tick=bool(A.get("round_to_tick", False)),
                )
                mtr = _metrics_with_fees(trades, float(A["fee_perc"]),
                                         contracts=float(A["contracts"]), contract_value=float(A["contract_value"]))
                score = (mtr["total_pnl"]*1.0) + (mtr["hit"]*100.0) + (mtr["payoff"]*10.0) - (mtr["maxdd"]*0.5)
                row = {
                    "wf_window":win_id,"method":m,"timeframe":tf, **mtr, "score":score, "is_combo":False, "combo_spec":"",
                    "params":json.dumps(params),
                    "timeout_mode": str(A["timeout_mode"]),
                    "max_hold_used": int(A["max_hold_map"].get(tf, A["max_hold_default"])),
                    "atr_timeout_mult_used": float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
                    "atr_timeout_len_used": int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
                    "use_atr_stop": bool(A["use_atr_stop"]),
                    "atr_stop_len_used": int(A["atr_stop_len"]),
                    "atr_stop_mult_used": float(A["atr_stop_mult"]),
                    "use_atr_tp": bool(A["use_atr_tp"]),
                    "atr_tp_len_used": int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
                    "atr_tp_mult_used": float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
                    "use_candle_stop": bool(A["use_candle_stop"]),
                    "candle_stop_lookback": int(A["candle_stop_lookback"]),
                }
                base_rows.append(row); all_rows.append(row)
                if not trades.empty:
                    tmp = trades.copy()
                    tmp["entry_time"] = pd.to_datetime(tmp["entry_time"], utc=True)
                    tmp["exit_time"]  = pd.to_datetime(tmp["exit_time"],  utc=True)
                    tmp["method"] = m; tmp["timeframe"] = tf; tmp["wf_window"] = win_id
                    if out_wf_trades:
                        _append_csv(out_wf_trades, tmp)
                    else:
                        oos_trades.append(tmp)

        # 3) combos OOS
        base_methods_tf = {}
        for (tf,m),info in best_by_tf.items():
            base_methods_tf.setdefault(tf, []).append((m, info["top_list"][0][1]))  # (method, score_tr)
        for tf in req_tfs:
            if tf not in TF_FRAMES or tf not in base_methods_tf:
                continue
            base_sorted = sorted(base_methods_tf[tf], key=lambda x: x[1], reverse=True)[:min(top_n,5)]
            candidates = [m for (m,_) in base_sorted if m in STRATEGY_FUNCS]
            if len(candidates) < 2:
                continue
            combos_list=[]
            if "AND" in combo_ops:
                combos_list += [f"AND({a},{b})" for (a,b) in combinations(candidates,2)]
                combos_list += [f"AND({a},{b},{c})" for (a,b,c) in combinations(candidates,3)]
            if "MAJ" in combo_ops:
                combos_list += [f"MAJ2({a},{b})" for (a,b) in combinations(candidates,2)]
                combos_list += [f"MAJ2({a},{b},{c})" for (a,b,c) in combinations(candidates,3)]
            if "SEQ" in combo_ops:
                combos_list += [f"SEQ({a}->{b})" for (a,b) in combinations(candidates,2)]
            combos_list = list(dict.fromkeys(combos_list))[:combo_cap]

            df = _slice_df_time(TF_FRAMES[tf], va_s, va_e)
            if df.empty:
                continue

            gate_cfg: GateConfig = A.get("gate_config") or GateConfig(
                cvd_slope_min=float(A.get("cvd_slope_min", 0.0)),
                imbalance_min=float(A.get("imbalance_min", 0.0)),
                atr_z_min=float(A.get("atr_z_min", 0.0)),
                vhf_min=float(A.get("vhf_min", 0.0)),
            )

            for raw in combos_list:
                try:
                    spec = parse_combo_spec(raw, default_window=combo_window, default_maj_votes=combo_min_votes)
                except Exception:
                    continue
                sigs=[]
                for m in spec.items:
                    params = (best_by_tf.get((tf,m)) or {}).get("best", {})
                    s = STRATEGY_FUNCS[m](df, params or {}, bool(A["futures"])).shift(1).fillna(0).astype(np.int8)
                    s = apply_signal_gates(s, df, gate_cfg).astype(np.int8)
                    sigs.append(s)
                if not sigs:
                    continue
                if spec.op == "AND":   sig = _combine_AND(sigs, spec.window, bool(A["futures"]))
                elif spec.op == "MAJ": sig = _combine_MAJ(sigs, spec.k, spec.window, bool(A["futures"]))
                else:                  sig = _combine_SEQ(sigs, spec.window, bool(A["futures"]))

                trades = backtest_from_signals(
                    df, sig,
                    hard_stop_usd=float(A["hard_stop_usd_map"].get(tf, A["hard_stop_usd_default"])),
                    hard_tp_usd=float(A["hard_tp_usd_map"].get(tf, A["hard_tp_usd_default"])),
                    max_hold=int(A["max_hold_map"].get(tf, A["max_hold_default"])),
                    fee_perc=float(A["fee_perc"]),
                    slippage_ticks=int(A["slippage"]), tick_size=float(A["tick_size"]),
                    contracts=float(A["contracts"]), contract_value=float(A["contract_value"]),
                    futures=bool(A["futures"]), use_atr_stop=bool(A["use_atr_stop"]),
                    atr_stop_len=int(A["atr_stop_len"]), atr_stop_mult=float(A["atr_stop_mult"]),
                    use_atr_tp=bool(A["use_atr_tp"]),
                    atr_tp_len=int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
                    atr_tp_mult=float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
                    use_candle_stop=bool(A["use_candle_stop"]),
                    candle_stop_lookback=int(A["candle_stop_lookback"]),
                    trailing=bool(A["trailing"]),
                    timeout_mode=str(A["timeout_mode"]),
                    atr_timeout_len=int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
                    atr_timeout_mult=float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
                    round_to_tick=bool(A.get("round_to_tick", False)),
                )
                mtr = _metrics_with_fees(trades, float(A["fee_perc"]),
                                         contracts=float(A["contracts"]), contract_value=float(A["contract_value"]))
                score = (mtr["total_pnl"]*1.0) + (mtr["hit"]*100.0) + (mtr["payoff"]*10.0) - (mtr["maxdd"]*0.5)
                row = {
                    "wf_window":win_id,"method":f"COMBO:{spec.op}","timeframe":tf, **mtr, "score":score, "is_combo":True, "combo_spec":spec.raw or raw,
                    "timeout_mode": str(A["timeout_mode"]),
                    "max_hold_used": int(A["max_hold_map"].get(tf, A["max_hold_default"])),
                    "atr_timeout_mult_used": float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
                    "atr_timeout_len_used": int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
                    "use_atr_stop": bool(A["use_atr_stop"]),
                    "atr_stop_len_used": int(A["atr_stop_len"]),
                    "atr_stop_mult_used": float(A["atr_stop_mult"]),
                    "use_atr_tp": bool(A["use_atr_tp"]),
                    "atr_tp_len_used": int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
                    "atr_tp_mult_used": float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
                    "use_candle_stop": bool(A["use_candle_stop"]),
                    "candle_stop_lookback": int(A["candle_stop_lookback"]),
                }
                combo_rows.append(row); all_rows.append(row)

        # streaming parcial
        if out_wf_base and base_rows:
            _append_csv(out_wf_base, pd.DataFrame(base_rows).sort_values(["wf_window","score"], ascending=[True,False]))
            base_rows.clear()
        if out_wf_combos and combo_rows:
            _append_csv(out_wf_combos, pd.DataFrame(combo_rows).sort_values(["wf_window","score"], ascending=[True,False]))
            combo_rows.clear()
        if out_wf_all and all_rows:
            _append_csv(out_wf_all, pd.DataFrame(all_rows).sort_values(["wf_window","score"], ascending=[True,False]))
            all_rows.clear()

    base_df  = pd.DataFrame(base_rows)
    combo_df = pd.DataFrame(combo_rows)
    all_df   = pd.DataFrame(all_rows)
    trades_oos_df = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()

    return base_df, combo_df, all_df, trades_oos_df, gate_stats_base, gate_stats_combo

# =============================== RUNTIME CONFIG (consolidador) ===============================

def _parse_params_cell(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip():
        try: return json.loads(x)
        except Exception: return {}
    return {}

def _aggregate_best_params_from_csv(csv_path: str, by_cols=("timeframe","method"), score_col="score") -> Dict[Tuple[str,str], dict]:
    out: Dict[Tuple[str,str], dict] = {}
    if not csv_path or not os.path.exists(csv_path) or os.path.getsize(csv_path)==0:
        return out
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out
    if df.empty or "params" not in df.columns:
        return out
    df["_params"] = df["params"].apply(_parse_params_cell)
    def _js(x): 
        try: return json.dumps(x, sort_keys=True)
        except Exception: return "{}"
    df["_params_key"] = df["_params"].apply(_js)
    g = df.groupby(list(by_cols)+["_params_key"], as_index=False)[score_col].sum()
    for (tf, method), gdf in g.groupby(list(by_cols)):
        gdf = gdf.sort_values(score_col, ascending=False)
        key = gdf.iloc[0]["_params_key"]
        try:
            out[(tf, method)] = json.loads(key)
        except Exception:
            out[(tf, method)] = {}
    return out

def _select_top_by_tf(csv_path: str, is_combo: bool, top_k: int = 3, score_col="score") -> Dict[str, List[dict]]:
    result: Dict[str, List[dict]] = {}
    if not csv_path or not os.path.exists(csv_path) or os.path.getsize(csv_path)==0:
        return result
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return result
    if df.empty:
        return result
    if is_combo and "combo_spec" not in df.columns:
        return result
    for tf, g in df.groupby("timeframe"):
        g2 = g.sort_values(score_col, ascending=False).head(top_k)
        rows = []
        for _, r in g2.iterrows():
            if is_combo:
                rows.append({
                    "combo_spec": str(r.get("combo_spec","")),
                    "score": float(r.get(score_col, 0.0))
                })
            else:
                rows.append({
                    "method": str(r.get("method","")),
                    "score": float(r.get(score_col, 0.0)),
                    "params": _parse_params_cell(r.get("params","{}"))
                })
        result[tf] = rows
    return result

def _expand_list_per_tf(raw: str, req_tfs: List[str], cast: Any, default: Any) -> Dict[str, Any]:
    vals: List[str]
    if raw is None:
        vals = []
    else:
        raw = str(raw).strip()
        if raw == "":
            vals = []
        else:
            vals = [x.strip() for x in raw.split(",")]
    if not vals:
        return {tf: cast(default) for tf in req_tfs}
    if len(vals) == 1:
        v = cast(vals[0])
        return {tf: v for tf in req_tfs}
    res = {}
    for i, tf in enumerate(req_tfs):
        if i < len(vals):
            try: res[tf] = cast(vals[i])
            except Exception: res[tf] = cast(default)
        else:
            res[tf] = cast(vals[-1])
    return res

def build_runtime_config(
    args, req_tfs: List[str], start_clamped: pd.Timestamp, end_clamped: pd.Timestamp,
    mode: str
) -> dict:
    if mode == "wfo":
        path_base   = args.out_wf_base
        path_combos = args.out_wf_combos
        path_all    = args.out_wf_all
    else:
        path_base   = args.out_leaderboard_base
        path_combos = args.out_leaderboard_combos
        path_all    = args.out_leaderboard_all

    best_params_map = _aggregate_best_params_from_csv(path_base, by_cols=("timeframe","method"), score_col="score")
    top_base_by_tf   = _select_top_by_tf(path_base, is_combo=False, top_k=args.runtime_top_k_base, score_col="score")
    top_combos_by_tf = _select_top_by_tf(path_combos, is_combo=True,  top_k=args.runtime_top_k_combos, score_col="score")

    base_methods: Dict[str, List[dict]] = {}
    for tf in req_tfs:
        lst = []
        for row in top_base_by_tf.get(tf, []):
            m = row["method"]
            params = row.get("params") or best_params_map.get((tf,m), {})
            lst.append({"name": m, "params": params})
        base_methods[tf] = lst

    combos_struct: Dict[str, List[dict]] = {}
    for tf in req_tfs:
        lst = []
        for row in top_combos_by_tf.get(tf, []):
            spec_s = row["combo_spec"]
            try:
                c = parse_combo_spec(spec_s, default_window=args.combo_window, default_maj_votes=args.combo_min_votes)
                st = {
                    "name": spec_s, "type": c.op, "window": c.window, "k": (c.k if c.op=="MAJ" else None),
                    "members": c.items, "params": {}
                }
                for m in c.items:
                    st["params"][m] = best_params_map.get((tf, m), {})
            except Exception:
                st = {"name": spec_s, "type": "UNKNOWN", "window": None, "k": None, "members": [], "params": {}}
            lst.append(st)
        combos_struct[tf] = lst

    max_hold_map        = _expand_list_per_tf(args.max_hold, req_tfs, int, args.max_hold_default)
    atr_timeout_len_map = _expand_list_per_tf(args.atr_timeout_len, req_tfs, int, args.atr_timeout_len_default)
    atr_timeout_mult_map= _expand_list_per_tf(args.atr_timeout_mult, req_tfs, float, args.atr_timeout_mult_default)

    atr_tp_len_map  = _expand_list_per_tf(args.atr_tp_len, req_tfs, int, args.atr_tp_len_default)
    atr_tp_mult_map = _expand_list_per_tf(args.atr_tp_mult, req_tfs, float, args.atr_tp_mult_default)
    args.hard_stop_usd_default = float(str(args.hard_stop_usd).split(",")[0]) if args.hard_stop_usd else 0.0
    args.hard_tp_usd_default   = float(str(args.hard_tp_usd).split(",")[0])   if args.hard_tp_usd   else 0.0
    hard_stop_usd_map = _expand_list_per_tf(args.hard_stop_usd, req_tfs, float, args.hard_stop_usd_default)
    hard_tp_usd_map   = _expand_list_per_tf(args.hard_tp_usd,   req_tfs, float, args.hard_tp_usd_default)

    runtime = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol": args.symbol,
            "range_used": {"start": str(start_clamped), "end": str(end_clamped)},
            "mode": mode,
            "files": {
                "base": path_base,
                "combos": path_combos,
                "all": path_all
            }
        },
        "timeframes": req_tfs,
        "execution": {
            "futures": bool(args.futures),
            "contracts": float(args.contracts),
            "contract_value": float(args.contract_value),
            "fee_perc": float(args.fee_perc),
            "slippage": int(args.slippage),
            "tick_size": float(args.tick_size),
        },
        "risk": {
            "hard_stop_usd_map": hard_stop_usd_map,
            "hard_tp_usd_map": hard_tp_usd_map
        },
        "gates": {
            "cvd_slope_min": float(args.cvd_slope_min),
            "imbalance_min": float(args.imbalance_min),
            "atr_z_min": float(getattr(args, "atr_z_min", 0.0) or 0.0),
            "vhf_min": float(getattr(args, "vhf_min", 0.0) or 0.0),
        },
        "timeout_cfg": {
            "mode": str(args.timeout_mode),
            "max_hold_map": max_hold_map,
            "atr_timeout_len_map": atr_timeout_len_map,
            "atr_timeout_mult_map": atr_timeout_mult_map,
            "atr_stop": {
                "enabled": bool(args.use_atr_stop),
                "len": int(args.atr_stop_len),
                "mult": float(args.atr_stop_mult)
            },
            "atr_tp": {
                "enabled": bool(args.use_atr_tp),
                "len_map": atr_tp_len_map,
                "mult_map": atr_tp_mult_map
            },
            "candle_stop": {
                "enabled": bool(args.use_candle_stop),
                "lookback": int(args.candle_stop_lookback)
            }
        },
        "base_methods": base_methods,
        "combos": combos_struct,
        "ml": {
            "enabled": bool(args.run_ml),
            "kind": str(args.ml_model_kind),
            "horizon": int(args.ml_horizon),
            "lags": int(args.ml_lags),
            "use_agg": bool(args.ml_use_agg),
            "use_depth": bool(args.ml_use_depth),
            "add_base_feats": bool(args.ml_add_base_feats)
        },
        "dl_heads": {
            "available": available_head_names(),
        }
    }
    return runtime



# ===================== Loader determinístico por mês (rápido) =====================

ALLOWED_EXT = (".parquet", ".zip", ".csv")

def _months_between(a, b):
    a = pd.Timestamp(a)
    a = a.tz_convert("UTC") if a.tzinfo else a.tz_localize("UTC")
    b = pd.Timestamp(b)
    b = b.tz_convert("UTC") if b.tzinfo else b.tz_localize("UTC")
    if b < a:
        a, b = b, a
    y, m = int(a.year), int(a.month)
    out = []
    while (y < int(b.year)) or (y == int(b.year) and m <= int(b.month)):
        out.append((y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out

def _month_label(y, m) -> str:
    return f"{int(y):04d}-{int(m):02d}"

def _candidate_for_month(base_dir: str, symbol: str, tf: str, y: int, m: int):
    sym = str(symbol).upper().strip()
    base_tf_dir = os.path.join(base_dir, "klines", tf)
    ym = _month_label(y, m)
    for ext in ALLOWED_EXT:
        cand = os.path.join(base_tf_dir, f"{sym}-{tf}-{ym}{ext}")
        if os.path.isfile(cand):
            return ym, cand
    return None

def _read_month_file_norm(path: str, symbol: str) -> pd.DataFrame:
    """Lê .parquet/.csv/.zip e normaliza para columns [time, open, high, low, close, volume]."""
    pl = path.lower()
    if pl.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif pl.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if not names:
                return pd.DataFrame()
            with zf.open(names[0]) as fh:
                raw = fh.read()
        bio = io.BytesIO(raw)
        try:
            df = pd.read_csv(bio, header=None)
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio)
    else:
        try:
            df = pd.read_csv(path, header=None)
        except Exception:
            df = pd.read_csv(path)

    norm, _ = _normalize_ohlcv(df, symbol)
    if norm is None:
        return pd.DataFrame()
    return norm

def fast_read_klines_monthly(symbol: str, base_dir: str, start, end, tfs, smoke_months: int = 0,
                             max_rows: int | None = None, verbose: bool = False):
    """Carrega klines determinísticamente por (YYYY-MM) sem glob recursivo.
    Retorna {tf: DataFrame} já normalizados e recortados no range.
    """
    out = {}
    months = _months_between(start, end)
    if smoke_months and smoke_months > 0:
        months = months[:smoke_months]
    for tf in [str(x).strip() for x in tfs if str(x).strip()]:
        cand = []
        for (y, m) in months:
            c = _candidate_for_month(base_dir, symbol, tf, y, m)
            if c:
                cand.append(c)
        if verbose:
            print(f"[PLAN] {symbol.upper()} {tf}: meses={len(months)} encontrados={len(cand)}")
            for ym, pth in cand[:10]:
                print(f"       {ym} → {os.path.basename(pth)}")
            if len(cand) > 10:
                print("       …")

        frames = []
        total_rows = 0
        for (ymlabel, path) in cand:
            try:
                df = _read_month_file_norm(path, symbol)
            except Exception as e:
                print(f"[SKIP] Falha lendo {os.path.basename(path)}: {e}")
                continue
            if df.empty:
                continue
            # recorte nas extremidades
            s = pd.Timestamp(start)
            s = s.tz_convert("UTC") if s.tzinfo else s.tz_localize("UTC")
            e = pd.Timestamp(end)
            e = e.tz_convert("UTC") if e.tzinfo else e.tz_localize("UTC")
            if ymlabel == _month_label(s.year, s.month):
                df = df[df["time"] >= s]
            if ymlabel == _month_label(e.year, e.month):
                df = df[df["time"] <= e]
            if df.empty:
                continue
            frames.append(df)
            total_rows += len(df)
            if max_rows and total_rows >= max_rows:
                break
        if not frames:
            print(f"[WARN] Sem arquivos úteis para {symbol.upper()} {tf} no período.")
            continue
        merged = pd.concat(frames, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        s = pd.Timestamp(start); s = s.tz_convert("UTC") if s.tzinfo else s.tz_localize("UTC")
        e = pd.Timestamp(end);   e = e.tz_convert("UTC") if e.tzinfo else e.tz_localize("UTC")
        merged = merged[(merged["time"] >= s) & (merged["time"] <= e)]
        out[tf] = merged
        if verbose:
            print(f"[OK] {symbol.upper()} {tf}: {len(merged):,} linhas (de {total_rows:,} lidas)")
    return out

# =============================== MAIN ==================================

def main():
    ap = argparse.ArgumentParser()
    import multiprocessing as mp
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        psutil = None
        ram_gb = 16.0
    cpu_count = mp.cpu_count()
    ap.add_argument("--umcsv_root", default="", help="Raiz Binance UM CSV/ZIP (ex: /opt/botscalp/datafull). Auto-ajusta data_glob/agg/depth/bbo/funding/etc.")
    ap.add_argument("--load_runtime", default="", help="Carrega runtime_config.json e executa com as mesmas configurações.")
    ap.add_argument("--data_dir", default=".", help="Pasta raiz onde estão os CSV/Parquet")
    ap.add_argument("--force_fallback", action="store_true",
                help="Força o modo rápido de leitura .parquet (ignora read_local_data)")
    ap.add_argument("--data_glob", default="", help="Glob(s) para arquivos (p.ex. 'BTCUSDT_*_1m.parquet;BTCUSDT_*_5m.parquet')")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", default="auto", help='Base interval ou "auto"')
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--auto_range", action="store_true")
    ap.add_argument("--smoke_months", type=int, default=0, help="Se >0, limita ao N de primeiros meses do range (loader rápido)")
    ap.add_argument("--max_rows_loader", type=int, default=0, help="Se >0, interrompe a leitura ao atingir N linhas por TF (loader)")
    ap.add_argument("--loader_verbose", action="store_true", help="Logs detalhados do loader mensal")
    ap.add_argument(
        "--long_only",
        action="store_true",
        help="Se setado, bloqueia posições vendidas (short)."
    )

    # Execução base (estática)
    ap.add_argument("--run_base", action="store_true")
    ap.add_argument("--exec_rules", default="1m,5m,15m")
    ap.add_argument("--methods", default="trend_breakout,keltner_breakout,rsi_reversion,ema_crossover,macd_trend,vwap_trend,boll_breakout,orb_breakout,orr_reversal,ema_pullback,donchian_breakout,vwap_poc_reject,ob_imbalance_break,cvd_divergence_reversal")

    # Combos (estático)
    ap.add_argument("--run_combos", action="store_true")
    ap.add_argument("--combo_ops", default="AND,MAJ,SEQ")
    ap.add_argument("--combo_cap",   type=int, default=400)
    ap.add_argument("--combo_window", type=int, default=2)
    ap.add_argument("--combo_min_votes", type=int, default=2)

    # Execução financeira
    ap.add_argument("--contracts", type=float, default=1.0)
    ap.add_argument("--contract_value", type=float, default=100.0)
    ap.add_argument("--fee_perc", type=float, default=0.0002)
    ap.add_argument("--slippage", type=int, default=0)
    ap.add_argument("--tick_size", type=float, default=0.01)
    ap.add_argument("--max_hold", type=str, default="480",
                help="Max hold por timeframe, pode ser único int ou lista ex: '120,60,30'")
    ap.add_argument("--futures", action="store_true")

    # Stops
    ap.add_argument("--use_atr_stop", action="store_true")
    ap.add_argument("--atr_stop_len", type=int, default=14)
    ap.add_argument("--atr_stop_mult", type=float, default=1.5)
    ap.add_argument("--trailing", action="store_true")
    ap.add_argument("--round_to_tick", action="store_true",
                    help="Arredonda preços de execução ao múltiplo de tick_size após slippage.")

    # Timeout por ATR
    ap.add_argument("--timeout_mode", choices=["bars","atr","both"], default="bars")
    ap.add_argument("--atr_timeout_len", default="", help="int ou lista por TF, ex: '14,14,14'")
    ap.add_argument("--atr_timeout_mult", default="", help="float ou lista por TF, ex: '6,10,12'")

    # TP por ATR (novo)
    ap.add_argument("--use_atr_tp", action="store_true")
    ap.add_argument("--atr_tp_len", default="", help="int ou lista por TF (ex.: '14,14,14'). Vazio => usa atr_stop_len.")
    ap.add_argument("--atr_tp_mult", default="", help="float ou lista por TF (ex.: '2,2,3'). Vazio => 0 (desliga).")
    # HARD STOP/TP em US$ por trade (por TF)
    ap.add_argument("--hard_stop_usd", default="", help="float ou lista por TF (ex.: '200,500,680'); 0 desliga")
    ap.add_argument("--hard_tp_usd",   default="", help="float ou lista por TF (ex.: '3000' ou '3000,3000,3000'); 0 desliga")

    # Stop por Candle (novo)
    ap.add_argument("--use_candle_stop", action="store_true")
    ap.add_argument("--candle_stop_lookback", type=int, default=1, help="N candles para stop por candle (default=1)")

    # Paralelismo
    ap.add_argument("--n_jobs", type=int, default=-1)  # -1 => os.cpu_count()
    ap.add_argument("--par_backend", default="thread", choices=["process","thread","none","auto"])

    # Baixa memória
    ap.add_argument("--lowmem", action="store_true", help="Ativa rotas de baixo uso de RAM (downcast, streaming, sequential).")
    ap.add_argument("--mem_lookback_bars", type=int, default=0, help="Se >0, limita barras usadas no treino por método (WFO).")

    # Saídas (estático)
    ap.add_argument("--out_report", default="selection_report.json")
    ap.add_argument("--out_leaderboard_base", default="leaderboard_base.csv")
    ap.add_argument("--out_leaderboard_combos", default="leaderboard_combos.csv")
    ap.add_argument("--out_leaderboard_all", default="leaderboard_all.csv")
    ap.add_argument("--out_best_trades", default="best_trades.csv")
    ap.add_argument("--out_root", help="Diretório raiz opcional para centralizar todas as saídas (WFO + estático + modelos)")


    # Filtros mínimos
    ap.add_argument("--min_trades", type=int, default=0)
    ap.add_argument("--min_hit", type=float, default=0.0)
    ap.add_argument("--min_pnl", type=float, default=float("-inf"))
    ap.add_argument("--min_sharpe", type=float, default=float("-inf"))
    ap.add_argument("--max_dd", type=float, default=float("inf"))

    # Features (AGG/DEPTH)
    ap.add_argument("--agg_dir", default=os.environ.get("BOTSCALP_AGG_DIR",""))
    ap.add_argument("--depth_dir", default=os.environ.get("BOTSCALP_DEPTH_DIR",""))
    ap.add_argument("--depth_field", default=os.environ.get("BOTSCALP_DEPTH_FIELD","bd_imb_50bps"))
    ap.add_argument("--cvd_slope_min", type=float, default=0.0)
    ap.add_argument("--imbalance_min", type=float, default=0.0)
    ap.add_argument("--atr_z_min", type=float, default=0.0, help="Threshold mínimo do ATR z-score para habilitar sinais.")
    ap.add_argument("--vhf_min", type=float, default=0.0, help="Threshold mínimo do Vertical Horizontal Filter (regime de tendência).")

    # Walk-Forward
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--wf_train_months", type=float, default=3)
    ap.add_argument("--wf_val_months", type=float, default=1)
    ap.add_argument("--wf_step_months", type=float, default=1)
    ap.add_argument("--wf_grid_mode", default="light", help="light|medium|full (tamanho do grid param)")
    ap.add_argument("--wf_top_n", type=int, default=5)
    ap.add_argument("--wf_expand", action="store_true",
                help="Ativa o modo Expand (treino cumulativo: cada janela inclui dados anteriores).")
    ap.add_argument("--wf_no_expand", action="store_true",
                help="Desativa o modo Expand (reset a cada janela, padrão clássico).")
    ap.add_argument("--out_wf_base",   default="wf_leaderboard_base.csv")
    ap.add_argument("--out_wf_combos", default="wf_leaderboard_combos.csv")
    ap.add_argument("--out_wf_all",    default="wf_leaderboard_all.csv")
    ap.add_argument("--out_wf_trades", default="wf_best_trades.csv")
    ap.add_argument("--out_wf_report", default="wf_report.json")

    # ML
    ap.add_argument("--run_ml", action="store_true", help="Executa pipeline ML (WF mensal)")
    ap.add_argument("--ml_model_kind", default="auto", help="auto|xgb|rf|logreg")
    ap.add_argument("--ml_horizon", type=int, default=5)
    ap.add_argument("--ml_ret_thr", type=float, default=0.0, help="margem para definir alvo binário")
    ap.add_argument("--ml_lags", type=int, default=3)
    ap.add_argument("--ml_use_agg", action="store_true")
    ap.add_argument("--ml_use_depth", action="store_true")
    ap.add_argument("--ml_opt_thr", action="store_true", help="Otimiza threshold no treino por janela")
    ap.add_argument("--ml_thr_grid", default="0.50,0.70,0.02", help="ini,fim,passo (ex 0.50,0.70,0.02)")
    ap.add_argument("--ml_thr_fixed", type=float, default=None,
                help="Threshold fixo; se definido, ignora grid e usa esse valor")
    ap.add_argument("--ml_neutral_band", type=float, default=0.0,
                    help="Faixa |p-0.5| < band => flat (0 desliga)")
    ap.add_argument("--ml_add_base_feats", action="store_true", help="Inclui sinais das regras base como features")
    # ---- ML: seeds e combos como features ----
    ap.add_argument("--ml_seed", type=int, default=42, help="Semente global dos modelos/np/random.")
    ap.add_argument("--ml_add_combo_feats", action="store_true", help="Injetar combos como features na ML.")
    ap.add_argument("--ml_combo_top_n", type=int, default=5, help="Top-N combos por TF para virar feature.")
    ap.add_argument("--ml_combo_ops", default="AND,MAJ,SEQ", help="Quais ops de combo considerar nas features (AND,MAJ,SEQ).")
    ap.add_argument("--out_wf_ml", default="wf_leaderboard_ml.csv")
    ap.add_argument("--ml_save_dir", default="", help="Se definido, salva modelo/scaler por janela (WFO‑ML).")
    # ---- ML: opções avançadas ----
    ap.add_argument("--ml_calibrate", default="", help="Método de calibração de probabilidades (ex: 'platt').")
    ap.add_argument("--ml_recency_mode", default="", help="Modo de ponderação temporal ('exp' para exponencial).")
    ap.add_argument("--ml_recency_half_life", type=int, default=0,
                    help="Meia-vida da ponderação temporal (em número de amostras).")


    # Console extra
    ap.add_argument("--print_top10", action="store_true", help="Imprime top 10 dos CSVs gerados")

    # Runtime config (novo)
    ap.add_argument("--out_runtime", default="runtime_config.json")
    ap.add_argument("--runtime_top_k_base", type=int, default=3)
    ap.add_argument("--runtime_top_k_combos", type=int, default=3)

    args = ap.parse_args()
    
    # === PATCH: suporte opcional a --out_root ===
    # mantém compatibilidade total com os defaults já definidos no argparse.

    # Se o usuário quiser centralizar tudo, pode passar --out_root /caminho/
    out_root = getattr(args, "out_root", None)
    if out_root:
        os.makedirs(out_root, exist_ok=True)

        def _auto_redirect(argname: str, default_filename: str):
            val = getattr(args, argname, None)
            # só redefine se o usuário não passou o argumento explicitamente
            if (not val) or (val in [default_filename, "", None]):
                new_path = os.path.join(out_root, default_filename)
                setattr(args, argname, new_path)
                print(f"[INIT] redirecionando {argname} → {new_path}")

        # Walk-Forward
        _auto_redirect("out_wf_base",   "wf_leaderboard_base.csv")
        _auto_redirect("out_wf_combos", "wf_leaderboard_combos.csv")
        _auto_redirect("out_wf_all",    "wf_leaderboard_all.csv")
        _auto_redirect("out_wf_trades", "wf_best_trades.csv")
        _auto_redirect("out_wf_ml",     "wf_leaderboard_ml.csv")
        _auto_redirect("out_wf_report", "wf_report.json")

        # Estático
        _auto_redirect("out_leaderboard_base",   "leaderboard_base.csv")
        _auto_redirect("out_leaderboard_combos", "leaderboard_combos.csv")
        _auto_redirect("out_leaderboard_all",    "leaderboard_all.csv")
        _auto_redirect("out_best_trades",        "best_trades.csv")
        _auto_redirect("out_report",             "selection_report.json")
        _auto_redirect("out_runtime",            "runtime_config.json")

        # Pasta de modelos ML
        # garante que diretório de modelos nunca fique vazio
        ml_dir = getattr(args, "ml_save_dir", None)
        if not ml_dir or ml_dir.strip() == "":
            ml_dir = os.path.join(out_root, "ml")
        args.ml_save_dir = ml_dir
        os.makedirs(args.ml_save_dir, exist_ok=True)

        print(f"[INIT] Out root definido: {out_root}")
        print(f"[INIT] ML save dir: {args.ml_save_dir}")
    else:
        print("[INIT] Out root não definido — usando defaults do argparse.")

    global BASE_KLINES_DIR
    BASE_KLINES_DIR = os.path.join(args.data_dir, "klines")
    # ===== Seeds globais =====
    import random as _random
    try:
        SEED = int(getattr(args, "ml_seed", 42))
    except Exception:
        SEED = 42

    # semente global (numpy/python)
    np.random.seed(SEED)
    _random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # tornar disponível a outros blocos
    globals()["ML_SEED"] = SEED

    # ---------- Auto-config UM CSV ----------
    bbo_dir = funding_dir = mark_dir = prem_dir = ""
    if args.umcsv_root:
        root = os.path.abspath(args.umcsv_root)
        os.environ["BOTSCALP_UMCSV_ROOT"] = root
        sym = args.symbol.upper()
        if not args.data_dir or args.data_dir == ".": 
            args.data_dir = root
        if not args.data_glob:
            args.data_glob = f"klines/1m/{sym}-1m-*.zip;klines/1m/{sym}-1m-*.csv"
        if not args.agg_dir:
            args.agg_dir = os.path.join(root, "aggTrades")
        if not args.depth_dir:
            args.depth_dir = os.path.join(root, "bookDepth")
        bbo_dir     = os.path.join(root, "bookTicker")         if os.path.isdir(os.path.join(root, "bookTicker")) else ""
        funding_dir = os.path.join(root, "fundingRate")        if os.path.isdir(os.path.join(root, "fundingRate")) else ""
        mark_dir    = os.path.join(root, "markPriceKlines")    if os.path.isdir(os.path.join(root, "markPriceKlines")) else ""
        prem_dir    = os.path.join(root, "premiumIndexKlines") if os.path.isdir(os.path.join(root, "premiumIndexKlines")) else ""

    
    # ---------- Leitura OHLCV ----------
    start = parse_when(args.start)
    end = parse_when(args.end)
    req_tfs = [x.strip() for x in args.exec_rules.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    # Loader determinístico mensal (sem fallback forçado)
    requested_tfs = req_tfs if args.interval == "auto" else [args.interval]
    data_map = fast_read_klines_monthly(
        symbol=args.symbol,
        base_dir=args.data_dir,
        start=start,
        end=end,
        tfs=requested_tfs,
        smoke_months=getattr(args, "smoke_months", 0) or 0,
        max_rows=getattr(args, "max_rows_loader", 0) or None,
        verbose=getattr(args, "loader_verbose", False),
    )
    if not data_map:
        print("[ERRO] Sem dados após leitura/normalização (loader mensal). Verifique --data_dir/--symbol/--exec_rules.")
        sys.exit(1)
# ---------- Clamp do range ao disponível ----------
    tmin, tmax = None, None
    for df in data_map.values():
        if df.empty:
            continue
        a = df["time"].iloc[0]
        b = df["time"].iloc[-1]
        tmin = a if tmin is None or a < tmin else tmin
        tmax = b if tmax is None or b > tmax else tmax

    start_clamped = max(start, tmin) if tmin is not None else start
    end_clamped   = min(end, tmax)   if tmax is not None else end

    if start_clamped > end_clamped:
        msg = (f"[ERRO] Range solicitado {start}→{end} não existe nos dados locais "
               f"({tmin}→{tmax}).")
        if args.auto_range and (tmin is not None) and (tmax is not None):
            start_clamped, end_clamped = tmin, tmax
            print(msg)
            print(f"[auto_range] Ajustado para {start_clamped} → {end_clamped} (UTC)")
        else:
            print(msg + " Use --auto_range para ajustar automaticamente ou corrija --start/--end.")
            sys.exit(1)

    print(f"Range local: {start_clamped} → {end_clamped} (UTC)")

    # Converte para timestamps em milissegundos
    start_ms = int(start_clamped.value // 10**6)
    end_ms   = int(end_clamped.value // 10**6)


    # ---------- Enriquecimento completo de features (Full ML-Ready) ----------
    TF_FRAMES.clear()
    FEATURES_META.clear()

    for tf in req_tfs:
        print(f"[ENRICH] Iniciando merge de features para {tf}...")
        try:
            TF_FRAMES[tf] = enrich_with_all_features(
                args.symbol,
                tf,
                start_ms,
                end_ms,
                args.umcsv_root,  # raiz completa: contém klines, agg, depth, funding, etc.
            )

            # Atualiza metadados de presença de features
            dd = TF_FRAMES[tf]
            FEATURES_META[tf] = compute_feature_presence(dd)

            print(f"[ENRICH] {tf} OK: {len(dd)} linhas | {len(dd.columns)} colunas")

        except Exception as e:
            print(f"[ERRO][{tf}] Falha no enriquecimento → {e}")

        # Run heavy pipeline only after all TFs are enriched
        if tf != req_tfs[-1]:
            continue

        # ---------- Empacotar args/TF-maps ----------
        def _safe_int(x, default): 
            try: return int(str(x).strip())
            except Exception: return int(default)
        def _safe_float(x, default):
            try: return float(str(x).strip())
            except Exception: return float(default)
    
        args.max_hold_default = _safe_int(str(args.max_hold).split(",")[0] if args.max_hold else 480, 480)
        args.atr_timeout_len_default = _safe_int(str(args.atr_timeout_len).split(",")[0] if args.atr_timeout_len else args.atr_stop_len, args.atr_stop_len)
        args.atr_timeout_mult_default= _safe_float(str(args.atr_timeout_mult).split(",")[0] if args.atr_timeout_mult else 0.0, 0.0)
    
        max_hold_map        = _expand_list_per_tf(args.max_hold, req_tfs, int, args.max_hold_default)
        atr_timeout_len_map = _expand_list_per_tf(args.atr_timeout_len, req_tfs, int, args.atr_timeout_len_default)
        atr_timeout_mult_map= _expand_list_per_tf(args.atr_timeout_mult, req_tfs, float, args.atr_timeout_mult_default)
    
        args.atr_tp_len_default   = _safe_int(str(args.atr_tp_len).split(",")[0] if args.atr_tp_len else args.atr_stop_len, args.atr_stop_len)
        args.atr_tp_mult_default  = _safe_float(str(args.atr_tp_mult).split(",")[0] if args.atr_tp_mult else 0.0, 0.0)
        atr_tp_len_map  = _expand_list_per_tf(args.atr_tp_len, req_tfs, int, args.atr_tp_len_default)
        atr_tp_mult_map = _expand_list_per_tf(args.atr_tp_mult, req_tfs, float, args.atr_tp_mult_default)
    
        # Hard Stop/TP por TF (pode ser vazio -> 0)
        args.hard_stop_usd_default = _safe_float((str(args.hard_stop_usd).split(",")[0] if args.hard_stop_usd else 0.0), 0.0)
        args.hard_tp_usd_default   = _safe_float((str(args.hard_tp_usd).split(",")[0]   if args.hard_tp_usd   else 0.0), 0.0)
        hard_stop_usd_map = _expand_list_per_tf(args.hard_stop_usd, req_tfs, float, args.hard_stop_usd_default)
        hard_tp_usd_map   = _expand_list_per_tf(args.hard_tp_usd,   req_tfs, float, args.hard_tp_usd_default)
    
        gate_cfg = gate_config_from_args(args)

        def _pack_args(args):
            return dict(
            fee_perc=args.fee_perc, slippage=args.slippage, tick_size=args.tick_size,
            contracts=args.contracts, contract_value=args.contract_value,
            futures=args.futures, use_atr_stop=args.use_atr_stop,
            atr_stop_len=args.atr_stop_len, atr_stop_mult=args.atr_stop_mult,
            trailing=args.trailing,
            cvd_slope_min=gate_cfg.cvd_slope_min, imbalance_min=gate_cfg.imbalance_min,
            atr_z_min=gate_cfg.atr_z_min, vhf_min=gate_cfg.vhf_min,
            gate_config=gate_cfg,
            timeout_mode=args.timeout_mode,
            max_hold_default=args.max_hold_default, max_hold_map=max_hold_map,
            atr_timeout_len_default=args.atr_timeout_len_default, atr_timeout_len_map=atr_timeout_len_map,
            atr_timeout_mult_default=args.atr_timeout_mult_default, atr_timeout_mult_map=atr_timeout_mult_map,
            use_atr_tp=args.use_atr_tp,
            atr_tp_len_default=args.atr_tp_len_default, atr_tp_len_map=atr_tp_len_map,
            atr_tp_mult_default=args.atr_tp_mult_default, atr_tp_mult_map=atr_tp_mult_map,
            use_candle_stop=args.use_candle_stop, candle_stop_lookback=args.candle_stop_lookback,
            round_to_tick=args.round_to_tick,
            hard_stop_usd_default=args.hard_stop_usd_default, hard_stop_usd_map=hard_stop_usd_map,
            hard_tp_usd_default=args.hard_tp_usd_default, hard_tp_usd_map=hard_tp_usd_map,
            lowmem=args.lowmem,
            mem_lookback_bars=args.mem_lookback_bars,
        )    
    
        A = _pack_args(args)
    
        # backend/n_jobs efetivos
        par_backend = args.par_backend
        if args.lowmem:
            print("[LOWMEM] Modo de economia de memória ativado.")
            if args.par_backend == "process":
                # Em vez de desativar totalmente o paralelismo, troca para threads leves
                par_backend = "thread"
                max_workers = min(4, max(1, os.cpu_count() // 4))
                print(f"[LOWMEM] Backend ajustado para threads ({max_workers} workers).")
            else:
                par_backend = args.par_backend
        else:
            par_backend = args.par_backend

        # ==== PATCH: paralelismo adaptativo ====
        import multiprocessing as mp
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            psutil = None
            ram_gb = 16.0
        cpu_count = mp.cpu_count()

        if args.n_jobs in (-1, 0, None):
            max_workers = min(cpu_count, max(2, int(ram_gb // 2)))
        else:
            max_workers = max(1, int(args.n_jobs))

        if args.par_backend == "auto":
            if ram_gb >= 32 and cpu_count >= 8:
                par_backend = "process"
            else:
                par_backend = "thread"
            print(f"[AUTO] RAM={ram_gb:.1f}GB, Cores={cpu_count} → backend={par_backend}, workers={max_workers}")
        else:
            par_backend = args.par_backend

        if par_backend == "none":
            max_workers = 1
        # ==== FIM DO PATCH ====
        # Safety: force 'process' backend to 'thread' (global TF_FRAMES is not picklable)
        if par_backend == "process":
            print("[WARN] Backend 'process' não suportado com TF_FRAMES global; forçando 'thread'.")
            par_backend = "thread"



    
    
        # ================= Caminho 1: Walk-Forward =================
        if args.walkforward:
            combo_ops = [s.strip().upper() for s in (args.combo_ops or "").split(",") if s.strip()]
            base_df, combo_df, all_df, trades_oos_df, gate_stats_base, gate_stats_combo = run_wfo_base_combos(
                req_tfs=req_tfs, methods=methods, A=A,
                start=start_clamped, end=end_clamped,
                train_m=args.wf_train_months, val_m=args.wf_val_months, step_m=args.wf_step_months,
                combo_ops=combo_ops, combo_cap=int(args.combo_cap), combo_window=int(args.combo_window),
                combo_min_votes=int(args.combo_min_votes), grid_mode=str(args.wf_grid_mode),
                n_jobs=max_workers, top_n=int(args.wf_top_n), par_backend=par_backend,
                out_wf_base=args.out_wf_base, out_wf_combos=args.out_wf_combos, out_wf_all=args.out_wf_all,
                out_wf_trades=args.out_wf_trades,
                lookback_bars_train=(args.mem_lookback_bars if args.lowmem and args.mem_lookback_bars>0 else None)
            )
            # Escreve CSVs caso não tenham sido preenchidos por streaming
            if not base_df.empty and (not os.path.exists(args.out_wf_base) or os.path.getsize(args.out_wf_base)==0):
                base_df.sort_values(["wf_window","score"], ascending=[True,False]).to_csv(args.out_wf_base, index=False)
            if not combo_df.empty and (not os.path.exists(args.out_wf_combos) or os.path.getsize(args.out_wf_combos)==0):
                combo_df.sort_values(["wf_window","score"], ascending=[True,False]).to_csv(args.out_wf_combos, index=False)
            if not all_df.empty and (not os.path.exists(args.out_wf_all) or os.path.getsize(args.out_wf_all)==0):
                all_df.sort_values(["wf_window","score"], ascending=[True,False]).to_csv(args.out_wf_all, index=False)
            if not trades_oos_df.empty and (not os.path.exists(args.out_wf_trades) or os.path.getsize(args.out_wf_trades)==0):
                trades_oos_df.to_csv(args.out_wf_trades, index=False)
    
                # ===== Combos como features (seleção top-N) para WFO =====
            combo_feats_by_tf = {}
            if getattr(args, "ml_add_combo_feats", False):
                ops_ok = set([s.strip().upper() for s in str(getattr(args, "ml_combo_ops", "AND,MAJ,SEQ")).split(",") if s.strip()])
                top_n = int(getattr(args, "ml_combo_top_n", 5))

                df_combo_lb = pd.DataFrame(combo_df) if 'combo_df' in locals() and not combo_df.empty else pd.DataFrame()
                if df_combo_lb.empty and os.path.exists(args.out_wf_combos) and os.path.getsize(args.out_wf_combos)>0:
                    try:
                        df_combo_lb = pd.read_csv(args.out_wf_combos)
                    except Exception:
                        df_combo_lb = pd.DataFrame()

                for tf2 in req_tfs:
                    top_specs = []
                    if not df_combo_lb.empty:
                        z = df_combo_lb[df_combo_lb["timeframe"]==tf2].copy()
                        if not z.empty:
                            z = z[z["method"].astype(str).str.startswith("COMBO:")]
                            z["_op"] = z["method"].str.extract(r"COMBO:([A-Z]+)")
                            z = z[z["_op"].str.upper().isin(ops_ok)]
                            z = z.sort_values("score", ascending=False).head(top_n)
                            top_specs = z["combo_spec"].dropna().astype(str).tolist()
                    if top_specs:
                        combo_feats_by_tf[tf2] = _build_combo_signal_feats(TF_FRAMES[tf2], top_specs, futures=bool(args.futures))
                    else:
                        combo_feats_by_tf[tf2] = pd.DataFrame(index=TF_FRAMES[tf2].index)
            else:
                for tf2 in req_tfs:
                    combo_feats_by_tf[tf2] = pd.DataFrame(index=TF_FRAMES[tf2].index)

            # ===== ML (WF mensal) =====
            # ===== ML (WF mensal) =====
            if args.run_ml:
                g_ini, g_fim, g_step = [float(x) for x in (args.ml_thr_grid or "0.50,0.70,0.02").split(",")]
                lb_ml_rows = []

                for tf in [t for t in req_tfs if t in TF_FRAMES]:
                    df_full = TF_FRAMES[tf]
                    windows = _wf_month_splits(
                        start_clamped, end_clamped,
                        args.wf_train_months, args.wf_val_months, args.wf_step_months
                    )
                    if not windows:
                        continue

                    # Controle de modo expand
                    prev_tr_start = None
                    print(f"[ML][{tf}] WFO janelas={len(windows)} | modo={'EXPAND' if args.wf_expand else 'RESET'}")

                    for win_id, (tr_s, tr_e, va_s, va_e) in enumerate(windows, 1):

                        # --- PATCH: modo expand cumulativo ---
                        if args.wf_expand and prev_tr_start is not None:
                            tr_s = prev_tr_start  # mantém o início fixo no 1º treino

                        # Seleção das janelas de treino e validação
                        df_tr = _slice_df_time(df_full, tr_s, tr_e)
                        df_va = _slice_df_time(df_full, va_s, va_e)
                        prev_tr_start = tr_s if prev_tr_start is None else prev_tr_start

                        # Log de acompanhamento
                        print(f"[WFO][{tf}][{win_id:02d}] {'EXPAND' if args.wf_expand else 'RESET'} "
                              f"train={len(df_tr):,} val={len(df_va):,} "
                              f"range={tr_s.date()}→{va_e.date()}")

                        # Verificações mínimas
                        if len(df_tr) < 200 or len(df_va) < 50:
                            continue

                        # ==== (restante do pipeline ML segue daqui) ====
                        # Aqui vem o código que prepara as features, o scaler,
                        # e o treinamento do ensemble dentro da janela atual.

                        # === Base features ===
                        base_feats_tr = _build_base_signal_feats(df_tr, methods, bool(args.futures)) if args.ml_add_base_feats else None
                        base_feats_va = _build_base_signal_feats(df_va, methods, bool(args.futures)) if args.ml_add_base_feats else None

                        # === Combo features (se habilitado) ===

                        if getattr(args, "ml_add_combo_feats", False):
                            def _align_combo_feats(tf_name: str, df_window: pd.DataFrame):
                                """
                                Alinha as combo-features pela coluna 'time' da janela (UTC) via reindex,
                                evitando merge e conflitos de timezone.
                                """
                                combos_full = combo_feats_by_tf.get(tf_name, None)
                                if combos_full is None or combos_full.empty:
                                    return None

                                # time completo do TF base (UTC)
                                base_time = pd.to_datetime(TF_FRAMES[tf_name]["time"], utc=True)

                                # índice do combos_full = time completo (UTC)
                                cf = combos_full.copy()
                                cf.index = base_time

                                # reindex nos times da janela (forçando UTC)
                                twin = pd.to_datetime(df_window["time"], utc=True)
                                cf_win = cf.reindex(twin)

                                return cf_win.reset_index(drop=True)

                            # usa a função para treino/validação da janela
                            combo_feats_tr = _align_combo_feats(tf, df_tr)
                            combo_feats_va = _align_combo_feats(tf, df_va)
                        else:
                            combo_feats_tr = None
                            combo_feats_va = None

                        # ---- diagnóstico (fora do if/else) ----
                        tr_cols = 0 if combo_feats_tr is None else combo_feats_tr.shape[1]
                        va_cols = 0 if combo_feats_va is None else combo_feats_va.shape[1]
                        tr_nnz  = 0 if combo_feats_tr is None else combo_feats_tr.notna().sum().sum()
                        va_nnz  = 0 if combo_feats_va is None else combo_feats_va.notna().sum().sum()
                        print(f"[ML][{tf}] combo_cols_tr={tr_cols} nnz_tr={tr_nnz}  |  combo_cols_va={va_cols} nnz_va={va_nnz}")


                         # === Features finais (OHLCV + Agg + Depth + Base + Combos) ===
                        X_tr_df = _make_ml_features_v2(
                            df_tr,
                            add_lags=int(args.ml_lags),
                            include_agg=bool(args.ml_use_agg),
                            include_depth=bool(args.ml_use_depth),
                            base_signals_df=base_feats_tr,
                            combo_signals_df=combo_feats_tr if 'combo_feats_tr' in locals() else combo_feats_tr
                        )
                        X_va_df = _make_ml_features_v2(
                            df_va,
                            add_lags=int(args.ml_lags),
                            include_agg=bool(args.ml_use_agg),
                            include_depth=bool(args.ml_use_depth),
                            base_signals_df=base_feats_va,
                            combo_signals_df=combo_feats_va if 'combo_feats_va' in locals() else combo_feats_va
                        )

                        # ==== Normalizações adicionais (cvd_norm, imb_zscore) ====
                        if "cvd" in df_tr.columns and "vol_per_bar" in df_tr.columns:
                            X_tr_df["cvd_norm"] = df_tr["cvd"] / (df_tr["vol_per_bar"].rolling(20).mean() + 1e-9)
                        if "imb_net_depth" in df_tr.columns:
                            X_tr_df["imb_zscore"] = (
                                (df_tr["imb_net_depth"] - df_tr["imb_net_depth"].rolling(50).mean())
                                / (df_tr["imb_net_depth"].rolling(50).std() + 1e-9)
                            )
                        if "cvd" in df_va.columns and "vol_per_bar" in df_va.columns:
                            X_va_df["cvd_norm"] = df_va["cvd"] / (df_va["vol_per_bar"].rolling(20).mean() + 1e-9)
                        if "imb_net_depth" in df_va.columns:
                            X_va_df["imb_zscore"] = (
                                (df_va["imb_net_depth"] - df_va["imb_net_depth"].rolling(50).mean())
                                / (df_va["imb_net_depth"].rolling(50).std() + 1e-9)
                            )

                        # ==== Alinhamento & limpeza ====
                        X_tr_df = X_tr_df.dropna().reset_index(drop=True)
                        X_va_df = X_va_df.dropna().reset_index(drop=True)
                        if X_tr_df.empty or X_va_df.empty:
                            continue

                        # ==== Labels adaptativos (baseados em ATR) ====
                        atr_tr = atr(df_tr, 14)
                        atr_va = atr(df_va, 14)

                        # k_mult por TF (mais “folgado” no 5m)
                        k_mult = 0.6 if tf == "5m" else 0.8
                        y_tr = _make_ml_labels(
                            df_tr["close"].astype(float).reindex(X_tr_df.index),
                            atr_series=atr_tr.reindex(X_tr_df.index),
                            k_mult=k_mult,
                            horizon=int(args.ml_horizon),
                        )
                        y_va = _make_ml_labels(
                            df_va["close"].astype(float).reindex(X_va_df.index),
                            atr_series=atr_va.reindex(X_va_df.index),
                            k_mult=k_mult,
                            horizon=int(args.ml_horizon),
                        )

                        # Limpa NaNs e mantém índices alinhados
                        y_tr = y_tr.dropna();  X_tr_df = X_tr_df.loc[y_tr.index]
                        y_va = y_va.dropna();  X_va_df = X_va_df.loc[y_va.index]
                        if y_tr.empty or y_va.empty:
                            continue
                        # --- PATCH: restringir treino a regime volátil ---
                        if "vol_bin" in df_tr.columns:
                            mask_tr = df_tr["vol_bin"].reindex(X_tr_df.index).fillna(0).astype(bool).values
                            before = len(X_tr_df)
                            X_tr_df = X_tr_df.loc[mask_tr].reset_index(drop=True)
                            y_tr = y_tr.loc[mask_tr].reset_index(drop=True)
                            print(f"[ML][{tf}][{win_id}] filtro vol_bin: {before} → {len(X_tr_df)} linhas em regime volátil")
                        # validação (opcional para manter coerência de distribuição)
                        if "vol_bin" in df_va.columns:
                            mask_va = df_va["vol_bin"].reindex(X_va_df.index).fillna(0).astype(bool).values
                            X_va_df = X_va_df.loc[mask_va].reset_index(drop=True)
                            y_va = y_va.loc[mask_va].reset_index(drop=True)

                        # ==== (Opcional) diagnóstico antes do scaling ====
                        # garante grid definido
                        try:
                            g_ini, g_fim, g_step = [float(v) for v in str(args.ml_thr_grid).split(",")]
                        except Exception:
                            g_ini, g_fim, g_step = 0.40, 0.75, 0.02

                        print(
                            f"[ML][{tf}][{win_id}] n_tr={len(X_tr_df)} n_va={len(X_va_df)} "
                            f"pos_tr={y_tr.mean():.3f} pos_va={y_va.mean():.3f} "
                            f"feats={X_tr_df.shape[1]} base={sum(c.startswith('sig_') for c in X_tr_df.columns)} "
                            f"combos={sum(c.startswith('sigc_') for c in X_tr_df.columns)} "
                            f"thr_grid=({g_ini:.2f},{g_fim:.2f},{g_step:.2f})"
                        )

                        # ==== Escalonamento ====
                        feat_cols = [c for c in X_tr_df.columns if c != "time"]
                        sc = _fit_scaler(X_tr_df[feat_cols].values)
                        Xtr = sc.transform(X_tr_df[feat_cols].values)
                        Xva = sc.transform(X_va_df[feat_cols].values)

                        # ==== Modelo (usa ML_SEED via _make_model) ====
                        kind = args.ml_model_kind.lower().strip()
                        if kind == "ensemble":
                            mk = "ensemble"
                            model = _EnsembleModel()
                            model.fit(Xtr, y_tr.values)
                        else:
                            mk, model = _make_model(args.ml_model_kind, seed=ML_SEED)

                            # --- PATCH: ponderação temporal (recency weighting) ---
                            sample_weight = None
                            if getattr(args, "ml_recency_mode", "") == "exp" and len(y_tr) > 10:
                                hl = max(1, int(getattr(args, "ml_recency_half_life", 500)))
                                decay = half_life_weights(len(y_tr), hl)
                                # Mantém compat com implementação anterior (normaliza max para 1)
                                decay_scaled = decay / decay.max()
                                sample_weight = decay_scaled
                                print(f"[ML] Aplicando ponderação temporal exponencial (half_life={hl}, "
                                      f"min={decay_scaled.min():.3f}, max={decay_scaled.max():.3f})")
                            # --- PATCH: peso adicional por regime de volatilidade ---
                            if "atr_zscore" in X_tr_df.columns:
                                vol_weight = np.clip(X_tr_df["atr_zscore"].values, 0.5, 2.0)
                                if sample_weight is None:
                                    sample_weight = vol_weight
                                else:
                                    sample_weight = sample_weight * vol_weight
                                print(f"[ML] Aplicando peso por volatilidade (ATR_zscore) — média {vol_weight.mean():.2f}")
                            # --- treinamento principal ---
                            try:
                                model.fit(Xtr, y_tr.values, sample_weight=sample_weight)
                            except TypeError:
                                model.fit(Xtr, y_tr.values)

                            # --- PATCH: calibração Platt (probabilidade sigmoid) ---
                            if getattr(args, "ml_calibrate", "") == "platt":
                                try:
                                    from sklearn.calibration import CalibratedClassifierCV
                                    # obtém o modelo base real (caso esteja dentro de wrapper)
                                    base_model = getattr(model, "m", model)
                                    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
                                    model.fit(Xtr, y_tr.values)
                                    print("[ML] Calibração Platt aplicada (sigmoid)")
                                except Exception as e:
                                    print(f"[ML][WARN] Falha ao calibrar modelo com Platt: {e}")

                        # ==== Otimização de threshold no treino ====
                        # grid de thresholds (ex.: "0.40,0.75,0.02")
                        try:
                            g_ini, g_fim, g_step = [float(v) for v in str(args.ml_thr_grid).split(",")]
                        except Exception:
                            g_ini, g_fim, g_step = 0.40, 0.75, 0.02

                        # define a lista de thresholds conforme flags (prioridade: fixed > grid > default)
                        if args.ml_thr_fixed is not None:
                            thr_list = np.array([args.ml_thr_fixed], dtype=float)
                            print(f"[ML] Threshold fixo forçado em {args.ml_thr_fixed}")
                        elif args.ml_opt_thr:
                            thr_list = np.arange(g_ini, g_fim + 1e-9, g_step)
                        else:
                            thr_list = np.array([0.55], dtype=float)

                        # probas do treino em 1D
                        p_tr = model.predict_proba(Xtr)
                        if p_tr.ndim == 2:
                            p_tr = p_tr[:, 1]
                        p_tr = np.asarray(p_tr, dtype=float).ravel()

                        # diagnóstico (opcional)
                        print(
                            f"[ML][{tf}][{win_id}] p_tr: mean={p_tr.mean():.3f} std={p_tr.std():.3f} "
                            f"p10={np.percentile(p_tr,10):.3f} p90={np.percentile(p_tr,90):.3f}"
                        )

                        best_thr, best_score = float(thr_list[0]), -1e18

                        for thr in thr_list:
                            # sinais 1D a partir das probabilidades
                            sig_tr = np.zeros(p_tr.shape[0], dtype=np.int8)
                            sig_tr[p_tr > thr] = 1
                            if args.futures:
                                sig_tr[p_tr < (1.0 - thr)] = -1  # shorts simétricos

                            trades_tr = backtest_from_signals(
                                df_tr.loc[X_tr_df.index].reset_index(drop=True),
                                pd.Series(sig_tr),
                                hard_stop_usd=float(A["hard_stop_usd_map"].get(tf, A["hard_stop_usd_default"])),
                                hard_tp_usd=float(A["hard_tp_usd_map"].get(tf, A["hard_tp_usd_default"])),
                                max_hold=int(A["max_hold_map"].get(tf, A["max_hold_default"])),
                                fee_perc=float(A["fee_perc"]),
                                slippage_ticks=int(A["slippage"]), tick_size=float(A["tick_size"]),
                                contracts=float(A["contracts"]), contract_value=float(A["contract_value"]),
                                futures=bool(A["futures"]),
                                use_atr_stop=bool(A["use_atr_stop"]), atr_stop_len=int(A["atr_stop_len"]),
                                atr_stop_mult=float(A["atr_stop_mult"]),
                                use_atr_tp=bool(A["use_atr_tp"]),
                                atr_tp_len=int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
                                atr_tp_mult=float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
                                use_candle_stop=bool(A["use_candle_stop"]),
                                candle_stop_lookback=int(A["candle_stop_lookback"]),
                                trailing=bool(A["trailing"]), timeout_mode=str(A["timeout_mode"]),
                                atr_timeout_len=int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
                                atr_timeout_mult=float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
                                round_to_tick=bool(A.get("round_to_tick", False)),
                            )

                            # <<< ATENÇÃO: métricas e atualização DO LADO DE DENTRO do loop >>>
                            mtr_tr = _metrics_with_fees(
                                trades_tr, float(A["fee_perc"]),
                                contracts=float(A["contracts"]), contract_value=float(A["contract_value"])
                            )
                            sc_tr = (mtr_tr["total_pnl"] * 1.0) + (mtr_tr["hit"] * 100.0) + \
                                    (mtr_tr["payoff"] * 10.0) - (mtr_tr["maxdd"] * 0.5)

                            if sc_tr > best_score:
                                best_score, best_thr = sc_tr, float(thr)

                        print(f"[ML][{tf}][{win_id}] selected_thr={best_thr:.2f} (best_score={best_score:.2f})")



    
                        # ==== OOS ====
                        p_va = model.predict_proba(Xva)
                        if p_va.ndim == 2:
                            p_va = p_va[:, 1]
                        sig_va = np.zeros_like(p_va, dtype=np.int8)
                        sig_va[p_va > best_thr] = 1
                        if args.futures:
                            sig_va[p_va < (1.0 - best_thr)] = -1
                        nb = max(0.0, float(getattr(args, "ml_neutral_band", 0.0)))
                        if nb > 0:
                            sig_va[np.abs(p_va - 0.5) < nb] = 0
                        # === DEBUG: verificar distribuição de sinais antes do filtro long-only ===
                        try:
                            print(f"[DEBUG][{tf}] unique(sig_tr)={np.unique(sig_tr, return_counts=True)}")
                            print(f"[DEBUG][{tf}] unique(sig_va)={np.unique(sig_va, return_counts=True)}")
                        except Exception as e:
                            print(f"[DEBUG][{tf}] erro ao inspecionar sinais: {e}")

                        # === LONG-ONLY PATCH (robusto, inclui sanitização total) ===
                        if args.long_only:
                            try:
                                if "sig_va" in locals():
                                    sig_va = np.array(sig_va).astype(float)
                                    sig_va[sig_va < 0] = 0
                                    sig_va[np.isnan(sig_va)] = 0
                                if "sig_tr" in locals():
                                    sig_tr = np.array(sig_tr).astype(float)
                                    sig_tr[sig_tr < 0] = 0
                                    sig_tr[np.isnan(sig_tr)] = 0
                                print(f"[LONG_ONLY][{tf}] ativos: longs={np.sum(sig_va>0)} / total={len(sig_va)}")
                            except Exception as e:
                                print(f"[WARN][LONG_ONLY] erro aplicando filtro long_only: {e}")
                        trades_va = backtest_from_signals(
                            df_va.reindex(X_va_df.index).reset_index(drop=True),
                            pd.Series(sig_va),
                            hard_stop_usd=float(A["hard_stop_usd_map"].get(tf, A["hard_stop_usd_default"])),
                            hard_tp_usd=float(A["hard_tp_usd_map"].get(tf, A["hard_tp_usd_default"])),
                            max_hold=int(A["max_hold_map"].get(tf, A["max_hold_default"])),
                            fee_perc=float(A["fee_perc"]),
                            slippage_ticks=int(A["slippage"]), tick_size=float(A["tick_size"]),
                            contracts=float(A["contracts"]), contract_value=float(A["contract_value"]),
                            futures=bool(A["futures"]), use_atr_stop=bool(A["use_atr_stop"]),
                            atr_stop_len=int(A["atr_stop_len"]), atr_stop_mult=float(A["atr_stop_mult"]),
                            use_atr_tp=bool(A["use_atr_tp"]),
                            atr_tp_len=int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
                            atr_tp_mult=float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
                            use_candle_stop=bool(A["use_candle_stop"]),
                            candle_stop_lookback=int(A["candle_stop_lookback"]),
                            trailing=bool(A["trailing"]), timeout_mode=str(A["timeout_mode"]),
                            atr_timeout_len=int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
                            atr_timeout_mult=float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
                            round_to_tick=bool(A.get("round_to_tick", False)),
                        )
                        # ===== Salvar probabilidades OOS por janela (para AUC/Brier/Precision@K) =====
                        try:
                            oos_dump = pd.DataFrame({
                                "time": df_va["time"].reindex(X_va_df.index).reset_index(drop=True),
                                "price": df_va["close"].reindex(X_va_df.index).reset_index(drop=True),
                                "p": pd.Series(p_va).reset_index(drop=True),
                                "sig": pd.Series(sig_va).reset_index(drop=True),
                                # se quiser incluir o rótulo real:
                                "y": y_va.reset_index(drop=True),
                            })
                            oos_path = os.path.join(args.out_root, f"oos_probs_{tf}_win{win_id:02d}.csv")
                            oos_dump.to_csv(oos_path, index=False)
                            print(f"[OOS] Probabilidades salvas em {oos_path}")
                        except Exception as e:
                            print(f"[WARN] Falha ao salvar oos_probs_{tf}_win{win_id:02d}: {e}")
    
                        m_va = _metrics_with_fees(trades_va, float(A["fee_perc"]),
                                                  contracts=float(A["contracts"]),
                                                  contract_value=float(A["contract_value"]))
                        score_va = (m_va["total_pnl"] * 1.0) + (m_va["hit"] * 100.0) + \
                                   (m_va["payoff"] * 10.0) - (m_va["maxdd"] * 0.5)
                        lb_ml_rows.append({
                            "wf_window": f"{tr_s.strftime('%Y-%m')}→{va_e.strftime('%Y-%m')}",
                            "method": f"ML:{mk}", "timeframe": tf, **m_va, "score": score_va,
                            "is_combo": False, "combo_spec": "", "timeout_mode": str(args.timeout_mode),
                            "max_hold_used": int(A["max_hold_map"].get(tf, A["max_hold_default"])),
                            "atr_timeout_mult_used": float(A["atr_timeout_mult_map"].get(tf, A["atr_timeout_mult_default"])),
                            "atr_timeout_len_used": int(A["atr_timeout_len_map"].get(tf, A["atr_timeout_len_default"])),
                            "use_atr_stop": bool(A["use_atr_stop"]),
                            "atr_stop_len_used": int(A["atr_stop_len"]),
                            "atr_stop_mult_used": float(A["atr_stop_mult"]),
                            "ml_thr_used": best_thr,
                            "use_atr_tp": bool(A["use_atr_tp"]),
                            "atr_tp_len_used": int(A["atr_tp_len_map"].get(tf, A["atr_tp_len_default"])),
                            "atr_tp_mult_used": float(A["atr_tp_mult_map"].get(tf, A["atr_tp_mult_default"])),
                            "use_candle_stop": bool(A["use_candle_stop"]),
                            "candle_stop_lookback": int(A["candle_stop_lookback"])
                        })
    
                        # ==== Salvamento opcional ====
                        if args.ml_save_dir:
                            try:
                                os.makedirs(args.ml_save_dir, exist_ok=True)
                                tag = f"{str(args.symbol).upper()}_{tf}_{tr_s.strftime('%Y%m')}_{va_e.strftime('%Y%m')}"
                                import joblib, pickle
                                joblib.dump(sc, os.path.join(args.ml_save_dir, f"scaler_{tag}.pkl"))
                                if hasattr(model, "m") and hasattr(model.m, "save_model"):
                                    model.m.save_model(os.path.join(args.ml_save_dir, f"model_{tag}.json"))
                                else:
                                    joblib.dump(getattr(model, "m", model), os.path.join(args.ml_save_dir, f"model_{tag}.pkl"))
                            except Exception as e:
                                print(f"[ML] Falha ao salvar em --ml_save_dir: {e}")
                        # --- PATCH: salvar ensemble global pronto para produção ---
                        try:
                            if args.ml_save_dir and lb_ml_rows:
                                import joblib
                                from sklearn.linear_model import LogisticRegression

                                os.makedirs(args.ml_save_dir, exist_ok=True)
                                final_tag = f"{str(args.symbol).upper()}_{tf}_ensemble_final"

                                # === Agregar modelos ===
                                model_files = [f for f in os.listdir(args.ml_save_dir)
                                               if f.startswith("model_") and f.endswith(".pkl")]
                                scalers = [f for f in os.listdir(args.ml_save_dir)
                                           if f.startswith("scaler_") and f.endswith(".pkl")]

                                models = []
                                for mf in model_files:
                                    try:
                                        m = joblib.load(os.path.join(args.ml_save_dir, mf))
                                        models.append(m)
                                    except Exception:
                                        pass

                                if models:
                                    # Usar o primeiro modelo como base
                                    base = models[0]
                                    if hasattr(base, "coef_") and hasattr(base, "intercept_"):
                                        # Média dos coeficientes e interceptos
                                        base.coef_ = np.mean([m.coef_ for m in models if hasattr(m, "coef_")], axis=0)
                                        base.intercept_ = np.mean([m.intercept_ for m in models if hasattr(m, "intercept_")], axis=0)
                                        joblib.dump(base, os.path.join(args.ml_save_dir, f"{final_tag}.pkl"))
                                        print(f"[ML] Modelo ensemble final salvo em {args.ml_save_dir}/{final_tag}.pkl")

                                    # === Scaler global ===
                                    if scalers:
                                        try:
                                            from sklearn.preprocessing import StandardScaler
                                            scalers_obj = [joblib.load(os.path.join(args.ml_save_dir, s)) for s in scalers]

                                            # --- PATCH: só usar scalers com mesma shape ---
                                            shapes = [len(s.mean_) for s in scalers_obj]
                                            mode_shape = max(set(shapes), key=shapes.count)
                                            scalers_ok = [s for s in scalers_obj if len(s.mean_) == mode_shape]

                                            if len(scalers_ok) >= 2:
                                                means = np.mean([s.mean_ for s in scalers_ok], axis=0)
                                                vars_ = np.mean([s.var_ for s in scalers_ok], axis=0)
                                                scaler_final = StandardScaler()
                                                scaler_final.mean_, scaler_final.var_ = means, vars_
                                                scaler_final.scale_ = np.sqrt(vars_)
                                                joblib.dump(scaler_final, os.path.join(args.ml_save_dir, f"scaler_{final_tag}.pkl"))
                                                print(f"[ML] Scaler ensemble final salvo em {args.ml_save_dir}/scaler_{final_tag}.pkl")
                                            else:
                                                # fallback: usa o último scaler válido se shapes inconsistentes
                                                joblib.dump(scalers_obj[-1], os.path.join(args.ml_save_dir, f"scaler_{final_tag}_last.pkl"))
                                                print("[ML] Pulando ensemble (shapes inconsistentes). Usando último scaler como fallback.")
                                        except Exception as e:
                                            print(f"[ML] Falha ao criar scaler ensemble: {e}")


                                    # === Metadata ===
                                    meta = {
                                        "symbol": str(args.symbol).upper(),
                                        "timeframe": tf,
                                        "n_models": len(models),
                                        "train_range": f"{start_clamped} → {end_clamped}",
                                        "features_count": int(base.coef_.shape[1]) if hasattr(base, "coef_") else None,
                                        "created_utc": pd.Timestamp.utcnow().isoformat(),
                                        "ensemble_model": f"{final_tag}.pkl",
                                        "ensemble_scaler": f"scaler_{final_tag}.pkl",
                                        "source": "WFO-ML ensemble average"
                                    }
                                    meta_path = os.path.join(args.ml_save_dir, f"{final_tag}_meta.json")
                                    with open(meta_path, "w", encoding="utf-8") as f:
                                        json.dump(meta, f, indent=2)
                                    print(f"[ML] Metadata do ensemble salva em {meta_path}")
                        except Exception as e:
                            print(f"[ML] Falha ao salvar ensemble final: {e}")
    
                # ==== Salva leaderboard ====
                if lb_ml_rows:
                    try:
                        pd.DataFrame(lb_ml_rows).sort_values(["wf_window", "score"], ascending=[True, False]).to_csv(
                            args.out_wf_ml, index=False
                        )
                        print(f"[WFO-ML] Leaderboard salvo em {args.out_wf_ml}")
                    except Exception as e:
                        print(f"[WFO-ML] Falha ao salvar {args.out_wf_ml}: {e}")
    
    
            # Top10 prints (opcional)
            if args.print_top10:
                def _print_top10(path: str, title: str, n: int = 10):
                    try:
                        if os.path.exists(path) and os.path.getsize(path) > 0:
                            df = pd.read_csv(path)
                            if not df.empty:
                                print(f"---- {title} (top {n}) ----")
                                print(df.head(n).to_csv(index=False).rstrip())
                    except Exception as e:
                        print(f"[warn] print_top10 {title}: {e}")
                _print_top10(args.out_wf_base,   "wf_leaderboard_base")
                _print_top10(args.out_wf_combos, "wf_leaderboard_combos")
                _print_top10(args.out_wf_all,    "wf_leaderboard_all")
                if args.run_ml: _print_top10(args.out_wf_ml, "wf_leaderboard_ml")
    
            # Verificação de uso de features
            print("[verify][WFO] uso de features por timeframe:")
            for tf in req_tfs:
                if tf not in TF_FRAMES:
                    continue
                fm = FEATURES_META.get(tf, {})
                print(f"[verify][{tf}] cvd_slope_agg: present={fm.get('cvd_present', False)} nnz={fm.get('cvd_nonnull',0.0):.1%}; "
                      f"imb_net_depth: present={fm.get('imb_present', False)} nnz={fm.get('imb_nonnull',0.0):.1%}.")
    
            report = {
                "range_used": {"start": str(start_clamped), "end": str(end_clamped)},
                "files": {
                    "wf_base": args.out_wf_base,
                    "wf_combos": args.out_wf_combos,
                    "wf_all": args.out_wf_all,
                    "wf_trades": args.out_wf_trades,
                    "wf_ml": args.out_wf_ml if args.run_ml else None,
                },
                "feature_meta": FEATURES_META,
                "timeout_cfg": {
                    "mode": str(args.timeout_mode),
                    "max_hold_map": max_hold_map,
                    "atr_timeout_len_map": atr_timeout_len_map,
                    "atr_timeout_mult_map": atr_timeout_mult_map,
                    "atr_stop": {"enabled": bool(args.use_atr_stop), "len": int(args.atr_stop_len), "mult": float(args.atr_stop_mult)}
                }
            }
            with open(args.out_wf_report, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[OK] Relatório WFO salvo em {args.out_wf_report}")
    
            # ===== Runtime config auto =====
            runtime = build_runtime_config(args, req_tfs, start_clamped, end_clamped, mode="wfo")
            with open(args.out_runtime, "w", encoding="utf-8") as f:
                json.dump(runtime, f, ensure_ascii=False, indent=2)
            print(f"[OK] Runtime config salvo em {args.out_runtime}")
    
        # ================= Caminho 2: Execução estática =================
        lb_base_rows = []; lb_combo_rows = []; best_trades_all = []
        gate_stats_base: Dict[str, Dict[str,int]] = {}
        gate_stats_combo: Dict[str, Dict[str,int]] = {}
    
        def _acc_gate(acc: Dict[str, Dict[str,int]], tf: str, gs: Dict[str,int]):
            cur = acc.get(tf, {})
            for k, v in gs.items():
                if k == "tf":
                    continue
                cur[k] = cur.get(k, 0) + int(v)
            acc[tf] = cur
    
        if args.run_base:
            base_tasks = []
            for m in methods:
                if m not in STRATEGY_FUNCS:
                    print(f"[BASE] Método inválido: {m}. Pulando.")
                    continue
                for tf in req_tfs:
                    if tf not in TF_FRAMES:
                        print(f"[BASE] Sem dados em {tf}. Pulando {m}.")
                        continue
                    base_tasks.append((m, tf))
    
            task_payloads = [(m, tf, A) for (m, tf) in base_tasks]
            for success, payload in iter_task_results(task_payloads, _worker_base, backend=par_backend, max_workers=max_workers):
                if success:
                    row, bt, gs = payload
                    lb_base_rows.append(row)
                    if bt is not None and not bt.empty:
                        best_trades_all.append(bt)
                    _acc_gate(gate_stats_base, row["timeframe"], gs)
                else:
                    print(f"[BASE] worker fail: {payload}")

        if args.run_combos:
            base_methods = [m for m in methods if m in STRATEGY_FUNCS]
            ops = [s.strip().upper() for s in (args.combo_ops or "").split(",") if s.strip()]
            combos_list = []
            for a,b in combinations(base_methods, 2):
                if "AND" in ops: combos_list.append(f"AND({a},{b})")
                if "MAJ" in ops: combos_list.append(f"MAJ2({a},{b})")
                if "SEQ" in ops: combos_list.append(f"SEQ({a}->{b})")
            for a,b,c in combinations(base_methods, 3):
                if "AND" in ops: combos_list.append(f"AND({a},{b},{c})")
                if "MAJ" in ops: combos_list.append(f"MAJ2({a},{b},{c})")
            combos_list = list(dict.fromkeys(combos_list))[:args.combo_cap]
    
            combo_tasks = []
            for tf in req_tfs:
                if tf not in TF_FRAMES:
                    continue
                for raw in combos_list:
                    combo_tasks.append((raw, tf))
    
            combo_payloads = [(raw, tf, A, args.combo_window, args.combo_min_votes) for (raw, tf) in combo_tasks]
            for success, payload in iter_task_results(combo_payloads, _worker_combo, backend=par_backend, max_workers=max_workers):
                if success:
                    row, gs = payload
                    if row.get("_skip"):
                        continue
                    lb_combo_rows.append(row)
                    _acc_gate(gate_stats_combo, row["timeframe"], gs)
                else:
                    print(f"[COMBOS] worker fail: {payload}")
    
        # ---------- Escritas estático ----------
        if lb_base_rows:
            pd.DataFrame(lb_base_rows).sort_values("score", ascending=False).to_csv(args.out_leaderboard_base, index=False)
            print(f"[BASE] Leaderboard salvo em {args.out_leaderboard_base}")
        if lb_combo_rows:
            pd.DataFrame(lb_combo_rows).sort_values("score", ascending=False).to_csv(args.out_leaderboard_combos, index=False)
            print(f"[COMBOS] Leaderboard salvo em {args.out_leaderboard_combos}")
            
        # ===== Combos como features (seleção top-N) =====
        combo_feats_by_tf = {}
        if getattr(args, "ml_add_combo_feats", False):
            ops_ok = set([s.strip().upper() for s in str(getattr(args, "ml_combo_ops", "AND,MAJ,SEQ")).split(",") if s.strip()])
            top_n = int(getattr(args, "ml_combo_top_n", 5))

            # usa o leaderboard de combos da execução atual (se existe em memória)
            df_combo_lb = pd.DataFrame(lb_combo_rows) if 'lb_combo_rows' in globals() else pd.DataFrame()
            for tf in req_tfs:
                top_specs = []
                if not df_combo_lb.empty:
                    z = df_combo_lb[df_combo_lb["timeframe"]==tf].copy()
                    if not z.empty:
                        z = z[z["method"].astype(str).str.startswith("COMBO:")]
                        z["_op"] = z["method"].str.extract(r"COMBO:([A-Z]+)")
                        z = z[z["_op"].str.upper().isin(ops_ok)]
                        z = z.sort_values("score", ascending=False).head(top_n)
                        top_specs = z["combo_spec"].dropna().astype(str).tolist()
                if top_specs:
                    combo_feats_by_tf[tf] = _build_combo_signal_feats(TF_FRAMES[tf], top_specs, futures=bool(args.futures))
                else:
                    combo_feats_by_tf[tf] = pd.DataFrame(index=TF_FRAMES[tf].index)
        else:
            for tf in req_tfs:
                combo_feats_by_tf[tf] = pd.DataFrame(index=TF_FRAMES[tf].index)
    
        lb_all = []
        if lb_base_rows: lb_all += lb_base_rows
        if lb_combo_rows: lb_all += lb_combo_rows
        if lb_all:
            all_df = pd.DataFrame(lb_all)
            flt = (
                (all_df["n_trades"] >= args.min_trades) &
                (all_df["hit"]      >= args.min_hit)    &
                (all_df["total_pnl"]>= args.min_pnl)   &
                (all_df["sharpe"]   >= args.min_sharpe)&
                (all_df["maxdd"]    <= args.max_dd)
            )
            out = all_df.loc[flt].copy() if flt.any() else all_df
            out.sort_values("score", ascending=False).to_csv(args.out_leaderboard_all, index=False)
            print(f"[ALL] Leaderboard combinado salvo em {args.out_leaderboard_all}")
    
        if best_trades_all:
            pd.concat(best_trades_all, ignore_index=True).to_csv(args.out_best_trades, index=False)
            print(f"[BEST] Trades consolidados salvos em {args.out_best_trades}")
    
        # ---------- Impressão top10 (estático) ----------
        if args.print_top10:
            def _print_top10(path: str, title: str, n: int = 10):
                try:
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        df = pd.read_csv(path)
                        if not df.empty:
                            print(f"---- {title} (top {n}) ----")
                            print(df.head(n).to_csv(index=False).rstrip())
                except Exception as e:
                    print(f"[warn] print_top10 {title}: {e}")
            _print_top10(args.out_leaderboard_base,   "leaderboard_base")
            _print_top10(args.out_leaderboard_combos, "leaderboard_combos")
            _print_top10(args.out_leaderboard_all,    "leaderboard_all")
            _print_top10(args.out_best_trades,        "best_trades")
    
        # ---------- Verificação de features ----------
        print("[verify] uso de features por timeframe:")
        for tf in req_tfs:
            if tf not in TF_FRAMES:
                continue
            fm = FEATURES_META.get(tf, {})
            print(f"[verify][{tf}] cvd_slope_agg: present={fm.get('cvd_present', False)} nnz={fm.get('cvd_nonnull',0.0):.1%}; "
                  f"imb_net_depth: present={fm.get('imb_present', False)} nnz={fm.get('imb_nonnull',0.0):.1%}.")
    
        report = {
            "range_used": {"start": str(start_clamped), "end": str(end_clamped)},
            "files": {
                "leaderboard_base": args.out_leaderboard_base,
                "leaderboard_combos": args.out_leaderboard_combos,
                "leaderboard_all": args.out_leaderboard_all,
                "best_trades": args.out_best_trades,
            },
            "feature_meta": FEATURES_META,
            "timeout_cfg": {
                "mode": str(args.timeout_mode),
                "max_hold_map": max_hold_map,
                "atr_timeout_len_map": atr_timeout_len_map,
                "atr_timeout_mult_map": atr_timeout_mult_map,
                "atr_stop": {"enabled": bool(args.use_atr_stop), "len": int(args.atr_stop_len), "mult": float(args.atr_stop_mult)}
            }
        }
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
        print(f"[OK] Relatório salvo em {args.out_report}")
    
        # ===== Runtime config auto (estático) =====
        runtime = build_runtime_config(args, req_tfs, start_clamped, end_clamped, mode="static")
        with open(args.out_runtime, "w", encoding="utf-8") as f:
            json.dump(runtime, f, ensure_ascii=False, indent=2)
        print(f"[OK] Runtime config salvo em {args.out_runtime}")
    # ===========================================================
# Função utilitária: carregar runtime_config.json e rodar replay
# ===========================================================
def _load_runtime(path: str):
    if not path or not os.path.exists(path):
        print(f"[ERRO] Runtime config inexistente: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        rt = json.load(f)

    print(f"[LOAD_RUNTIME] Carregado {path}")
    sym = rt["meta"].get("symbol", "BTCUSDT")
    req_tfs = rt.get("timeframes", [])
    ml_cfg = rt.get("ml", {})
    base_methods = []
    for tf in req_tfs:
        if tf in rt["base_methods"]:
            base_methods += [m["name"] for m in rt["base_methods"][tf]]
    base_methods = list(set(base_methods))
    print(f"[LOAD_RUNTIME] Symbol={sym}, TFs={req_tfs}, Métodos={base_methods}")

    # --- reconstruir config básica de execução ---
    exec_cfg = rt.get("execution", {})
    fee = exec_cfg.get("fee_perc", 0.0002)
    tick = exec_cfg.get("tick_size", 0.1)
    contr = exec_cfg.get("contracts", 1.0)
    cval = exec_cfg.get("contract_value", 100.0)
    futures = exec_cfg.get("futures", True)

    # --- localizar dados ---
    data_dir = "/opt/botscalp/datafull"
    kl_dir = os.path.join(data_dir, f"{sym}.klines.{req_tfs[0]}")
    agg_dir = os.path.join(data_dir, f"{sym}.aggtrades.parquet")
    depth_dir = os.path.join(data_dir, f"{sym}.depthfeat_{req_tfs[0]}.parquet")

    if not os.path.exists(kl_dir):
        print(f"[WARN] klines {kl_dir} não encontrado. Continuando com enrich_with_all_features.")

    print(f"[LOAD_RUNTIME] Recarregando data e features...")
    data_map = read_local_data(sym, requested_tfs=req_tfs, base_dir=data_dir)
    start = pd.to_datetime(rt["meta"]["range_used"]["start"])
    end   = pd.to_datetime(rt["meta"]["range_used"]["end"])
    start_ms = int(start.value // 10**6)
    end_ms   = int(end.value // 10**6)
    # SUBSTITUA o bloco de "recarregar data e features" por:
    TF_FRAMES.clear()
    start_ms = int(start.value // 10**6)
    end_ms   = int(end.value   // 10**6)
    for tf in req_tfs:
        TF_FRAMES[tf] = enrich_with_all_features(
            sym, tf, start_ms, end_ms, data_dir  
        )

    # --- Executar sinais e backtest base e ML ---
    results = []
    for tf in req_tfs:
        df = TF_FRAMES[tf]
        print(f"[REPLAY][{tf}] {len(df)} barras carregadas")
        for m in base_methods:
            if m not in STRATEGY_FUNCS:
                continue
            sig = STRATEGY_FUNCS[m](df, {}, futures).shift(1).fillna(0).astype("int8")
            trades = backtest_from_signals(
                df, sig, fee_perc=fee, tick_size=tick,
                contracts=contr, contract_value=cval, futures=futures,
                use_atr_stop=True, atr_stop_len=14, atr_stop_mult=3.0,
                use_atr_tp=True, atr_tp_len=14, atr_tp_mult=2.2,
                use_candle_stop=True, candle_stop_lookback=2,
                timeout_mode="both", max_hold=100,
                atr_timeout_len=14, atr_timeout_mult=4,
                round_to_tick=True)
            mtr = _metrics_with_fees(trades, fee,
                contracts=contr, contract_value=cval)
            results.append({
                "timeframe": tf, "method": m,
                **mtr, "n_bars": len(df),
                "start": start, "end": end,
                "datetime_exec": datetime.utcnow().isoformat()
            })
    if results:
        out_csv = os.path.join(os.path.dirname(path), "runtime_replay.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"[LOAD_RUNTIME] Replay concluído → {out_csv}")
    else:
        print("[LOAD_RUNTIME] Nenhum resultado gerado.")

# ===========================================================
# Execução automática se --load_runtime for usado
# ===========================================================
# ===========================================================

# ============================================================
# Loader robusto de AggTrades — compatível com Binance Vision
# ============================================================
def load_aggtrades_dir(dir_path: str, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Lê arquivos AggTrades (CSV/ZIP/Parquet) da Binance Vision — compatível com formato oficial.
    Corrige timestamps, agrega volumes e gera colunas: cvd, taker_buy_vol, taker_sell_vol, vol_per_bar.

    ⚙️ Otimizações:
    - Filtro ultra-rápido por nome (sem abrir arquivos fora do range)
    - Checagem de schema sem ler parquet inteiro (pyarrow.read_schema)
    - Leitura paralela (até 16 threads)
    - Fallback seguro pra CSV/ZIP
    """
    import os, glob, io, zipfile, re
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    try:
        import pyarrow.parquet as pq
    except Exception:
        pq = None

    if not os.path.isdir(dir_path):
        print(f"[WARN] Diretório inexistente: {dir_path}")
        return pd.DataFrame()

    # --- FAST LISTING (sem recursion) ---
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith((".parquet", ".csv", ".zip"))]
    if not files:
        print(f"[LOAD] Nenhum AggTrades encontrado em {dir_path}")
        return pd.DataFrame()

    # --- FAST RANGE FILTER pelo nome (sem I/O) ---
    start_num = int(pd.to_datetime(start_ms, unit="ms").strftime("%Y%m"))
    end_num   = int(pd.to_datetime(end_ms, unit="ms").strftime("%Y%m"))
    valid_files = []
    for f in files:
        m = re.search(r"(\d{4})-(\d{2})", f)
        if not m:
            valid_files.append(f)
            continue
        ym = int(m.group(1)) * 100 + int(m.group(2))
        if start_num <= ym <= end_num:
            valid_files.append(f)
        else:
            # loga mas não lê
            print(f"[SKIP] {os.path.basename(f)} fora do range {start_num}-{end_num}")
    files = sorted(valid_files)
    if not files:
        print(f"[LOAD] Nenhum Agg válido no range {start_num}-{end_num}")
        return pd.DataFrame()

    # --- FAST-METADATA CHECK via pyarrow ---
    def parquet_has_time(path):
        if pq is None:
            return True
        try:
            schema = pq.read_schema(path)
            return "time" in schema.names or "timestamp" in schema.names
        except Exception:
            return True  # assume válido para CSV/ZIP

    with ThreadPoolExecutor(max_workers=8) as ex:
        meta_valid = list(ex.map(lambda f: f if parquet_has_time(f) else None, files))
    files = [f for f in meta_valid if f]
    print(f"[INFO] {len(files)} arquivos válidos após filtro de schema e range.")

    dfs = []
    for f in files:
        try:
            # === Leitura flexível ===
            if f.lower().endswith(".zip"):
                with zipfile.ZipFile(f) as z:
                    inner = z.namelist()[0]
                    with z.open(inner) as zf:
                        d = pd.read_csv(io.TextIOWrapper(zf, encoding="utf-8"), header=None)
            elif f.lower().endswith(".csv"):
                d = pd.read_csv(f, header=None)
            else:
                d = pd.read_parquet(f)
                # limpa colunas duplicadas e fragmentos
                d = d.loc[:, ~d.columns.duplicated()]
                drop_cols = [c for c in d.columns if c.startswith("__")]
                if drop_cols:
                    d = d.drop(columns=drop_cols)
                if all(str(c).isdigit() for c in d.columns) and len(d.columns) >= 7:
                    d.columns = ["trade_id","price","qty","first_id","last_id","time","is_buyer_maker"]
                    print(f"[PATCH-AGG] Renomeadas colunas numéricas em {os.path.basename(f)}")

            # === Formato Binance Vision ===
            if d.shape[1] >= 7:
                d = d.iloc[:, :7]
                d.columns = ["trade_id","price","qty","first_id","last_id","time","is_buyer_maker"]
            elif "time" not in d.columns:
                print(f"[ERRO LOAD AGG] {os.path.basename(f)} → CSV sem coluna time")
                continue

            # Corrige timestamps
            d["time"] = pd.to_numeric(d["time"], errors="coerce")
            if d["time"].max() < 1e12:
                d["time"] *= 1000
            d["time"] = d["time"].astype("int64")

            # Flags e assinaturas
            if "is_buyer_maker" in d.columns:
                d["is_buyer_maker"] = (
                    d["is_buyer_maker"]
                    .astype(str).str.lower()
                    .isin(["true", "1", "t", "yes"])
                ).fillna(False)
            else:
                d["is_buyer_maker"] = False

            # Corrige quantidades e assina o volume
            d["qty"] = pd.to_numeric(d["qty"], errors="coerce").fillna(0.0)
            d["signed_qty"] = d["qty"].where(~d["is_buyer_maker"], -d["qty"])

            # === Agregação por barra ===
            d["time_dt"] = pd.to_datetime(d["time"], unit="ms", utc=True)
            rule = _tf_rule(tf)
            g = d.set_index("time_dt").groupby(pd.Grouper(freq=rule, label="right", closed="right"))
            out = pd.DataFrame({
                "time": (g.size().index.astype("int64") // 10**6).astype("int64"),
                "taker_buy_vol": g.apply(lambda x: x.loc[~x["is_buyer_maker"], "qty"].sum() if len(x) else 0.0),
                "taker_sell_vol": g.apply(lambda x: x.loc[x["is_buyer_maker"], "qty"].sum() if len(x) else 0.0),
                "vol_per_bar": g["qty"].sum(),
                "cvd_delta": g["signed_qty"].sum(),
            }).reset_index(drop=True)
            out["cvd"] = out["cvd_delta"].cumsum()

            out = out[(out["time"] >= start_ms) & (out["time"] <= end_ms)]
            if out.empty:
                continue

            dfs.append(out)
            print(f"[LOAD] Agg OK: {os.path.basename(f)} ({len(out)} barras)")

        except Exception as e:
            print(f"[ERRO LOAD AGG] {f} → {e}")

    if not dfs:
        print("[LOAD] Nenhum Agg válido no range final.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True).sort_values("time")
    return _downcast_inplace(df[["time","cvd","taker_buy_vol","taker_sell_vol","vol_per_bar"]])


def load_depthfeat_dir(dir_path: str, tf: str, start_ms: int, end_ms: int, use_field: str = "bd_imb_50bps") -> pd.DataFrame:
    """
    Lê arquivos de Depth da Binance Vision e calcula imbalance médio por barra.

    Compatível com:
      - Histogram: (timestamp, percentage, depth, notional)
      - Bids/Asks em JSON
      - Campos diretos (imb_net_depth / bd_imb_*)
    Agora com filtro rápido por range (YYYY-MM).
    """
    import os, glob, ast, pandas as pd, re

    if not os.path.isdir(dir_path):
        print(f"[WARN] Diretório inexistente: {dir_path}")
        return pd.DataFrame()

    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith((".parquet", ".csv", ".zip"))]
    if not files:
        print(f"[LOAD] Nenhum Depth encontrado em {dir_path}")
        return pd.DataFrame()

    # === Filtro rápido pelo nome e range (YYYY-MM) ===
    start_num = int(pd.to_datetime(start_ms, unit="ms").strftime("%Y%m"))
    end_num   = int(pd.to_datetime(end_ms, unit="ms").strftime("%Y%m"))
    valid_files = []
    for f in files:
        m = re.search(r"(\d{4})-(\d{2})", f)
        if not m:
            valid_files.append(f)
            continue
        ym = int(m.group(1)) * 100 + int(m.group(2))
        if start_num <= ym <= end_num:
            valid_files.append(f)
        else:
            print(f"[SKIP] {os.path.basename(f)} fora do range {start_num}-{end_num}")
    files = sorted(valid_files)

    if not files:
        print(f"[LOAD] Nenhum Depth válido no range {start_num}-{end_num}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            d = d.loc[:, ~d.columns.duplicated()]

            # --- PATCH: garantir que 'timestamp' seja numérico em ms ---
            if "timestamp" in d.columns:
                if np.issubdtype(d["timestamp"].dtype, np.datetime64):
                    d["timestamp"] = (
                        pd.to_datetime(d["timestamp"], utc=True)
                        .astype("int64") // 10**6
                    ).astype("int64")

            # === Caso 1: histogram (timestamp, percentage, depth, notional) ===
            if set(["timestamp", "percentage", "depth"]).issubset(d.columns):
                d["timestamp"] = pd.to_numeric(d["timestamp"], errors="coerce")
                # somas por timestamp sem apply/param include_groups
                mask_pos = d["percentage"] > 0
                mask_neg = d["percentage"] < 0
                pos = pd.to_numeric(d.loc[mask_pos, "depth"], errors="coerce").groupby(d.loc[mask_pos, "timestamp"]).sum()
                neg = pd.to_numeric(d.loc[mask_neg, "depth"], errors="coerce").groupby(d.loc[mask_neg, "timestamp"]).sum()
                ix = pos.index.union(neg.index).sort_values()
                pos = pos.reindex(ix, fill_value=0.0)
                neg = neg.reindex(ix, fill_value=0.0)
                out_df = pd.DataFrame({
                    "time": ix.astype("int64"),
                    "imb_net_depth": (neg.values - pos.values) / (neg.values + pos.values + 1e-9),
                }).dropna()
                dfs.append(out_df)
                print(f"[LOAD] Depth histogram OK: {os.path.basename(f)} ({len(out_df)} barras)")
                continue

            # === Caso 2: bids/asks em listas JSON ===
            bids_col = next((c for c in d.columns if "bids" in c.lower()), None)
            asks_col = next((c for c in d.columns if "asks" in c.lower()), None)
            if bids_col and asks_col:
                def _sum_qty(v):
                    try:
                        arr = ast.literal_eval(str(v))
                        return float(sum(float(x[1]) for x in arr if isinstance(x, (list, tuple)) and len(x) >= 2))
                    except Exception:
                        return 0.0
                d["_b"] = d[bids_col].apply(_sum_qty)
                d["_a"] = d[asks_col].apply(_sum_qty)
                d["imb_net_depth"] = (d["_b"] - d["_a"]) / (d["_b"] + d["_a"] + 1e-9)
                if "time" not in d.columns:
                    d["time"] = pd.NaT
                d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
                d["time"] = (d["time"].astype("int64") // 10**6).astype("int64")
                dfs.append(d[["time", "imb_net_depth"]])
                print(f"[LOAD] Depth bids/asks OK: {os.path.basename(f)} ({len(d)} linhas)")
                continue

            # === Caso 3: campo direto (bd_imb_50bps / imb_net_depth) ===
            if use_field in d.columns or "imb_net_depth" in d.columns:
                col = use_field if use_field in d.columns else "imb_net_depth"
                d["time"] = pd.to_numeric(d.get("time", pd.NaT), errors="coerce")
                if d["time"].max() < 1e12:
                    d["time"] *= 1000
                d["time"] = d["time"].astype("int64")
                dfs.append(d[["time", col]].rename(columns={col: "imb_net_depth"}))
                print(f"[LOAD] Depth field OK: {os.path.basename(f)} ({len(d)} linhas)")
                continue

            print(f"[WARN] {os.path.basename(f)} sem campos reconhecíveis.")

        except Exception as e:
            print(f"[ERRO LOAD DEPTH] {os.path.basename(f)} → {e}")

    # --- Fim do loop for f in files ---
    if not dfs:
        print("[LOAD] Nenhum Depth válido no range final.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True).sort_values("time")

    if df.empty:
        return pd.DataFrame()
    # Resample to requested timeframe and return
    dtime = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="ms", utc=True)
    g = df.assign(time=dtime).set_index("time").resample(_tf_rule(tf), label="right", closed="right")["imb_net_depth"].mean()
    out = g.reset_index()
    out["time"] = (out["time"].astype("int64") // 10**6).astype("int64")
    return _downcast_inplace(out[["time","imb_net_depth"]])


# ============================================================
# Execução principal
# ============================================================


def _make_ml_features_v2(df: pd.DataFrame, *, add_lags: int = 3,
                         include_agg: bool = True, include_depth: bool = True,
                         base_signals_df: Optional[pd.DataFrame] = None,
                         combo_signals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Nova versão que evita fragmentação de DF:
      - monta todas as colunas em dicts e concatena de uma vez;
      - aplica lags em batch;
      - faz limpeza de NaN/inf no final.
    """
    idx = df.index

    cols = {}
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    # Preço/retornos
    ret_1 = close.pct_change(1)
    cols["ret_1"] = ret_1
    cols["ret_5"] = close.pct_change(5)
    cols["ret_20"] = close.pct_change(20)
    cols["ret_vol_20"] = ret_1.rolling(20, min_periods=5).std()

    # ATR relativo
    _a14 = atr(df, 14)
    cols["atr_pct"] = (_a14 / close.abs().clip(lower=1e-9))

    # Tendência clássica
    ef, es = ema(close, 9), ema(close, 21)
    cols["ema_f"] = ef
    cols["ema_s"] = es
    cols["rsi14"] = rsi(close, 14)
    mac = macd_line(close, 12, 26)
    cols["macd"] = mac
    cols["macd_sig"] = ema(mac, 9)
    m, u, l = bollinger_bands(close, 20, 2.0)
    cols["bb_pos"] = (close - m) / (u - l + 1e-9)
    vwap20 = vwap_win(df, 20)
    cols["vwap20"] = vwap20
    cols["dist_ema_f"] = (close - ef) / (ef + 1e-9)
    cols["dist_vwap20"] = (close - vwap20) / (vwap20 + 1e-9)

    # AggTrades
    if include_agg:
        cvd = pd.to_numeric(df.get("cvd", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0)
        cols["cvd"] = cvd
        cols["cvd_slope"] = cvd.diff(3)
        cols["taker_buy_vol"]  = pd.to_numeric(df.get("taker_buy_vol",  0.0), errors="coerce")
        cols["taker_sell_vol"] = pd.to_numeric(df.get("taker_sell_vol", 0.0), errors="coerce")
        cols["vol_per_bar"]    = pd.to_numeric(df.get("vol_per_bar",    0.0), errors="coerce")

    # Depth
    if include_depth:
        cols["imb_net_depth"] = pd.to_numeric(df.get("imb_net_depth", 0.0), errors="coerce")

    # Funding/premium/mark se existirem
    if "fundingRate" in df.columns and "funding_rate" not in cols:
        fr = pd.to_numeric(df["fundingRate"], errors="coerce")
        cols["funding_rate"] = fr.fillna(0.0)
        cols["funding_chg"]  = cols["funding_rate"].diff().fillna(0.0)
    elif "funding_rate" in df.columns:
        fr = pd.to_numeric(df["funding_rate"], errors="coerce")
        cols["funding_rate"] = fr.fillna(0.0)
        cols["funding_chg"]  = cols["funding_rate"].diff().fillna(0.0)

    if "markPrice" in df.columns and "mark_close_ratio" not in cols:
        mp = pd.to_numeric(df["markPrice"], errors="coerce")
        cols["mark_close_ratio"] = (mp / (close.replace(0, np.nan))) - 1.0
    elif "mark_price" in df.columns and "mark_close_ratio" not in cols:
        mp = pd.to_numeric(df["mark_price"], errors="coerce")
        cols["mark_close_ratio"] = (mp / (close.replace(0, np.nan))) - 1.0

    if "premiumIndex" in df.columns and "premium_z" not in cols:
        pr = pd.to_numeric(df["premiumIndex"], errors="coerce")
        cols["premium_z"]  = (pr - pr.rolling(96, min_periods=10).mean()) / (pr.rolling(96, min_periods=10).std() + 1e-9)
        cols["prem_abs"]   = pr.fillna(0.0)
    elif "premium_index" in df.columns and "premium_z" not in cols:
        pr = pd.to_numeric(df["premium_index"], errors="coerce")
        cols["premium_z"]  = (pr - pr.rolling(96, min_periods=10).mean()) / (pr.rolling(96, min_periods=10).std() + 1e-9)
        cols["prem_abs"]   = pr.fillna(0.0)

    # BBO/microprice
    if ("bbo_spread" in df.columns) or ("spread" in df.columns):
        cols["bbo_spread"] = pd.to_numeric(df["bbo_spread"] if "bbo_spread" in df.columns else df["spread"], errors="coerce")
    if ("bbo_mid" in df.columns) or ("mid" in df.columns):
        cols["bbo_mid"] = pd.to_numeric(df["bbo_mid"] if "bbo_mid" in df.columns else df["mid"], errors="coerce")
    if "microprice_imb" in df.columns:
        cols["microprice_imb"] = pd.to_numeric(df["microprice_imb"], errors="coerce")

    # Sinais base/combos
    if base_signals_df is not None and not base_signals_df.empty:
        for c in base_signals_df.columns:
            cols[f"sig_{c}"] = pd.to_numeric(base_signals_df[c].shift(1), errors="coerce")
    if combo_signals_df is not None and not combo_signals_df.empty:
        for c in combo_signals_df.columns:
            cols[f"sigc_{c}"] = pd.to_numeric(combo_signals_df[c].shift(1), errors="coerce")

    X0 = pd.DataFrame(cols, index=idx)

    # Lags em batch
    lag_dict = {}
    for L in range(1, int(add_lags)+1):
        if "funding_rate" in X0.columns:
            lag_dict[f"funding_lag{L}"] = X0["funding_rate"].shift(L)
        if "bbo_spread" in X0.columns:
            lag_dict[f"bbo_spread_lag{L}"] = X0["bbo_spread"].shift(L)
        if "microprice_imb" in X0.columns:
            lag_dict[f"microprice_lag{L}"] = X0["microprice_imb"].shift(L)
        for basecol in ["ret_1", "rsi14", "bb_pos", "dist_ema_f", "dist_vwap20"]:
            if basecol in X0.columns:
                lag_dict[f"{basecol}_lag{L}"] = X0[basecol].shift(L)
        if include_agg and "cvd" in X0.columns:
            lag_dict[f"cvd_lag{L}"] = X0["cvd"].shift(L)
        if include_depth and "imb_net_depth" in X0.columns:
            lag_dict[f"imb_lag{L}"] = X0["imb_net_depth"].shift(L)

    X = X0 if not lag_dict else pd.concat([X0, pd.DataFrame(lag_dict, index=idx)], axis=1)

    X["time"] = pd.to_datetime(df["time"], utc=True)
    X = X.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return _downcast_inplace(X)


if __name__ == "__main__":
    import sys, argparse
    # Execução direta ou com --load_runtime
    if any("--load_runtime" in x for x in sys.argv):
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--load_runtime", required=True)
        args, _ = ap.parse_known_args()
        _load_runtime(args.load_runtime)
    else:
        main()
