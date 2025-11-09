#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dl_heads_v8.py — Deep Heads + Merger (Selector ↔ DL) • v8.1 ultra-optimized

Highlights
- GPU: AMP + torch.compile + TF32 + cudnn.benchmark + (fused AdamW quando disponível)
- DataLoader turbo (workers auto, prefetch 8, pin_memory, persistent_workers)
- Sequências CHUNKED (stream) para reduzir pico de RAM no host
- Batch dinâmico com fallback em OOM (8192→6144→4096→3072→2048→1024)
- Logging rico: NVML + torch.mem + throughput; JSONL por época e CSV final
- WFO mensal com frações (train/val/step floats), ou semanal (tw/vw/sw)
- Merger com Selector (LR/LS) + isotonic opcional; artefatos prontos p/ live

Requisitos: numpy, pandas, torch (CUDA opcional), sklearn (opcional; fallback LS).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Compatibilidade: expõe a nova API OO do módulo ``dl_head``.
from dl_head import DLHeadConfig, DeepLearningHead  # noqa: F401
from heads import available_head_names, instantiate_head, resolve_requested_heads
# evita fragmentação entre heads compilados
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
# desativa compilação persistente de grafos no torch.compile (evita leak)
os.environ["TORCHINDUCTOR_FALLBACK_FORCE_EAGER"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.data_pipeline import EnrichmentRequest, enrich_from_request

# ================= GPU & Determinism =================
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.1+
except Exception:
    pass

_HAS_SK = _HAS_NVML = False
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    _HAS_SK = True
except Exception:
    pass

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    pass

def gutil():
    if not _HAS_NVML:
        return None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {"gpu": util.gpu, "mem": int(100*mem.used/max(1,mem.total))}
    except Exception:
        return None

def set_seeds(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def log(m): print(m, flush=True)

def log_json(path: str, obj: dict):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

# -------------------- Dados / enrich --------------------
def _parse_date(s: str) -> pd.Timestamp: return pd.Timestamp(s, tz="UTC")
def _to_ms(dt: pd.Timestamp) -> int: return int(dt.value // 10**6)

def load_premerged(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "time" not in df.columns: raise RuntimeError("Parquet sem 'time'")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)

def _augment_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features leves específicas para os heads DL."""
    if df is None or df.empty:
        return df

    base = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    # OFI proxy
    try:
        mid = None
        for candidate in ("mid", "bbo_mid", "mid_price", "close"):
            if candidate in base.columns:
                mid = pd.to_numeric(base[candidate], errors="coerce")
                break
        if "microprice_imb" in base.columns and mid is not None:
            micro = pd.to_numeric(base["microprice_imb"], errors="coerce").fillna(0.0)
            dmid = mid.diff().fillna(0.0)
            base["ofi_proxy"] = (dmid * micro).astype(np.float32)
    except Exception as exc:
        log(f"[ENRICH][DL] ofi_proxy: {exc}")

    # VHF proxies
    def _vhf(close: pd.Series, length: int) -> pd.Series:
        series = pd.to_numeric(close, errors="coerce")
        high = series.rolling(length, 1).max()
        low = series.rolling(length, 1).min()
        num = (high - low).abs()
        den = series.diff().abs().rolling(length, 1).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            val = (num / den.replace(0, np.nan)).fillna(0.0)
        return val.astype(np.float32)

    try:
        base["vhf_20"] = _vhf(base["close"], 20)
        base["vhf_40"] = _vhf(base["close"], 40)
    except Exception as exc:
        log(f"[ENRICH][DL] vhf: {exc}")

    # ATR z-score fallback
    try:
        if "atr_zscore" not in base.columns:
            high = base["high"] - base["low"]
            prev = base["close"].shift()
            tr = pd.concat(
                [
                    high,
                    (base["high"] - prev).abs(),
                    (base["low"] - prev).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr14 = tr.rolling(14, 1).mean()
            atr100 = tr.rolling(100, 1).mean()
            base["atr_zscore"] = (atr14 / (atr100 + 1e-9)).bfill().astype(np.float32)
    except Exception as exc:
        log(f"[ENRICH][DL] atr_zscore: {exc}")

    # L2 assist leve
    try:
        def _ema(series: pd.Series, span: int) -> pd.Series:
            series = pd.to_numeric(series, errors="coerce")
            return series.ewm(span=span, adjust=False, min_periods=1).mean().astype(np.float32)

        candidates = [
            "imb_net_depth",
            "bd_imb_50bps",
            "bd_imb_25bps",
            "ask_size_sum",
            "bid_size_sum",
            "depth_imbalance",
        ]
        for col in candidates:
            if col in base.columns:
                series = pd.to_numeric(base[col], errors="coerce")
                base[f"{col}_ema20"] = _ema(series, 20)
                base[f"{col}_ema60"] = _ema(series, 60)
                base[f"{col}_z20"] = (
                    (series - series.rolling(20, 1).mean())
                    / (series.rolling(20, 1).std() + 1e-9)
                ).astype(np.float32)
        if "ask_size_sum" in base.columns and "bid_size_sum" in base.columns:
            bid = pd.to_numeric(base["bid_size_sum"], errors="coerce")
            ask = pd.to_numeric(base["ask_size_sum"], errors="coerce")
            base["l2_press"] = ((bid - ask) / (bid + ask + 1e-9)).astype(np.float32)
            base["l2_press_ema"] = _ema(base["l2_press"], 30)
    except Exception as exc:
        log(f"[ENRICH][DL] l2assist: {exc}")

    return base.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_enriched(v2_path: str, symbol: str, tf: str, start: str, end: str, data_dir: str) -> pd.DataFrame:
    if v2_path:
        log(f"[INFO] Ignorando v2_path={v2_path}; usando pipeline compartilhado")
    start_ms = _to_ms(_parse_date(start))
    end_ms = _to_ms(_parse_date(end))
    request = EnrichmentRequest(
        symbol=symbol,
        timeframe=tf,
        start_ms=start_ms,
        end_ms=end_ms,
        root_dir=data_dir,
    )
    df = enrich_from_request(request)
    df = _augment_enrichment(df)
    if "time" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "time"})
        else:
            raise RuntimeError("DF sem 'time'")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)

# -------------------- Seq/labels (CHUNKED) --------------------
def make_label(df: pd.DataFrame, horizon: int) -> Tuple[pd.Series, pd.Series]:
    ret=(df["close"].shift(-horizon)/df["close"]-1.0).astype(np.float32)
    y=(ret>0).astype(np.float32)
    return ret,y

def number_cols(df: pd.DataFrame) -> List[str]:
    drop={"time"}; return [c for c in df.columns if c not in drop and np.issubdtype(np.array(df[c]).dtype, np.number)]

def build_sequences_chunked(
    df: pd.DataFrame, feats: List[str], lags: int, horizon: int, chunk_rows: int = 250_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monta X/Y em chunks para reduzir pico de RAM."""
    A = df[feats].astype(np.float32).to_numpy()
    N, F = A.shape
    close = df["close"].to_numpy()
    Xs, Ys, Rs, IDX = [], [], [], []
    start = lags - 1
    end = N - horizon
    if end <= start:
        return (np.zeros((0, lags, F), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int64))
    i = start
    while i < end:
        j = min(end, i + chunk_rows)
        lens = j - i
        # retorno futuro alinhado 1:1
        rchunk = (close[i + horizon:j + horizon] / close[i:j] - 1.0).astype(np.float32)
        ychunk = (rchunk > 0).astype(np.float32)
        # constrói blocos 3D
        Xchunk = np.lib.stride_tricks.sliding_window_view(
            A[i - lags + 1:j + 1], window_shape=(lags, F)
        )[:, 0, :, :]
        idxchunk = np.arange(i, j, dtype=np.int64)
        # --- CORREÇÃO: corta o excedente pra alinhar ---
        n = min(len(rchunk), Xchunk.shape[0])
        Xs.append(Xchunk[:n]); Ys.append(ychunk[:n]); Rs.append(rchunk[:n]); IDX.append(idxchunk[:n])
        i = j


    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    R = np.concatenate(Rs, axis=0)
    I = np.concatenate(IDX, axis=0)
    return X, Y, R, I

class SeqDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
        self.X = X
        self.y = y
        self.w = w if w is not None else np.ones_like(y, dtype=np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i], self.w[i]

# -------------------- Métricas --------------------
def brier(p,y): p=np.clip(p,1e-6,1-1e-6); return float(np.mean((p-y)**2))
def acc(p,y,thr=0.5): return float(np.mean((p>=thr)==(y>0.5)))
def auc(p,y):
    if not _HAS_SK: return None
    try: return float(roc_auc_score(y,p))
    except Exception: return None
def pr_auc(p,y):
    if not _HAS_SK:
        # manual AP
        order=np.argsort(-p); y=y[order]; p=p[order]
        tp=np.cumsum(y); fp=np.cumsum(1-y)
        prec = tp/np.maximum(1, tp+fp); rec = tp/np.maximum(1, tp[-1] if tp.size else 1)
        ap=0.0
        for i in range(1,len(rec)):
            ap += prec[i] * max(0.0, rec[i]-rec[i-1])
        return float(ap)
    try:
        return float(average_precision_score(y,p))
    except Exception:
        pr, rc, _ = precision_recall_curve(y,p)
        return float(np.trapz(pr, rc))

# -------------------- Thresholds --------------------
def thr_grid_parse(s: str) -> np.ndarray:
    lo, hi, step = [float(x) for x in s.split(",")]
    grid=np.arange(lo, hi+1e-12, step, dtype=np.float64)
    return np.clip(grid,1e-6,1-1e-6)

def trades_per_day(p: np.ndarray, t: pd.Series, thr: float, neutral_band: float=0.0) -> float:
    p=np.clip(p,1e-6,1-1e-6)
    long_mask = (p >= thr) & (p >= (0.5 + neutral_band))
    short_mask= (p <= (1.0 - thr)) & (p <= (0.5 - neutral_band))
    trig = (long_mask | short_mask).astype(int)
    d = pd.Series(trig, index=t).groupby(t.dt.date).sum().values
    return float(np.mean(d)) if len(d)>0 else 0.0

def find_thr_for_trades(p: np.ndarray, t: pd.Series, target: float, grid: np.ndarray, neutral_band: float=0.0) -> float:
    vals=[(thr, trades_per_day(p,t,thr,neutral_band)) for thr in grid]
    if not vals: return 0.5
    thr, _ = min(vals, key=lambda kv: abs(kv[1]-target))
    return float(thr)

# -------------------- Treino --------------------
class EarlyStop:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience=patience; self.min_delta=min_delta; self.best=np.inf; self.count=0
    def step(self, val):
        if val < self.best - self.min_delta: self.best=val; self.count=0; return False
        self.count+=1; return self.count>self.patience

def _adamw(params, lr):
    try:
        return torch.optim.AdamW(params, lr=lr, fused=True)  # PyTorch 2.3+
    except Exception:
        return torch.optim.AdamW(params, lr=lr)

def _safe_compile(model: nn.Module):
    try:
        return torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        return model

def _dataloader(ds, batch, device_is_cuda=True, workers_auto=True, prefetch=8, shuffle=True):
    if workers_auto:
        try:
            cpu = os.cpu_count() or 8
            nw = min(16, max(4, cpu//2))
        except Exception:
            nw = 8
    else:
        nw = 16
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=nw,
        pin_memory=device_is_cuda, persistent_workers=True, prefetch_factor=prefetch
    )

def train_one_head(
    name: str,
    model: nn.Module,
    Xtr,
    ytr,
    wtr,
    Xva,
    yva,
    wva,
    device: str = "cuda",
    epochs: int = 20,
    batch_try=(8192, 6144, 4096, 3072, 2048, 1024),
    lr: float = 1e-3,
    clip: float = 1.0,
    out_dir=None,
    jsonl_path=None,
):
    """Treina um head sequencial com fallback seguro p/ Windows."""

    import multiprocessing as mp

    cuda_enabled = device == "cuda" and torch.cuda.is_available()

    if os.name == "nt":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        use_workers = 0
        pin = False
        persistent = False
    else:
        cpu = os.cpu_count() or 8
        use_workers = min(16, max(4, cpu // 2))
        pin = cuda_enabled
        persistent = True

    model = _safe_compile(model.to(device))

    bce = nn.BCEWithLogitsLoss(reduction="none")
    opt = _adamw(model.parameters(), lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    es = EarlyStop(patience=4, min_delta=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=cuda_enabled)

    ds_tr, ds_va = SeqDS(Xtr, ytr, wtr), SeqDS(Xva, yva, wva)
    last_good_batch = None

    for batch in batch_try:
        try:
            dl_test = DataLoader(ds_tr, batch_size=batch, num_workers=0)
            xb, _, _ = next(iter(dl_test))
            with torch.cuda.amp.autocast(enabled=cuda_enabled):
                _ = model(xb.to(device, non_blocking=cuda_enabled))
            last_good_batch = batch
            break
        except RuntimeError as exc:
            if cuda_enabled and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                continue
            raise
    if last_good_batch is None:
        raise RuntimeError("VRAM insuficiente para qualquer batch.")

    loader_kwargs = dict(
        batch_size=last_good_batch,
        num_workers=use_workers,
        pin_memory=pin,
    )
    if use_workers > 0:
        loader_kwargs["persistent_workers"] = persistent
        loader_kwargs["prefetch_factor"] = 4

    dl_tr = DataLoader(ds_tr, shuffle=True, **loader_kwargs)
    dl_va = DataLoader(ds_va, shuffle=False, **loader_kwargs)

    best, best_b = None, np.inf

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for xb, yb, wb in dl_tr:
            xb = xb.to(device, non_blocking=cuda_enabled)
            yb = yb.to(device, non_blocking=cuda_enabled)
            wb = wb.to(device, non_blocking=cuda_enabled)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cuda_enabled):
                logits = model(xb).float()
                loss = (bce(logits, yb) * wb).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            scaler.step(opt)
            scaler.update()

        model.eval()
        ps, ys = [], []
        with torch.inference_mode():
            for xb, yb, _ in dl_va:
                xb = xb.to(device, non_blocking=cuda_enabled)
                with torch.cuda.amp.autocast(enabled=cuda_enabled):
                    logits = model(xb).float()
                ps.append(torch.sigmoid(logits).cpu())
                ys.append(yb)
        p_va = torch.cat(ps).numpy() if ps else np.zeros((0,), np.float32)
        y_va = torch.cat(ys).numpy() if ys else np.zeros((0,), np.float32)
        b = float(np.mean((np.clip(p_va, 1e-6, 1 - 1e-6) - y_va) ** 2))
        a = float(np.mean((p_va >= 0.5) == (y_va > 0.5)))
        sch.step(b)
        if b < best_b:
            best_b = b
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stop = es.step(b)

        util = gutil() if cuda_enabled else {"gpu": None, "mem": None}
        if util is None:
            util = {"gpu": None, "mem": None}
        mem_alloc = torch.cuda.memory_allocated() / 1024 ** 2 if cuda_enabled else 0.0
        dt = time.time() - t0
        sps = int(len(ds_tr) / dt) if dt > 0 else 0
        msg = {
            "epoch": ep,
            "head": name,
            "brier": b,
            "acc": a,
            "lr": float(opt.param_groups[0]["lr"]),
            "gpu_util": util["gpu"],
            "gpu_mem%": util["mem"],
            "gpu_mem_alloc_mb": mem_alloc,
            "sps": sps,
        }
        log(
            f"[EPOCH] {name} ep={ep}/{epochs} | brier={b:.6f} acc={a:.3f} "
            f"lr={opt.param_groups[0]['lr']:.2e} gpu={util['gpu']}% mem={util['mem']}% "
            f"alloc={mem_alloc:.0f}MB sps={sps}"
        )
        if jsonl_path:
            log_json(jsonl_path, {"ts": time.time(), **msg})
        if stop:
            log(f"[EARLY] {name} best_brier={best_b:.6f}")
            break

    if best:
        model.load_state_dict(best)
    if cuda_enabled:
        torch.cuda.synchronize()
    return model


def _cleanup_after_head() -> None:
    """Libera memória após treinar um head, respeitando CPU-only."""

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    gc.collect()


# -------------------- WFO Calendars --------------------
def weekly_windows(start, end, tw, vw, sw):
    S=pd.Timestamp(start, tz="UTC"); E=pd.Timestamp(end, tz="UTC"); out=[]; cur=S
    while True:
        tr_end = cur + pd.Timedelta(weeks=tw) - pd.Timedelta(minutes=1)
        va_end = tr_end + pd.Timedelta(weeks=vw)
        if tr_end>=E or cur>=E: break
        out.append((cur,tr_end,tr_end+pd.Timedelta(minutes=1), min(va_end,E)))
        cur = cur + pd.Timedelta(weeks=sw)
        if cur>=E: break
    return out

def _month_split_points(month_start: pd.Timestamp, k: int) -> List[pd.Timestamp]:
    month_end = (month_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(minutes=1)
    days = (month_end - month_start + pd.Timedelta(minutes=1)).days
    pts=[month_start]
    for i in range(1,k):
        d=int(round(days*i/k))
        pts.append(month_start + pd.Timedelta(days=d))
    pts.append(month_end + pd.Timedelta(minutes=1))  # limite superior exclusivo
    return pts

def monthly_windows(start, end, train_months=0.5, val_months=0.25, step_months=0.25):
    S=pd.Timestamp(start, tz="UTC"); E=pd.Timestamp(end, tz="UTC")
    k = int(round(1/step_months))  # 1,2,4
    if k not in (1,2,4): k=2
    cur = pd.Timestamp(year=S.year, month=S.month, day=1, tz="UTC")
    if cur<S: cur = cur + pd.offsets.MonthBegin(1)
    edges=[]
    while cur < E:
        pts=_month_split_points(cur, k)
        edges += pts[:-1]  # sem o limite superior
        cur = cur + pd.offsets.MonthBegin(1)
    edges.append(pd.Timestamp(year=E.year, month=E.month, day=1, tz="UTC") + pd.offsets.MonthBegin(1))
    tr_steps = max(1, int(round(train_months/step_months)))
    va_steps = max(1, int(round(val_months/step_months)))
    out=[]
    i=0
    while True:
        tr_start_idx=i
        tr_end_idx=i+tr_steps
        va_end_idx=tr_end_idx+va_steps
        if va_end_idx >= len(edges): break
        tr_s=edges[tr_start_idx]
        tr_e=edges[tr_end_idx]-pd.Timedelta(minutes=1)
        va_s=edges[tr_end_idx]
        va_e=min(edges[va_end_idx]-pd.Timedelta(minutes=1), E)
        if tr_s>=E: break
        out.append((tr_s, tr_e, va_s, va_e))
        i += 1
    return out

# -------------------- Merger com Selector --------------------
def preferred_selector_prob(df_list: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not df_list:
        return None
    def score(name: str) -> int:
        n=name.lower()
        if "stack" in n or "ensemble" in n: return 0
        if "xgb" in n: return 1
        return 2
    ranked=sorted(df_list, key=lambda d: score(getattr(d, "_src_name", "unknown")))
    best=ranked[0].copy()
    return best.rename(columns={"p":"p_selector"})[["time","price","p_selector"]]

def merge_selector_dl(selector_csv_glob: str, dl_csv: str, out_csv: str, extra_feats_csv: str|None=None,
                      calibrate: bool=True, neutral_band: float=0.0):
    import glob
    dl=pd.read_csv(dl_csv, parse_dates=["time"])
    cand=[]
    for fp in sorted(glob.glob(selector_csv_glob)):
        try:
            d=pd.read_csv(fp, parse_dates=["time"])
            d._src_name=os.path.basename(fp)
            cand.append(d)
        except Exception:
            pass
    sel=preferred_selector_prob(cand)
    if sel is None:
        raise RuntimeError("Nenhum CSV de selector encontrado pelo glob.")
    merged=pd.merge_asof(dl.sort_values("time"), sel.sort_values("time"), on="time", direction="nearest", tolerance=pd.Timedelta("30s"))
    if extra_feats_csv and os.path.isfile(extra_feats_csv):
        ef=pd.read_csv(extra_feats_csv, parse_dates=["time"])
        merged=pd.merge_asof(merged.sort_values("time"), ef.sort_values("time"), on="time", direction="nearest", tolerance=pd.Timedelta("30s"))
    pdl = np.clip(merged["p"].to_numpy(),1e-6,1-1e-6)
    y   = merged.get("y", pd.Series(np.zeros(len(merged)))).to_numpy().astype(np.float32)
    feats=["p"]
    if "p_selector" in merged.columns: feats.append("p_selector")
    if "atr_zscore" in merged.columns: feats.append("atr_zscore")
    if "vhf_20" in merged.columns: feats.append("vhf_20")
    X=merged[feats].fillna(0.0).to_numpy()

    # scaler
    scaler=None
    if _HAS_SK:
        scaler=StandardScaler().fit(X); Xn=scaler.transform(X)
    else:
        mu=X.mean(0, keepdims=True); sd=X.std(0, keepdims=True)+1e-6; Xn=(X-mu)/sd
        scaler=(mu,sd)

    # meta model
    if _HAS_SK:
        clf=LogisticRegression(max_iter=1000).fit(Xn,y)
        p_meta=clf.predict_proba(Xn)[:,1]
    else:
        Z=np.column_stack([np.ones(len(Xn)), Xn])
        w,_res,_rank,_s=np.linalg.lstsq(Z, y.astype(np.float64), rcond=None)
        p_meta=np.clip(Z@w,1e-6,1-1e-6)
        clf=("LS", w.tolist())

    if calibrate and _HAS_SK:
        try:
            ir=IsotonicRegression(out_of_bounds="clip"); p_meta=ir.fit_transform(p_meta, y)
            clf=("LR+ISO", clf)
        except Exception:
            pass
    merged["p_meta"]=np.clip(p_meta,1e-6,1-1e-6)
    merged.to_csv(out_csv, index=False)

    art_dir=os.path.dirname(out_csv)
    with open(os.path.join(art_dir,"meta_model.pkl"),"wb") as f: pickle.dump(clf,f)
    with open(os.path.join(art_dir,"meta_scaler.pkl"),"wb") as f: pickle.dump(scaler,f)
    thr={"long":0.65, "short":0.35, "neutral_band": float(neutral_band)}
    with open(os.path.join(art_dir,"thresholds.json"),"w") as f: json.dump(thr,f,indent=2)
    metrics={"brier": brier(merged["p_meta"].to_numpy(), y)}
    with open(os.path.join(art_dir,"meta_metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    return out_csv

# -------------------- Main --------------------
def main():
    ap=argparse.ArgumentParser()
    # Dados
    ap.add_argument("--data_file", default="")
    ap.add_argument("--v2_path", default=""); ap.add_argument("--data_dir", default=""); ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", required=True, choices=["1m","5m","15m"]); ap.add_argument("--start", default=""); ap.add_argument("--end", default="")
    ap.add_argument("--out", required=True)
    # Modelos
    ap.add_argument("--models", default="gru,lstm,cnn,transformer,dense")
    ap.add_argument("--horizon", type=int, default=3); ap.add_argument("--lags", type=int, default=128)
    # WFO
    ap.add_argument("--wfo_mode", default="monthly", choices=["weekly","monthly"])
    ap.add_argument("--wfo_train_months", type=float, default=0.5)
    ap.add_argument("--wfo_val_months", type=float, default=0.25)
    ap.add_argument("--wfo_step_months", type=float, default=0.25)
    ap.add_argument("--train_weeks", type=int, default=2); ap.add_argument("--val_weeks", type=int, default=1); ap.add_argument("--step_weeks", type=int, default=1)
    # Treino / infra
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--num_workers", type=int, default=-1); ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ckpt_every", type=int, default=0, help="Se >0, salva checkpoint a cada N épocas (por head)")
    # Calibração e thresholds
    ap.add_argument("--calibrate", default="auto", choices=["off","auto","isotonic","platt"])
    ap.add_argument("--thr_grid", default="0.40,0.60,0.01")
    ap.add_argument("--target_trades_per_day", type=float, default=0.0)
    ap.add_argument("--neutral_band", type=float, default=0.0)
    # Merger
    ap.add_argument("--selector_glob", default="", help="Glob p/ CSVs de probs do selector (ex: /runs/selector/oos_probs_*_5m_win*.csv)")
    ap.add_argument("--extra_feats_csv", default="", help="CSV extra com features (time, atr_zscore, vhf_20 etc.)")
    args=ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seeds(args.seed)
    device = "cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu")
    log(f"[DLv8] device={device} | tf={args.tf} | models={args.models} | WFO={args.wfo_mode} | out={args.out}")

    # Dados
    if args.data_file:
        df=load_premerged(args.data_file)
    else:
        if not (args.v2_path and args.data_dir and args.start and args.end):
            raise SystemExit("Forneça --data_file OU ( --v2_path + --data_dir + --start + --end ).")
        df=load_enriched(args.v2_path, args.symbol, args.tf, args.start, args.end, args.data_dir)

    # Labels/feats
    log("[STAGE] MAKE_LABELS")
    ret,y = make_label(df, args.horizon); df["ret_future"]=ret; df["y"]=y
    feats=number_cols(df)
    log(f"[STAGE] BUILD_SEQ (lags={args.lags}, feats={len(feats)})")
    X, Y, R, IDX = build_sequences_chunked(df, feats, args.lags, args.horizon, chunk_rows=250_000)
    if len(X)==0: raise RuntimeError("Sem amostras após BUILD_SEQ")
    T = df.loc[IDX,"time"].reset_index(drop=True); P=df.loc[IDX,"close"].reset_index(drop=True)

    # WFO
    if args.wfo_mode=="weekly":
        W = weekly_windows(args.start or str(T.iloc[0]), args.end or str(T.iloc[-1]),
                           args.train_weeks, args.val_weeks, args.step_weeks)
    else:
        W = monthly_windows(args.start or str(T.iloc[0]), args.end or str(T.iloc[-1]),
                            train_months=args.wfo_train_months, val_months=args.wfo_val_months, step_months=args.wfo_step_months)
    log(f"[STAGE] WINDOWS n={len(W)}")

    try:
        requested_heads = resolve_requested_heads(args.models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not requested_heads:
        allowed = ", ".join(sorted(available_head_names()))
        raise SystemExit(f"Nenhum head válido informado. Use um de: {allowed}")
    log(f"[STAGE] HEADS {','.join(h.name for h in requested_heads)}")
    thr_grid = thr_grid_parse(args.thr_grid)

    leaderboard = []  # agregação final

    for wi,(tr_s,tr_e,va_s,va_e) in enumerate(W,1):
        mask_tr=(T>=tr_s)&(T<=tr_e); mask_va=(T>=va_s)&(T<=va_e)
        Xtr,Ytr = X[mask_tr], Y[mask_tr]
        Xva,Yva = X[mask_va], Y[mask_va]
        Tva = T[mask_va].reset_index(drop=True); Pva=P[mask_va].reset_index(drop=True)
        if len(Xtr)<200 or len(Xva)<50:
            log(f"[WIN{wi:02d}] pouca amostra (tr={len(Xtr)} va={len(Xva)}). skip."); continue

        # Normalização por janela
        mu=Xtr.mean(axis=(0,1), keepdims=True); sd=Xtr.std(axis=(0,1), keepdims=True)+1e-6
        Xtr_n=(Xtr-mu)/sd; Xva_n=(Xva-mu)/sd
        Wtr=np.ones_like(Ytr, dtype=np.float32); Wva=np.ones_like(Yva, dtype=np.float32)

        # ===== HEADS =====
        preds = {}
        jsonl = os.path.join(args.out, f"train_log_win{wi:02d}.jsonl")
        for head_def in requested_heads:
            mdl = instantiate_head(head_def.name, Xtr.shape[-1], Xtr.shape[1])
            mdl = train_one_head(
                head_def.display_name,
                mdl,
                Xtr_n,
                Ytr,
                Wtr,
                Xva_n,
                Yva,
                Wva,
                device=device,
                epochs=args.epochs,
                batch_try=(args.batch, 6144, 4096, 3072, 2048, 1024),
                lr=1e-3,
                clip=1.0,
                out_dir=args.out,
                jsonl_path=jsonl,
            )
            with torch.inference_mode():
                tensor = torch.from_numpy(Xva_n).to(
                    device, non_blocking=(device == "cuda" and torch.cuda.is_available())
                )
                p = torch.sigmoid(mdl(tensor)).cpu().numpy()
            preds[head_def.name] = p
            del mdl
            _cleanup_after_head()

        # --- Métricas, calibração e thresholds por trades/dia ---
        def ece(p,y,bins=10):
            p=np.clip(p,1e-6,1-1e-6); y=y.astype(np.float32)
            idx = np.minimum((p*bins).astype(int), bins-1)
            e=0.0; N=len(p)
            for b in range(bins):
                m=(idx==b)
                if not np.any(m): continue
                conf=float(p[m].mean()); 
                e += abs(float(y[m].mean())-conf)* (np.sum(m)/N)
            return float(e)

        thr_sel = {}
        if args.target_trades_per_day>0:
            for k,p in preds.items():
                thr = find_thr_for_trades(p, Tva, args.target_trades_per_day, thr_grid, neutral_band=args.neutral_band)
                thr_sel[k]=thr

        # salvar OOS por head + métricas
        for k,p in preds.items():
            outcsv=os.path.join(args.out, f"oos_probs_{k}_{args.tf}_win{wi:02d}.csv")
            p = np.clip(p,1e-6,1-1e-6)
            pd.DataFrame({"time":Tva,"price":Pva,"p":p,"y":Yva}).to_csv(outcsv, index=False)
            met = {
                "brier": brier(p,Yva),
                "acc":   acc(p,Yva),
                "auc":   auc(p,Yva),
                "pr_auc": pr_auc(p,Yva),
                "ece":   ece(p,Yva, bins=10),
                "n_val": int(len(Yva)),
                "thr_target_trades_per_day": thr_sel.get(k, None),
                "win": wi, "model": k
            }
            with open(os.path.join(args.out, f"metrics_{k}_{args.tf}_win{wi:02d}.json"),"w") as f: json.dump(met,f,indent=2)
            leaderboard.append(met)
            log(f"[OOS] {k} → {outcsv} | {met}")

        # checkpoint opcional
        if args.ckpt_every and (wi % max(1,args.ckpt_every) == 0):
            try:
                torch.save({"mu":mu, "sd":sd, "feats":feats},
                           os.path.join(args.out, f"ckpt_meta_{args.tf}_win{wi:02d}.pt"))
            except Exception:
                pass

    # Leaderboard final
    if leaderboard:
        df_ld=pd.DataFrame(leaderboard)
        df_ld.to_csv(os.path.join(args.out, "DL_LEADERBOARD.csv"), index=False)
        by_model=df_ld.groupby("model").agg({"brier":"mean","acc":"mean","auc":"mean","pr_auc":"mean","ece":"mean","n_val":"sum"}).reset_index()
        by_model.sort_values(["brier","auc"], ascending=[True, False]).to_csv(os.path.join(args.out,"DL_SUMMARY.csv"), index=False)
        log("[SUMMARY]")
        for _,r in by_model.sort_values(["brier","auc"], ascending=[True, False]).iterrows():
            log(f"{r['model']}: brier={r['brier']:.4f} auc={r['auc']:.3f} acc={r['acc']:.3f} pr_auc={r['pr_auc']:.3f} n={int(r['n_val'])}")

    # Merger automático (opcional)
    if args.selector_glob:
        try:
            best = df_ld.sort_values(["brier","auc"], ascending=[True, False]).iloc[0]["model"]
            cand = sorted([p for p in os.listdir(args.out) if p.startswith(f"oos_probs_{best}_{args.tf}_")]) or [None]
            if cand[0] is None: raise RuntimeError("Sem oos_probs para DL.")
            dl_csv = os.path.join(args.out, cand[-1])
            out_csv = os.path.join(args.out, f"MERGED_meta_{args.tf}.csv")
            merge_selector_dl(args.selector_glob, dl_csv, out_csv, extra_feats_csv=(args.extra_feats_csv or None),
                              calibrate=(args.calibrate!="off"), neutral_band=args.neutral_band)
            log(f"[MERGE] Selecionado DL head='{best}'. Merge salvo em: {out_csv}")
        except Exception as e:
            log(f"[MERGE][WARN] {e}")

    # limpeza final
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

if __name__=="__main__":
    main()
