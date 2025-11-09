#!/usr/bin/env python3
"""
Generate sequential datasets (train/validation NPZ) per timeframe/window.

This tool runs on the CPU side right after selector21 finishes. It reuses the
shared enrichment pipeline to materialise features, builds lagged sequences and
splits them according to a WFO schedule (monthly or weekly).  Each window
produces:

    <out>/<tf>/winXX/seq_train.npz
    <out>/<tf>/winXX/seq_val.npz
    <out>/<tf>/winXX/meta.json

The manifest file (<out>/manifest.json) summarises all generated artefacts so
that the GPU step can rsync only what is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.dl.seq_utils import (
    build_sequences_chunked,
    make_label,
    monthly_windows,
    number_cols,
    weekly_windows,
)
from core.selectors.data_pipeline import EnrichmentRequest, enrich_from_request


def _utc_ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 10**6)


def _log(msg: str) -> None:
    print(f"[make_seq] {msg}", flush=True)


def _ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df
    # best effort fallback
    for candidate in ("open_time", "timestamp", "datetime", "time_ms", "timestamp_ms"):
        if candidate in df.columns:
            df["time"] = pd.to_datetime(df[candidate], utc=True, errors="coerce")
            return df
    raise RuntimeError("Frame sem coluna de tempo (esperado 'time').")


def build_sequence_bundle(
    symbol: str,
    tf: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_root: str,
    horizon: int,
    lags: int,
) -> Dict[str, object]:
    request = EnrichmentRequest(
        symbol=symbol,
        timeframe=tf,
        start_ms=_to_ms(start),
        end_ms=_to_ms(end),
        root_dir=data_root,
    )
    frame = enrich_from_request(request)
    if frame.empty:
        raise RuntimeError("Pipeline retornou dataframe vazio.")
    frame = _ensure_time_column(frame)
    frame = frame.sort_values("time").dropna(subset=["time", "close"]).reset_index(drop=True)

    ret, y = make_label(frame, horizon)
    frame["ret_future"] = ret
    frame["y"] = y
    feats = number_cols(frame)
    if not feats:
        raise RuntimeError("Nenhuma feature numérica encontrada para construir sequências.")

    X, Y, R, idx = build_sequences_chunked(frame, feats, lags, horizon)
    if len(X) == 0:
        raise RuntimeError("Sem amostras após construir sequências (verifique dados).")
    times = frame.loc[idx, "time"].reset_index(drop=True)
    return {"frame": frame, "features": feats, "X": X, "Y": Y, "R": R, "T": times}


def save_npz(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    ret: np.ndarray,
    time_index: pd.Series,
    price: pd.Series | None = None,
) -> None:
    if time_index.dt.tz is None:
        timestamps = time_index.dt.tz_localize("UTC")
    else:
        timestamps = time_index.dt.tz_convert("UTC")
    t_ms = (timestamps.view("int64") // 1_000_000).astype("int64")
    payload = dict(
        X=X.astype(np.float32, copy=False),
        y=y.astype(np.float32, copy=False),
        ret=ret.astype(np.float32, copy=False),
        time=t_ms,
    )
    if price is not None:
        payload["price"] = price.reset_index(drop=True).astype(np.float32, copy=False).to_numpy()
    np.savez_compressed(path, **payload)


def windows_for_mode(
    mode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    train_months: float,
    val_months: float,
    step_months: float,
    train_weeks: int,
    val_weeks: int,
    step_weeks: int,
) -> List[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if mode == "weekly":
        return weekly_windows(start, end, train_weeks, val_weeks, step_weeks)
    return monthly_windows(
        start,
        end,
        train_months=train_months,
        val_months=val_months,
        step_months=step_months,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera seq_train/seq_val para DL (NPZ).")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tf", "--timeframes", dest="timeframes", required=True,
                        help="Lista separada por vírgula (ex: 1m,5m,15m)")
    parser.add_argument("--start", required=True, help="ISO timestamp (UTC).")
    parser.add_argument("--end", required=True, help="ISO timestamp (UTC).")
    parser.add_argument("--data-root", default="./data", help="Raiz com dados enriquecidos.")
    parser.add_argument("--lags", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--mode", choices=["monthly", "weekly"], default="monthly")
    parser.add_argument("--train-months", type=float, default=0.5)
    parser.add_argument("--val-months", type=float, default=0.25)
    parser.add_argument("--step-months", type=float, default=0.25)
    parser.add_argument("--train-weeks", type=int, default=2)
    parser.add_argument("--val-weeks", type=int, default=1)
    parser.add_argument("--step-weeks", type=int, default=1)
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--min-val", type=int, default=50)
    parser.add_argument("--out", required=True, help="Diretório de saída.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    if not timeframes:
        raise SystemExit("Informe ao menos um timeframe em --tf.")

    start_ts = _utc_ts(args.start)
    end_ts = _utc_ts(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "symbol": args.symbol,
        "range": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
        "lags": args.lags,
        "horizon": args.horizon,
        "timeframes": {},
    }

    for tf in timeframes:
        _log(f"TF {tf}: enriquecendo dados…")
        tf_entry = {"windows": []}
        try:
            bundle = build_sequence_bundle(
                args.symbol, tf, start_ts, end_ts, args.data_root, args.horizon, args.lags
            )
        except Exception as exc:
            _log(f"⚠️  Falha ao montar sequências para {tf}: {exc}")
            manifest["timeframes"][tf] = {**tf_entry, "error": str(exc)}
            continue

        feats_file = out_dir / f"{tf}_features.json"
        if not feats_file.exists():
            feats_file.write_text(
                json.dumps(
                    {
                        "symbol": args.symbol,
                        "timeframe": tf,
                        "features": bundle["features"],
                        "lags": args.lags,
                        "horizon": args.horizon,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        tf_entry["features_file"] = str(feats_file)

        windows = windows_for_mode(
            args.mode,
            start_ts,
            end_ts,
            train_months=args.train_months,
            val_months=args.val_months,
            step_months=args.step_months,
            train_weeks=args.train_weeks,
            val_weeks=args.val_weeks,
            step_weeks=args.step_weeks,
        )
        if not windows:
            _log(f"⚠️  Nenhum WFO window gerado para {tf}.")
            manifest["timeframes"][tf] = {**tf_entry, "error": "no_windows"}
            continue

        X = bundle["X"]; Y = bundle["Y"]; R = bundle["R"]; T = bundle["T"]
        # price series opcional para auxiliar métricas/merge no estágio DL
        try:
            P = bundle["frame"].loc[bundle["T"].index, "close"]
        except Exception:
            P = None
        tf_base = out_dir / tf
        tf_base.mkdir(parents=True, exist_ok=True)

        usable = 0
        for idx, (tr_s, tr_e, va_s, va_e) in enumerate(windows, 1):
            mask_tr = ((T >= tr_s) & (T <= tr_e)).to_numpy()
            mask_va = ((T >= va_s) & (T <= va_e)).to_numpy()
            n_tr = int(mask_tr.sum())
            n_va = int(mask_va.sum())

            if n_tr < args.min_train or n_va < args.min_val:
                _log(
                    f"TF {tf} win{idx:02d}: amostras insuficientes "
                    f"(train={n_tr}, val={n_va}) – ignorado."
                )
                continue

            win_dir = tf_base / f"win{idx:02d}"
            win_dir.mkdir(parents=True, exist_ok=True)

            save_npz(win_dir / "seq_train.npz", X[mask_tr], Y[mask_tr], R[mask_tr], T[mask_tr], price=P[mask_tr] if P is not None else None)
            save_npz(win_dir / "seq_val.npz", X[mask_va], Y[mask_va], R[mask_va], T[mask_va], price=P[mask_va] if P is not None else None)

            meta = {
                "timeframe": tf,
                "window": idx,
                "train_range": {"start": tr_s.isoformat(), "end": tr_e.isoformat()},
                "val_range": {"start": va_s.isoformat(), "end": va_e.isoformat()},
                "train_samples": n_tr,
                "val_samples": n_va,
                "lags": args.lags,
                "horizon": args.horizon,
            }
            (win_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            tf_entry["windows"].append(
                {
                    **meta,
                    "train_file": str(win_dir / "seq_train.npz"),
                    "val_file": str(win_dir / "seq_val.npz"),
                }
            )
            usable += 1

        if usable == 0:
            tf_entry["warning"] = "no_windows_with_min_samples"
            _log(f"⚠️  TF {tf}: nenhum window válido atingiu os thresholds.")
        manifest["timeframes"][tf] = tf_entry

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"Manifesto salvo em {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
