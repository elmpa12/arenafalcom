#!/usr/bin/env python3
"""
make_seq_synthetic.py — Gera NPZs sintéticos para smoke de pipeline.

Uso:
  python tools/make_seq_synthetic.py --out out/seq/5m/win01 \
    --tf 5m --lags 64 --feats 16 --n-tr 500 --n-va 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def save_npz(path: Path, X: np.ndarray, y: np.ndarray, ret: np.ndarray, t: pd.DatetimeIndex, price: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t_ms = (t.view("int64") // 1_000_000).astype("int64")
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.float32), ret=ret.astype(np.float32), time=t_ms, price=price.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Diretório base da janela (ex: out/seq/5m/win01)")
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--lags", type=int, default=64)
    ap.add_argument("--feats", type=int, default=16)
    ap.add_argument("--n-tr", type=int, default=500)
    ap.add_argument("--n-va", type=int, default=200)
    args = ap.parse_args()

    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)

    def gen(n: int):
        X = np.random.normal(size=(n, args.lags, args.feats)).astype(np.float32)
        y = (np.random.rand(n) > 0.5).astype(np.float32)
        ret = np.random.normal(loc=0.0, scale=0.01, size=(n,)).astype(np.float32)
        # timestamps: 5m cadenciado
        t0 = pd.Timestamp("2024-01-01T00:00:00Z")
        t = pd.date_range(t0, periods=n, freq=args.tf, tz="UTC")
        price = 50000 + np.cumsum(ret) * 1000
        return X, y, ret, t, price

    Xtr, ytr, rtr, ttr, ptr = gen(args.n_tr)
    Xva, yva, rva, tva, pva = gen(args.n_va)

    save_npz(base / "seq_train.npz", Xtr, ytr, rtr, ttr, ptr)
    save_npz(base / "seq_val.npz", Xva, yva, rva, tva, pva)
    print(f"[synthetic] seq_train/seq_val gerados em {base}")


if __name__ == "__main__":
    main()
