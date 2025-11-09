#!/usr/bin/env python3
"""
replay_merged.py — Converte MERGED_meta_*.csv (probabilidades calibradas) em PnL/trades simulados (paper).
Entrada: glob para MERGED_meta_*.csv + configs de execução (trading.json)
Saída: CSV de trades e métricas agregadas (compatível com reports).
"""
from pathlib import Path
import argparse
import json
import pandas as pd


def load_config(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate_from_csv(csv_path: Path, cfg: dict) -> dict:
    # TODO: Alfred implementa — aplicar threshold/neutral_band, fees, slippage, tick,
    # stops/timeout/max_hold. Retornar dict com pnl, n_trades, winrate, pf, sharpe, mdd, etc.
    return {"csv": str(csv_path), "pnl": 0.0, "n_trades": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-glob", required=True, help="glob p/ MERGED_meta_*.csv")
    ap.add_argument("--config", default="configs/trading.json", help="config de paper")
    ap.add_argument("--out-metrics", default="reports/replay_metrics.csv")
    ap.add_argument("--out-trades", default="reports/replay_trades.csv")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    files = list(Path(".").glob(args.merged_glob))
    rows = []
    all_trades = []

    for f in files:
        res = simulate_from_csv(f, cfg)
        rows.append(res)
        # all_trades.extend(... cada trade como dict ...)

    pd.DataFrame(rows).to_csv(args.out_metrics, index=False)
    # pd.DataFrame(all_trades).to_csv(args.out_trades, index=False)
    print(f"[replay] metrics → {args.out_metrics}")


if __name__ == "__main__":
    main()
