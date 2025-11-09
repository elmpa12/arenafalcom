#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import math
import os

def sanitize_nan(v):
    """Remove valores NaN ou infinitos do JSON final"""
    if isinstance(v, dict):
        return {k: sanitize_nan(x) for k, x in v.items() if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))}
    if isinstance(v, list):
        return [sanitize_nan(x) for x in v]
    return v

def main():
    excel_file = "chosenones.xlsx"
    out_file   = "best_params_all_ext.json"

    df = pd.read_excel(excel_file)

    # Normaliza colunas
    df.columns = [c.strip().lower() for c in df.columns]

    # Separa bases e combos
    base_rows  = df[df["is_combo"] == False]
    combo_rows = df[df["is_combo"] == True]

    out = {"version": 1, "timeframes": {}}

    for tf in sorted(df["timeframe"].unique()):
        tf = str(tf).lower()
        out["timeframes"][tf] = []

        # BASE methods
        for _, row in base_rows[base_rows["timeframe"] == tf].iterrows():
            item = {
                "type": "base",
                "method": str(row["method"]),
                "params": json.loads(row["params"]) if isinstance(row["params"], str) else (row["params"] if isinstance(row["params"], dict) else {}),
                "metrics": {
                    "expectancy": float(row.get("expectancy", 0)),
                    "hit": float(row.get("hit", 0)),
                    "sharpe": float(row.get("sharpe", 0)),
                    "n_trades": int(row.get("n_trades", 0)),
                    "total_pnl": float(row.get("total_pnl", 0)),
                    "maxdd": float(row.get("maxdd", 0)),
                    "payoff": float(row.get("payoff", 0)),
                    "score": float(row.get("score", 0)),
                },
                "config": {
                    "timeouts": {
                        "max_hold": int(row.get("max_hold_used", 30)),
                        "atr_len": int(row.get("atr_timeout_len_used", 14)),
                        "atr_mult": float(row.get("atr_timeout_mult_used", 1.5)),
                    },
                    "atr_stop": {
                        "enabled": bool(row.get("use_atr_stop", False)),
                        "len": int(row.get("atr_stop_len_used", 14)),
                        "mult": float(row.get("atr_stop_mult_used", 1.5)),
                    },
                    "atr_tp": {
                        "enabled": bool(row.get("use_atr_tp", False)),
                        "len": int(row.get("atr_tp_len_used", 14)),
                        "mult": float(row.get("atr_tp_mult_used", 2.0)),
                    },
                    "candle_stop": {
                        "enabled": bool(row.get("use_candle_stop", False)),
                        "lookback": int(row.get("candle_stop_lookback", 1)),
                    },
                }
            }
            out["timeframes"][tf].append(sanitize_nan(item))

        # COMBO methods
        for _, row in combo_rows[combo_rows["timeframe"] == tf].iterrows():
            item = {
                "type": "combo",
                "method": str(row["method"]),
                "combo_spec": str(row["combo_spec"]) if "combo_spec" in row and pd.notna(row["combo_spec"]) else None,
                "metrics": {
                    "expectancy": float(row.get("expectancy", 0)),
                    "hit": float(row.get("hit", 0)),
                    "sharpe": float(row.get("sharpe", 0)),
                    "n_trades": int(row.get("n_trades", 0)),
                    "total_pnl": float(row.get("total_pnl", 0)),
                    "maxdd": float(row.get("maxdd", 0)),
                    "payoff": float(row.get("payoff", 0)),
                    "score": float(row.get("score", 0)),
                },
                "config": {
                    "timeouts": {
                        "max_hold": int(row.get("max_hold_used", 30)),
                        "atr_len": int(row.get("atr_timeout_len_used", 14)),
                        "atr_mult": float(row.get("atr_timeout_mult_used", 1.5)),
                    }
                }
            }
            out["timeframes"][tf].append(sanitize_nan(item))

    # Dedup por método+params
    for tf, arr in out["timeframes"].items():
        seen = {}
        deduped = []
        for item in arr:
            key = (item["type"], item.get("method"), json.dumps(item.get("params", {}), sort_keys=True))
            score = float(item.get("metrics", {}).get("score", 0))
            if key not in seen or score > seen[key][0]:
                seen[key] = (score, item)
        deduped = [v[1] for v in seen.values()]
        out["timeframes"][tf] = deduped

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Escrito {out_file} com {sum(len(v) for v in out['timeframes'].values())} variantes.")

if __name__ == "__main__":
    main()
