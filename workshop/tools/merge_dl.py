#!/usr/bin/env python3
"""
Merge selector outputs with deep-learning artefacts into a single report.json.

Usage example:
    python tools/merge_dl.py \
        --selector-report resultados/run42/selection_report.json \
        --runtime-config resultados/run42/runtime_config.json \
        --dl-metrics out/run42/dl/metrics_summary.json \
        --dl-preds out/run42/dl/MERGED_meta_5m.csv \
        --out resultados/run42/report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def summarise_prediction_file(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"file": str(path)}
    if not path.exists():
        summary["error"] = "missing"
        return summary
    try:
        df = pd.read_csv(path, parse_dates=["time"], infer_datetime_format=True)
    except Exception as exc:
        summary["error"] = f"read_error: {exc}"
        return summary

    summary["rows"] = int(len(df))
    if "time" in df.columns and not df.empty:
        summary["time_start"] = df["time"].min().isoformat()
        summary["time_end"] = df["time"].max().isoformat()
    if "model" in df.columns:
        summary["models"] = sorted(set(df["model"].dropna().astype(str)))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unifica artefatos Selector ↔ DL em report.json.")
    parser.add_argument("--selector-report", required=True, help="selection_report.json do selector.")
    parser.add_argument("--runtime-config", help="runtime_config.json opcional.")
    parser.add_argument(
        "--dl-metrics",
        nargs="+",
        required=True,
        help="Arquivos JSON com métricas do DL (um ou mais).",
    )
    parser.add_argument(
        "--dl-preds",
        nargs="*",
        default=[],
        help="CSV(s) com probabilidades / merges (serão sumarizados).",
    )
    parser.add_argument("--out", required=True, help="Arquivo final (report.json).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selector_report = load_json(Path(args.selector_report))
    runtime_cfg = load_json(Path(args.runtime_config)) if args.runtime_config else None

    metrics_payload: List[Any] = []
    for path in args.dl_metrics:
        metrics_payload.append(load_json(Path(path)))

    preds_summary = [summarise_prediction_file(Path(p)) for p in args.dl_preds]

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "selector": selector_report,
        "runtime_config": runtime_cfg,
        "deep_learning": {
            "metrics": metrics_payload if len(metrics_payload) > 1 else metrics_payload[0],
            "predictions": preds_summary,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    print(f"[merge_dl] Report salvo em {args.out}")


if __name__ == "__main__":
    main()
