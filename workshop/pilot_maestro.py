#!/usr/bin/env python3
"""
PILOT MAESTRO - Teste piloto com 10 micro-backtests
Valida arquitetura antes de escalar para 500
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_pilot():
    """Run pilot with 10 micro-backtests"""

    session_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    session_dir = Path(f"sessions/pilot_{session_id}")
    session_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "tf": "5m",
        "total_tests": 10,  # Pilot: 10 tests
        "parallel": 5,      # 5 parallel

        "targets": {
            "hit": 0.52,
            "payoff": 1.25,
            "maxdd": -2000
        }
    }

    print(f"\n{'='*80}")
    print(f"🎭 MAESTRO PILOT SESSION")
    print(f"{'='*80}")
    print(f"Session: pilot_{session_id}")
    print(f"Tests: {config['total_tests']}")
    print(f"Parallel: {config['parallel']}")
    print(f"{'='*80}\n")

    # Generate 10 test configurations
    tests = []
    methods = ["macd_trend", "ema_crossover", "trend_breakout", "vwap_trend", "rsi_reversion",
               "ema_pullback", "orr_reversal", "boll_breakout", "pivot_reversion", "donchian_breakout"]

    for i in range(config["total_tests"]):
        test_config = {
            "name": f"pilot_test{i:02d}",
            "args": [
                "--umcsv_root", "./data_monthly",
                "--symbol", "BTCUSDT",
                "--start", "2024-01-01",
                "--end", "2024-01-15",  # 15 days micro-backtest
                "--exec_rules", config["tf"],
                "--methods", methods[i],
                "--run_base",
                "--n_jobs", "2",
                "--out_root", str(session_dir / f"pilot_test{i:02d}")
            ]
        }
        tests.append(test_config)

    # Save config
    with open(session_dir / "pilot_config.json", "w") as f:
        json.dump({"config": config, "tests": tests}, f, indent=2)

    # Execute tests in parallel
    running = []
    pending = list(tests)
    completed = []

    while pending or running:
        # Start new tests up to parallel limit
        while len(running) < config["parallel"] and pending:
            test = pending.pop(0)

            # Create output directory
            out_dir = Path(test["args"][test["args"].index("--out_root") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)

            log_file = out_dir / "test.log"

            # Start process
            proc = subprocess.Popen(
                ["python3", "selector21.py"] + test["args"],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT
            )

            running.append({
                "name": test["name"],
                "proc": proc,
                "start": time.time(),
                "out_dir": out_dir
            })

            print(f"[{len(completed)+len(running)}/{config['total_tests']}] ▶️  {test['name']}")

        # Check for completed tests
        time.sleep(1)

        done = [r for r in running if r["proc"].poll() is not None]
        for r in done:
            elapsed = time.time() - r["start"]
            status = "✅" if r["proc"].returncode == 0 else "❌"

            # Read results
            csv_path = r["out_dir"] / "leaderboard_base.csv"
            metrics = read_metrics(csv_path) if csv_path.exists() else None

            result = {
                "name": r["name"],
                "elapsed": elapsed,
                "success": r["proc"].returncode == 0,
                "metrics": metrics
            }

            completed.append(result)
            running.remove(r)

            print(f"{status} {r['name']} ({elapsed:.1f}s)")

            if metrics:
                print(f"    PnL: {metrics['total_pnl']:>10,.0f} | Sharpe: {metrics['sharpe']:>6.2f} | Hit: {metrics['hit']:>5.2%}")

    # Analyze results
    print(f"\n{'='*80}")
    print(f"📊 PILOT RESULTS")
    print(f"{'='*80}")

    successful = [r for r in completed if r["success"] and r["metrics"]]
    profitable = [r for r in successful if r["metrics"]["total_pnl"] > 0]

    print(f"Completed: {len(successful)}/{config['total_tests']}")
    print(f"Profitable: {len(profitable)} ({len(profitable)/len(successful)*100:.1f}%)")

    if successful:
        avg_pnl = sum(r["metrics"]["total_pnl"] for r in successful) / len(successful)
        avg_sharpe = sum(r["metrics"]["sharpe"] for r in successful) / len(successful)
        avg_hit = sum(r["metrics"]["hit"] for r in successful) / len(successful)

        print(f"\nAverage Metrics:")
        print(f"  PnL: {avg_pnl:,.2f}")
        print(f"  Sharpe: {avg_sharpe:.4f}")
        print(f"  Hit Rate: {avg_hit:.2%}")

        # Targets
        hit_met = avg_hit >= config["targets"]["hit"]
        print(f"\nTargets:")
        print(f"  {'✅' if hit_met else '❌'} Hit >= {config['targets']['hit']:.2%}: {avg_hit:.2%}")

    if profitable:
        print(f"\nTop 5 Profitable:")
        top5 = sorted(profitable, key=lambda x: x["metrics"]["total_pnl"], reverse=True)[:5]
        for r in top5:
            m = r["metrics"]
            print(f"  {r['name']}: PnL={m['total_pnl']:>10,.0f}, Sharpe={m['sharpe']:>6.2f}, Hit={m['hit']:>5.2%}")

    # Save summary
    summary = {
        "session": f"pilot_{session_id}",
        "completed": len(successful),
        "profitable": len(profitable),
        "profitable_pct": len(profitable)/len(successful)*100 if successful else 0,
        "results": completed
    }

    with open(session_dir / "pilot_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✅ Pilot summary saved to: {session_dir / 'pilot_summary.json'}")
    print(f"{'='*80}\n")

    return summary

def read_metrics(csv_path: Path):
    """Read metrics from leaderboard CSV"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            row = df.iloc[0]
            return {
                "hit": row["hit"],
                "payoff": row.get("payoff", 0),
                "total_pnl": row["total_pnl"],
                "sharpe": row["sharpe"],
                "maxdd": row["maxdd"],
                "n_trades": row["n_trades"]
            }
    except Exception as e:
        return None

    return None

if __name__ == "__main__":
    run_pilot()
