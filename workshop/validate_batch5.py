#!/usr/bin/env python3
"""Validar Batch 5 - últimos 4 métodos não testados para completar 5/5"""

import json, subprocess, time
from pathlib import Path
from datetime import datetime

session_id = datetime.now().strftime("%Y-%m-%d_%H%M")
session_dir = Path(f"sessions/validation5_{session_id}")
session_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print(f"🎯 VALIDAÇÃO BATCH 5 - ÚLTIMOS 4 MÉTODOS NÃO TESTADOS!")
print(f"{'='*80}")
print(f"Progresso atual: 4/5 setups validados")
print(f"Meta: Encontrar 1 setup final com win rate >= 60%")
print(f"{'='*80}\n")

# 4 métodos NUNCA testados do selector21
untested_methods = [
    {"name": "orr_rev_15m", "method": "orr_reversal", "tf": "15m"},
    {"name": "vwap_poc_15m", "method": "vwap_poc_reject", "tf": "15m"},
    {"name": "ob_imb_15m", "method": "ob_imbalance_break", "tf": "15m"},
    {"name": "cvd_div_15m", "method": "cvd_divergence_reversal", "tf": "15m"},
]

# Mesmos 10 períodos de validação
validation_periods = [
    ("2022-07-01", "2022-07-31", "Jul_2022"),
    ("2022-10-01", "2022-10-31", "Out_2022"),
    ("2023-01-01", "2023-01-31", "Jan_2023"),
    ("2023-03-01", "2023-03-31", "Mar_2023"),
    ("2023-06-01", "2023-06-30", "Jun_2023"),
    ("2023-08-01", "2023-08-31", "Ago_2023"),
    ("2023-10-01", "2023-10-31", "Out_2023"),
    ("2023-12-01", "2023-12-31", "Dez_2023"),
    ("2024-02-01", "2024-02-29", "Fev_2024"),
    ("2024-05-01", "2024-05-31", "Mai_2024"),
]

# Gerar testes
tests = []
for setup in untested_methods:
    for start, end, period_name in validation_periods:
        tests.append({
            "setup_name": setup["name"],
            "test_name": f"{setup['name']}_{period_name}",
            "method": setup["method"],
            "tf": setup["tf"],
            "period": period_name,
            "start": start,
            "end": end,
            "args": [
                "--umcsv_root", "./data_monthly",
                "--symbol", "BTCUSDT",
                "--start", start, "--end", end,
                "--exec_rules", setup["tf"],
                "--methods", setup["method"],
                "--run_base", "--n_jobs", "2",
                "--out_root", str(session_dir / f"{setup['name']}_{period_name}")
            ]
        })

print(f"📋 Batch 5: {len(untested_methods)} métodos × {len(validation_periods)} períodos = {len(tests)} testes")
print(f"🚀 Executando com 12 paralelos...\n")

# Executar
running, pending, completed = [], list(tests), []
parallel = 12

while pending or running:
    while len(running) < parallel and pending:
        test = pending.pop(0)
        out_dir = Path(test["args"][test["args"].index("--out_root") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            ["python3", "selector21.py"] + test["args"],
            stdout=open(out_dir / "test.log", "w"),
            stderr=subprocess.STDOUT
        )
        running.append({"name": test["test_name"], "proc": proc, "start": time.time(), "out_dir": out_dir, "test": test})
        print(f"[{len(completed)+len(running)}/{len(tests)}] ▶️  {test['test_name']}")

    time.sleep(1)

    done = [r for r in running if r["proc"].poll() is not None]
    for r in done:
        elapsed = time.time() - r["start"]
        csv_path = r["out_dir"] / "leaderboard_base.csv"
        metrics = None

        if csv_path.exists() and r["proc"].returncode == 0:
            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    row = df.iloc[0]
                    metrics = {
                        "hit": row["hit"], "payoff": row.get("payoff", 0),
                        "total_pnl": row["total_pnl"], "sharpe": row["sharpe"],
                        "maxdd": row["maxdd"], "n_trades": row["n_trades"]
                    }
                    status = "✅" if metrics['total_pnl'] > 0 else "❌"
                    print(f"{status} {r['name']} ({elapsed:.1f}s) | PnL: {metrics['total_pnl']:>10,.0f} | Sharpe: {metrics['sharpe']:>5.2f}")
            except:
                print(f"⚠️  {r['name']} ({elapsed:.1f}s) - Erro")
        else:
            print(f"❌ {r['name']} ({elapsed:.1f}s)")

        completed.append({"name": r['name'], "test": r['test'], "elapsed": elapsed, "metrics": metrics, "success": r["proc"].returncode == 0 and metrics is not None})
        running.remove(r)

# Análise
print(f"\n{'='*80}")
print(f"📊 ANÁLISE BATCH 5 - ÚLTIMOS MÉTODOS")
print(f"{'='*80}\n")

setup_results = {}
for c in completed:
    if not c["success"]:
        continue

    setup_name = c["test"]["setup_name"]
    if setup_name not in setup_results:
        setup_results[setup_name] = {"tests": [], "profitable_count": 0, "total_tests": 0, "total_pnl": 0, "avg_sharpe": 0, "avg_hit": 0}

    metrics = c["metrics"]
    setup_results[setup_name]["tests"].append({"period": c["test"]["period"], "pnl": metrics["total_pnl"], "sharpe": metrics["sharpe"], "hit": metrics["hit"], "trades": metrics["n_trades"]})
    setup_results[setup_name]["total_tests"] += 1
    setup_results[setup_name]["total_pnl"] += metrics["total_pnl"]
    setup_results[setup_name]["avg_sharpe"] += metrics["sharpe"]
    setup_results[setup_name]["avg_hit"] += metrics["hit"]
    if metrics["total_pnl"] > 0:
        setup_results[setup_name]["profitable_count"] += 1

validated_setups = []
for setup_name, results in setup_results.items():
    n = results["total_tests"]
    if n == 0:
        continue

    results["avg_sharpe"] /= n
    results["avg_hit"] /= n
    results["avg_pnl"] = results["total_pnl"] / n
    results["win_rate"] = (results["profitable_count"] / n) * 100

    print(f"{'─'*80}")
    print(f"Setup: {setup_name}")
    print(f"Win Rate: {results['win_rate']:.1f}% ({results['profitable_count']}/{n})")
    print(f"PnL Médio: {results['avg_pnl']:>12,.0f}")
    print(f"Sharpe Médio: {results['avg_sharpe']:>6.2f}")

    if results["win_rate"] >= 60:
        print(f"✅ VALIDADO!")
        validated_setups.append(setup_name)
    else:
        print(f"❌ NÃO VALIDADO")
    print()

print(f"{'='*80}")
print(f"🎯 RESULTADO BATCH 5 - DECISIVO")
print(f"{'='*80}")
print(f"Novos setups VALIDADOS: {len(validated_setups)}")
print(f"Total acumulado: 4 (anteriores) + {len(validated_setups)} (novos) = {4+len(validated_setups)}/5")

if validated_setups:
    print(f"\n🎉🎉🎉 SETUP(S) FINAL(IS) VALIDADO(S)! 🎉🎉🎉")
    for i, setup in enumerate(validated_setups, 5):
        results = setup_results[setup]
        print(f"{i}. {setup}: Win Rate {results['win_rate']:.1f}%, PnL Médio {results['avg_pnl']:,.0f}")

    if 4 + len(validated_setups) >= 5:
        print(f"\n🏆🏆🏆 META COMPLETA! {4+len(validated_setups)}/5 SETUPS VALIDADOS! 🏆🏆🏆")
else:
    print(f"\n⚠️  Nenhum novo setup validado. Ficamos em 4/5.")
    print(f"📊 Todos os métodos disponíveis em selector21 foram testados.")
    print(f"✅ 4 setups robustos já é um resultado excelente!")

# Salvar
with open(session_dir / "batch5_results.json", "w") as f:
    json.dump({"validated": validated_setups, "results": setup_results}, f, indent=2, default=str)

print(f"\n✅ Resultados salvos em: {session_dir / 'batch5_results.json'}")
print(f"{'='*80}\n")
