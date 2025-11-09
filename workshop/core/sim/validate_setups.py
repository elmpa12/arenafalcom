#!/usr/bin/env python3
"""
VALIDAÇÃO RIGOROSA DE SETUPS
Meta: 5 setups que LUCRAM CONSISTENTEMENTE em múltiplos períodos
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

session_id = datetime.now().strftime("%Y-%m-%d_%H%M")
session_dir = Path(f"sessions/validation_{session_id}")
session_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print(f"🎯 VALIDAÇÃO RIGOROSA DE SETUPS")
print(f"{'='*80}")
print(f"META: Encontrar 5 setups que lucram CONSISTENTEMENTE")
print(f"Critério: Lucrar em 60%+ dos períodos testados")
print(f"{'='*80}\n")

# SETUPS PROMISSORES (descobertos anteriormente)
candidate_setups = [
    {
        "name": "ema_crossover_15d",
        "method": "ema_crossover",
        "tf": "5m",
        "reason": "Lucrou +261K em Mar/2023 com payoff 5.83x"
    },
    {
        "name": "vwap_trend_30d_15m",
        "method": "vwap_trend",
        "tf": "15m",
        "reason": "Lucrou +141K em Out/2023 com Sharpe 0.41"
    },
    {
        "name": "macd_trend_15m",
        "method": "macd_trend",
        "tf": "15m",
        "reason": "Melhor em Gen3: +277K, Sharpe 0.84"
    },
    {
        "name": "trend_breakout_15m",
        "method": "trend_breakout",
        "tf": "15m",
        "reason": "Alto hit em Gen3: 67.9%, +194K, Sharpe 1.13"
    },
]

# 10 PERÍODOS DIFERENTES para validação (meses variados)
validation_periods = [
    ("2022-07-01", "2022-07-31", "Jul_2022"),
    ("2022-10-01", "2022-10-31", "Out_2022"),
    ("2023-01-01", "2023-01-31", "Jan_2023"),
    ("2023-03-01", "2023-03-31", "Mar_2023"),  # Onde ema_crossover lucrou
    ("2023-06-01", "2023-06-30", "Jun_2023"),
    ("2023-08-01", "2023-08-31", "Ago_2023"),
    ("2023-10-01", "2023-10-31", "Out_2023"),  # Onde vwap lucrou
    ("2023-12-01", "2023-12-31", "Dez_2023"),
    ("2024-02-01", "2024-02-29", "Fev_2024"),
    ("2024-05-01", "2024-05-31", "Mai_2024"),
]

# Gerar testes: cada setup × 10 períodos = 40 testes
tests = []

for setup in candidate_setups:
    for start, end, period_name in validation_periods:
        test = {
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
                "--start", start,
                "--end", end,
                "--exec_rules", setup["tf"],
                "--methods", setup["method"],
                "--run_base",
                "--n_jobs", "2",
                "--out_root", str(session_dir / f"{setup['name']}_{period_name}")
            ]
        }
        tests.append(test)

print(f"📋 Validação configurada:")
print(f"   {len(candidate_setups)} setups candidatos")
print(f"   {len(validation_periods)} períodos de teste")
print(f"   {len(tests)} testes totais ({len(candidate_setups)} × {len(validation_periods)})")
print(f"\n🚀 Executando com 10 paralelos...\n")

# Executar testes
running = []
pending = list(tests)
completed = []

parallel = 10

while pending or running:
    while len(running) < parallel and pending:
        test = pending.pop(0)

        out_dir = Path(test["args"][test["args"].index("--out_root") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)

        log_file = out_dir / "test.log"

        proc = subprocess.Popen(
            ["python3", "-m", "core.selectors.selector21"] + test["args"],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT
        )

        running.append({
            "name": test["test_name"],
            "proc": proc,
            "start": time.time(),
            "out_dir": out_dir,
            "test": test
        })

        print(f"[{len(completed)+len(running)}/{len(tests)}] ▶️  {test['test_name']}")

    time.sleep(1)

    done = [r for r in running if r["proc"].poll() is not None]
    for r in done:
        elapsed = time.time() - r["start"]

        # Ler métricas
        csv_path = r["out_dir"] / "leaderboard_base.csv"
        metrics = None

        if csv_path.exists() and r["proc"].returncode == 0:
            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    row = df.iloc[0]
                    metrics = {
                        "hit": row["hit"],
                        "payoff": row.get("payoff", 0),
                        "total_pnl": row["total_pnl"],
                        "sharpe": row["sharpe"],
                        "maxdd": row["maxdd"],
                        "n_trades": row["n_trades"]
                    }

                    status = "✅" if metrics['total_pnl'] > 0 else "❌"

                    print(f"{status} {r['name']} ({elapsed:.1f}s) | PnL: {metrics['total_pnl']:>10,.0f} | Sharpe: {metrics['sharpe']:>5.2f}")

            except Exception as e:
                print(f"⚠️  {r['name']} ({elapsed:.1f}s) - Erro ao ler: {e}")
        else:
            print(f"❌ {r['name']} ({elapsed:.1f}s) - Falhou")

        completed.append({
            "name": r['name'],
            "test": r['test'],
            "elapsed": elapsed,
            "metrics": metrics,
            "success": r["proc"].returncode == 0 and metrics is not None
        })

        running.remove(r)

# ANÁLISE DE VALIDAÇÃO
print(f"\n{'='*80}")
print(f"📊 ANÁLISE DE VALIDAÇÃO")
print(f"{'='*80}\n")

# Agrupar por setup
setup_results = {}
for c in completed:
    if not c["success"]:
        continue

    setup_name = c["test"]["setup_name"]

    if setup_name not in setup_results:
        setup_results[setup_name] = {
            "tests": [],
            "profitable_count": 0,
            "total_tests": 0,
            "total_pnl": 0,
            "avg_sharpe": 0,
            "avg_hit": 0
        }

    metrics = c["metrics"]
    setup_results[setup_name]["tests"].append({
        "period": c["test"]["period"],
        "pnl": metrics["total_pnl"],
        "sharpe": metrics["sharpe"],
        "hit": metrics["hit"],
        "trades": metrics["n_trades"]
    })

    setup_results[setup_name]["total_tests"] += 1
    setup_results[setup_name]["total_pnl"] += metrics["total_pnl"]
    setup_results[setup_name]["avg_sharpe"] += metrics["sharpe"]
    setup_results[setup_name]["avg_hit"] += metrics["hit"]

    if metrics["total_pnl"] > 0:
        setup_results[setup_name]["profitable_count"] += 1

# Calcular médias e taxa de sucesso
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
    print(f"{'─'*80}")
    print(f"Testado em: {n} períodos")
    print(f"Lucrativos: {results['profitable_count']}/{n} ({results['win_rate']:.1f}%)")
    print(f"PnL Total: {results['total_pnl']:>12,.0f}")
    print(f"PnL Médio: {results['avg_pnl']:>12,.0f}")
    print(f"Sharpe Médio: {results['avg_sharpe']:>6.2f}")
    print(f"Hit Médio: {results['avg_hit']:>5.2%}")

    # CRITÉRIO DE VALIDAÇÃO: 60%+ de win rate
    if results["win_rate"] >= 60:
        print(f"✅ VALIDADO! Win rate {results['win_rate']:.1f}% >= 60%")
        validated_setups.append(setup_name)
    else:
        print(f"❌ NÃO VALIDADO. Win rate {results['win_rate']:.1f}% < 60%")

    # Mostrar detalhes por período
    print(f"\nDetalhes por período:")
    for t in sorted(results["tests"], key=lambda x: x["pnl"], reverse=True):
        status = "🟢" if t["pnl"] > 0 else "🔴"
        print(f"  {status} {t['period']:12s}: PnL {t['pnl']:>10,.0f} | Sharpe {t['sharpe']:>5.2f} | Hit {t['hit']:>5.2%} | Trades {t['trades']}")
    print()

# RESULTADO FINAL
print(f"{'='*80}")
print(f"🎯 RESULTADO FINAL")
print(f"{'='*80}")
print(f"Setups testados: {len(candidate_setups)}")
print(f"Setups VALIDADOS: {len(validated_setups)}")
print(f"\nMETA: 5 setups consistentes")
print(f"ALCANÇADO: {len(validated_setups)}/5")

if validated_setups:
    print(f"\n✅ SETUPS VALIDADOS (lucram em 60%+ dos períodos):")
    for i, setup in enumerate(validated_setups, 1):
        results = setup_results[setup]
        print(f"{i}. {setup}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   PnL Médio: {results['avg_pnl']:,.0f}")
        print(f"   Sharpe Médio: {results['avg_sharpe']:.2f}")
else:
    print(f"\n❌ Nenhum setup passou na validação (60%+ win rate)")

# Salvar resultados
with open(session_dir / "validation_results.json", "w") as f:
    json.dump({
        "session": session_id,
        "candidate_setups": candidate_setups,
        "validation_periods": validation_periods,
        "setup_results": setup_results,
        "validated_setups": validated_setups,
        "target": "5 setups consistentes",
        "achieved": len(validated_setups)
    }, f, indent=2, default=str)

print(f"\n✅ Resultados salvos em: {session_dir / 'validation_results.json'}")
print(f"{'='*80}\n")
