#!/usr/bin/env python3
"""
EXECUÇÃO PARALELA MASSIVA - BotScalp V3

Aproveita os 64 cores e 128GB RAM para rodar MÚLTIPLOS backtests simultaneamente.

Uso:
    python3 run_parallel_backtests.py --num_parallel 12
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# CONFIGURAÇÕES DE TESTES (aproveitando recursos)
TESTS = [
    {
        "name": "test1_1m_quick",
        "desc": "1m apenas, 3 métodos, 2 meses",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-03-01",
            "--interval", "auto",
            "--exec_rules", "1m",
            "--methods", "trend_breakout,rsi_reversion,ema_crossover",
            "--run_base",
            "--loader_verbose",
            "--n_jobs", "4",  # 4 cores por teste
            "--par_backend", "process",
            "--out_root", "./resultados/test1",
            "--print_top10"
        ]
    },
    {
        "name": "test2_5m_quick",
        "desc": "5m apenas, 3 métodos, 2 meses",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-03-01",
            "--interval", "auto",
            "--exec_rules", "5m",
            "--methods", "macd_trend,vwap_trend,boll_breakout",
            "--run_base",
            "--loader_verbose",
            "--n_jobs", "4",
            "--par_backend", "process",
            "--out_root", "./resultados/test2",
            "--print_top10"
        ]
    },
    {
        "name": "test3_15m_quick",
        "desc": "15m apenas, 3 métodos, 2 meses",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-03-01",
            "--interval", "auto",
            "--exec_rules", "15m",
            "--methods", "orb_breakout,ema_pullback,donchian_breakout",
            "--run_base",
            "--loader_verbose",
            "--n_jobs", "4",
            "--par_backend", "process",
            "--out_root", "./resultados/test3",
            "--print_top10"
        ]
    },
    {
        "name": "test4_multi_tf",
        "desc": "Multi-timeframe, 6 métodos, 2 meses",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-03-01",
            "--interval", "auto",
            "--exec_rules", "1m,5m,15m",
            "--methods", "trend_breakout,rsi_reversion,macd_trend,vwap_trend,boll_breakout,ema_crossover",
            "--run_base",
            "--loader_verbose",
            "--n_jobs", "6",
            "--par_backend", "process",
            "--out_root", "./resultados/test4",
            "--print_top10"
        ]
    },
    {
        "name": "test5_combos_small",
        "desc": "1m + combos (cap 20), 4 métodos, 2 meses",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-03-01",
            "--interval", "auto",
            "--exec_rules", "1m",
            "--methods", "trend_breakout,rsi_reversion,ema_crossover,macd_trend",
            "--run_base",
            "--run_combos",
            "--combo_ops", "AND,MAJ",
            "--combo_cap", "20",
            "--loader_verbose",
            "--n_jobs", "6",
            "--par_backend", "process",
            "--out_root", "./resultados/test5",
            "--print_top10"
        ]
    },
    {
        "name": "test6_wf_simple",
        "desc": "Walk-forward 1m, 3 métodos, 4 meses",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-05-01",
            "--interval", "auto",
            "--exec_rules", "1m",
            "--methods", "trend_breakout,rsi_reversion,ema_crossover",
            "--run_base",
            "--walkforward",
            "--wf_train_months", "2",
            "--wf_val_months", "1",
            "--wf_step_months", "1",
            "--wf_grid_mode", "light",
            "--loader_verbose",
            "--n_jobs", "6",
            "--par_backend", "process",
            "--out_root", "./resultados/test6",
            "--print_top10"
        ]
    },
    {
        "name": "test7_all_methods_1m",
        "desc": "TODOS os métodos, 1m, 2 meses",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-03-01",
            "--interval", "auto",
            "--exec_rules", "1m",
            "--methods", "trend_breakout,keltner_breakout,rsi_reversion,ema_crossover,macd_trend,vwap_trend,boll_breakout,orb_breakout,orr_reversal,ema_pullback,donchian_breakout,vwap_poc_reject,ob_imbalance_break,cvd_divergence_reversal",
            "--run_base",
            "--loader_verbose",
            "--n_jobs", "8",
            "--par_backend", "process",
            "--out_root", "./resultados/test7",
            "--print_top10"
        ]
    },
    {
        "name": "test8_period_q1",
        "desc": "Q1 2024, multi-tf, 6 métodos",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-04-01",
            "--interval", "auto",
            "--exec_rules", "1m,5m,15m",
            "--methods", "trend_breakout,rsi_reversion,macd_trend,vwap_trend,boll_breakout,ema_crossover",
            "--run_base",
            "--loader_verbose",
            "--n_jobs", "6",
            "--par_backend", "process",
            "--out_root", "./resultados/test8",
            "--print_top10"
        ]
    },
    {
        "name": "test9_period_q2_start",
        "desc": "Início Q2 2024, multi-tf, 6 métodos",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-04-01",
            "--end", "2024-06-01",
            "--interval", "auto",
            "--exec_rules", "1m,5m,15m",
            "--methods", "trend_breakout,rsi_reversion,macd_trend,vwap_trend,boll_breakout,ema_crossover",
            "--run_base",
            "--loader_verbose",
            "--n_jobs", "6",
            "--par_backend", "process",
            "--out_root", "./resultados/test9",
            "--print_top10"
        ]
    },
    {
        "name": "test10_combos_medium",
        "desc": "Multi-tf + combos (cap 50), 6 métodos",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-03-01",
            "--interval", "auto",
            "--exec_rules", "1m,5m",
            "--methods", "trend_breakout,rsi_reversion,macd_trend,vwap_trend,boll_breakout,ema_crossover",
            "--run_base",
            "--run_combos",
            "--combo_ops", "AND,MAJ,SEQ",
            "--combo_cap", "50",
            "--loader_verbose",
            "--n_jobs", "8",
            "--par_backend", "process",
            "--out_root", "./resultados/test10",
            "--print_top10"
        ]
    }
]


def run_test(test_config, test_num, total):
    """Executa um teste individual."""
    name = test_config["name"]
    desc = test_config["desc"]
    args = test_config["args"]

    # Criar diretório de output
    out_root = None
    for i, arg in enumerate(args):
        if arg == "--out_root":
            out_root = args[i+1]
            break

    if out_root:
        Path(out_root).mkdir(parents=True, exist_ok=True)

    cmd = ["python3", "selector21.py"] + args
    log_file = f"{out_root}/{name}.log" if out_root else f"{name}.log"

    print(f"[{test_num}/{total}] 🚀 {name}: {desc}")
    print(f"         Log: {log_file}")

    start_time = time.time()

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    return {
        "name": name,
        "desc": desc,
        "proc": proc,
        "log": log_file,
        "start": start_time,
        "out_root": out_root
    }


def monitor_processes(running):
    """Monitora processos rodando."""
    while running:
        time.sleep(5)

        completed = []
        for test in running:
            if test["proc"].poll() is not None:
                elapsed = time.time() - test["start"]
                exit_code = test["proc"].returncode

                status = "✅" if exit_code == 0 else "❌"
                print(f"{status} {test['name']} completou em {elapsed:.1f}s (exit: {exit_code})")

                # Mostra últimas linhas do log
                if exit_code == 0:
                    try:
                        with open(test["log"]) as f:
                            lines = f.readlines()
                            print(f"   Últimas 3 linhas:")
                            for line in lines[-3:]:
                                print(f"   {line.rstrip()}")
                    except:
                        pass

                completed.append(test)

        for test in completed:
            running.remove(test)

        if not running:
            break


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_parallel", type=int, default=10,
                        help="Número de testes paralelos (default: 10)")
    parser.add_argument("--tests", type=str, default="all",
                        help="Quais testes rodar: all, 1-3, 1,3,5, etc")
    args = parser.parse_args()

    # Filtrar testes
    if args.tests == "all":
        tests_to_run = TESTS
    else:
        indices = []
        if "-" in args.tests:
            start, end = map(int, args.tests.split("-"))
            indices = list(range(start-1, end))
        else:
            indices = [int(x)-1 for x in args.tests.split(",")]
        tests_to_run = [TESTS[i] for i in indices if i < len(TESTS)]

    print("="*80)
    print("🔥 EXECUÇÃO PARALELA MASSIVA DE BACKTESTS")
    print("="*80)
    print(f"\n📊 Recursos disponíveis:")
    print(f"   • CPU: 64 cores")
    print(f"   • RAM: 128GB")
    print(f"   • Testes selecionados: {len(tests_to_run)}")
    print(f"   • Paralelos simultâneos: {args.num_parallel}")
    print()

    running = []
    pending = list(tests_to_run)
    completed_count = 0
    total = len(tests_to_run)

    print("🚀 INICIANDO TESTES...\n")

    # Loop principal
    while pending or running:
        # Inicia novos testes se houver slots disponíveis
        while len(running) < args.num_parallel and pending:
            test_config = pending.pop(0)
            completed_count += 1
            test_info = run_test(test_config, completed_count, total)
            running.append(test_info)
            time.sleep(1)  # Pequeno delay entre starts

        # Monitora e limpa completados
        if running:
            time.sleep(10)

            completed = []
            for test in running:
                if test["proc"].poll() is not None:
                    elapsed = time.time() - test["start"]
                    exit_code = test["proc"].returncode

                    status = "✅" if exit_code == 0 else "❌"
                    print(f"\n{status} {test['name']} completou em {elapsed:.1f}s (exit: {exit_code})")

                    completed.append(test)

            for test in completed:
                running.remove(test)

    print("\n" + "="*80)
    print("✅ TODOS OS TESTES COMPLETADOS!")
    print("="*80)
    print("\n📂 Resultados em:")
    print("   ./resultados/test1/ ... test10/")
    print("\n📊 Próximo passo:")
    print("   python3 compare_results.py")
    print()


if __name__ == "__main__":
    main()
