#!/usr/bin/env python3
"""
RUN BACKTEST WITH IAS - BotScalp V3

Executa backtest conforme consenso das IAs (Claude + GPT).
Sistema integrado com auto-evolution para aprendizado contínuo.

Uso:
    python3 run_backtest_with_ias.py --start 2024-01-01 --end 2024-06-01
"""

import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Consenso das IAs (baseado nas 12 rodadas de debate)
CONSENSUS = {
    "metricas_principais": {
        "sharpe_ratio": {"min": 1.5, "ideal": 2.0},
        "max_drawdown": {"max": 0.20, "ideal": 0.10},  # 20% max, 10% ideal
        "profit_factor": {"min": 1.5, "ideal": 2.0},
        "win_rate": {"min": 0.55, "ideal": 0.65},  # 55% min, 65% ideal
        "sortino_ratio": {"min": 1.5, "ideal": 2.5},
        "calmar_ratio": {"min": 1.0, "ideal": 2.0}
    },

    "validacao": {
        "metodo": "walk_forward",
        "train_window_months": 6,
        "test_window_months": 1,
        "min_folds": 6
    },

    "features": {
        "price_action": ["returns", "volatility", "momentum"],
        "volume": ["volume", "vwap", "buy_sell_imbalance"],
        "temporal": ["hour_of_day", "day_of_week"],
        "avoid_look_ahead_bias": True
    },

    "anti_overfitting": {
        "regularization": True,
        "early_stopping": True,
        "feature_selection": True,
        "ensemble": True,
        "out_of_sample_test": True
    }
}


def run_selector21_backtest(
    symbol: str,
    start: str,
    end: str,
    data_dir: str = "./data_monthly"
) -> dict:
    """Roda selector21 com configurações do consenso."""

    print("\n" + "="*70)
    print("🎯 RODANDO BACKTEST - Configuração Consenso IAs")
    print("="*70)
    print(f"\nSímbolo: {symbol}")
    print(f"Período: {start} → {end}")
    print(f"Dados: {data_dir}")
    print(f"\nMétricas Alvo:")
    for metric, thresholds in CONSENSUS["metricas_principais"].items():
        print(f"  {metric}: {thresholds}")

    # Comando selector21
    cmd = [
        "python3", "-m", "core.selectors.selector21",
        "--symbol", symbol,
        "--start", start,
        "--end", end,
        "--data_dir", data_dir,
        "--run_ml", "true",
        "--walkforward", "true"
    ]

    print(f"\n💻 Executando: {' '.join(cmd)}")
    print()

    # Executar
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 min max
        )

        output = result.stdout + result.stderr

        # Parse métricas do output
        metrics = parse_output_metrics(output)

        return {
            "success": result.returncode == 0,
            "metrics": metrics,
            "output": output,
            "timestamp": datetime.now().isoformat()
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Timeout (>10min)",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def parse_output_metrics(output: str) -> dict:
    """Extrai métricas do output do selector21."""
    metrics = {}

    # Patterns comuns
    patterns = {
        "win_rate": r"win.*rate.*?(\d+\.?\d*)%",
        "total_trades": r"total.*trades.*?(\d+)",
        "profit_factor": r"profit.*factor.*?(\d+\.?\d*)",
        "sharpe": r"sharpe.*?(\d+\.?\d*)",
        "max_drawdown": r"max.*drawdown.*?(\d+\.?\d*)%"
    }

    import re
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[metric_name] = float(match.group(1))

    return metrics


def evaluate_metrics(metrics: dict) -> dict:
    """Avalia métricas contra thresholds do consenso."""
    evaluation = {
        "passed": [],
        "failed": [],
        "score": 0,
        "grade": "FAIL"
    }

    thresholds = CONSENSUS["metricas_principais"]

    # Sharpe Ratio
    if "sharpe" in metrics:
        if metrics["sharpe"] >= thresholds["sharpe_ratio"]["ideal"]:
            evaluation["passed"].append(f"Sharpe {metrics['sharpe']} ≥ {thresholds['sharpe_ratio']['ideal']} ✓")
            evaluation["score"] += 20
        elif metrics["sharpe"] >= thresholds["sharpe_ratio"]["min"]:
            evaluation["passed"].append(f"Sharpe {metrics['sharpe']} ≥ {thresholds['sharpe_ratio']['min']} ✓")
            evaluation["score"] += 10
        else:
            evaluation["failed"].append(f"Sharpe {metrics['sharpe']} < {thresholds['sharpe_ratio']['min']} ✗")

    # Win Rate
    if "win_rate" in metrics:
        win_rate = metrics["win_rate"] / 100  # Convert % to decimal
        if win_rate >= thresholds["win_rate"]["ideal"]:
            evaluation["passed"].append(f"Win Rate {metrics['win_rate']}% ≥ {thresholds['win_rate']['ideal']*100}% ✓")
            evaluation["score"] += 20
        elif win_rate >= thresholds["win_rate"]["min"]:
            evaluation["passed"].append(f"Win Rate {metrics['win_rate']}% ≥ {thresholds['win_rate']['min']*100}% ✓")
            evaluation["score"] += 10
        else:
            evaluation["failed"].append(f"Win Rate {metrics['win_rate']}% < {thresholds['win_rate']['min']*100}% ✗")

    # Max Drawdown
    if "max_drawdown" in metrics:
        dd = metrics["max_drawdown"] / 100  # Convert % to decimal
        if dd <= thresholds["max_drawdown"]["ideal"]:
            evaluation["passed"].append(f"Max DD {metrics['max_drawdown']}% ≤ {thresholds['max_drawdown']['ideal']*100}% ✓")
            evaluation["score"] += 20
        elif dd <= thresholds["max_drawdown"]["max"]:
            evaluation["passed"].append(f"Max DD {metrics['max_drawdown']}% ≤ {thresholds['max_drawdown']['max']*100}% ✓")
            evaluation["score"] += 10
        else:
            evaluation["failed"].append(f"Max DD {metrics['max_drawdown']}% > {thresholds['max_drawdown']['max']*100}% ✗")

    # Profit Factor
    if "profit_factor" in metrics:
        if metrics["profit_factor"] >= thresholds["profit_factor"]["ideal"]:
            evaluation["passed"].append(f"Profit Factor {metrics['profit_factor']} ≥ {thresholds['profit_factor']['ideal']} ✓")
            evaluation["score"] += 20
        elif metrics["profit_factor"] >= thresholds["profit_factor"]["min"]:
            evaluation["passed"].append(f"Profit Factor {metrics['profit_factor']} ≥ {thresholds['profit_factor']['min']} ✓")
            evaluation["score"] += 10
        else:
            evaluation["failed"].append(f"Profit Factor {metrics['profit_factor']} < {thresholds['profit_factor']['min']} ✗")

    # Grade
    if evaluation["score"] >= 80:
        evaluation["grade"] = "A - QUALIFICADO PARA PAPER TRADING"
    elif evaluation["score"] >= 60:
        evaluation["grade"] = "B - BOM, MELHORAR ANTES DE PAPER"
    elif evaluation["score"] >= 40:
        evaluation["grade"] = "C - MÉDIO, PRECISA EVOLUIR"
    else:
        evaluation["grade"] = "F - REPROVAR, REVISAR ESTRATÉGIA"

    return evaluation


def trigger_auto_evolution(backtest_result: dict, evaluation: dict):
    """Dispara auto-evolution para analisar backtest."""

    print("\n" + "="*70)
    print("🤖 DISPARANDO AUTO-EVOLUTION")
    print("="*70)

    from auto_evolution_system import AutoEvolutionSystem

    # Criar evento de backtest
    event = {
        "type": "backtest_completed",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "metrics": backtest_result["metrics"],
            "evaluation": evaluation,
            "consensus_thresholds": CONSENSUS["metricas_principais"]
        }
    }

    # Disparar análise dual
    evo = AutoEvolutionSystem(apply_mode="review")
    analysis = evo.intercept_event(event)

    print("\n✅ Auto-evolution analysis saved to claudex/LEARNING_LOG.jsonl")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Run backtest with AI consensus")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2024-06-01", help="End date")
    parser.add_argument("--data_dir", default="./data_monthly", help="Data directory")
    parser.add_argument("--skip-auto-evolution", action="store_true", help="Skip auto-evolution analysis")

    args = parser.parse_args()

    # Run backtest
    result = run_selector21_backtest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        data_dir=args.data_dir
    )

    # Evaluate
    evaluation = evaluate_metrics(result.get("metrics", {}))

    # Print results
    print("\n" + "="*70)
    print("📊 RESULTADO DO BACKTEST")
    print("="*70)
    print(f"\nGrade: {evaluation['grade']}")
    print(f"Score: {evaluation['score']}/100")

    print(f"\n✅ PASSED ({len(evaluation['passed'])}):")
    for item in evaluation["passed"]:
        print(f"  {item}")

    if evaluation["failed"]:
        print(f"\n❌ FAILED ({len(evaluation['failed'])}):")
        for item in evaluation["failed"]:
            print(f"  {item}")

    # Auto-evolution
    if not args.skip_auto_evolution and result.get("success"):
        analysis = trigger_auto_evolution(result, evaluation)

    # Save results
    output_file = f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "backtest": result,
            "evaluation": evaluation,
            "consensus": CONSENSUS
        }, f, indent=2)

    print(f"\n💾 Resultados salvos em: {output_file}")
    print()

    return evaluation["score"] >= 60  # Return True if grade B or better


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
