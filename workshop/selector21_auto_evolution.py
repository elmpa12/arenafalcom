#!/usr/bin/env python3
"""
SELECTOR21 + AUTO EVOLUTION INTEGRATION

Integração NÃO-INVASIVA que adiciona auto-evolution ao selector21.py
SEM MODIFICAR os modelos ML existentes (XGBoost, RF, LogReg, Ensemble).

Uso:
    # Opção 1: Wrapper manual
    from selector21_auto_evolution import run_selector_with_evolution

    run_selector_with_evolution(
        symbol="BTCUSDT",
        start="2024-01-01",
        end="2024-06-01",
        apply_mode="review",  # ou "interactive" ou "auto"
    )

    # Opção 2: Patch automático
    import selector21_auto_evolution
    selector21_auto_evolution.patch_selector21()

    # Agora use selector21 normalmente - auto-evolution automático!
    import selector21
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import backtest integration
try:
    from backtest_integration import (
        with_auto_evolution,
        create_backtest_event,
        extract_backtest_metrics,
    )
    from auto_evolution_system import AutoEvolutionSystem, TradingEvent, EventType
    AUTO_EVOLUTION_AVAILABLE = True
except ImportError:
    AUTO_EVOLUTION_AVAILABLE = False
    print("⚠️  Auto Evolution not available. Install backtest_integration.py")


def parse_selector_output(output: str) -> Dict[str, Any]:
    """
    Parse output do selector21 para extrair métricas.

    Args:
        output: String com output do selector21

    Returns:
        Dict com métricas extraídas
    """
    metrics = {
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "sharpe_ratio": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
    }

    # Parse patterns comuns do selector21
    import re

    # Buscar por padrões de métricas
    patterns = {
        "total_trades": r"(?i)total.*trades?[:\s]+(\d+)",
        "win_rate": r"(?i)win.*rate[:\s]+([\d.]+)%?",
        "total_pnl": r"(?i)(?:total\s+)?pnl[:\s]+\$?([-\d.,]+)",
        "sharpe_ratio": r"(?i)sharpe[:\s]+([-\d.]+)",
        "profit_factor": r"(?i)profit.*factor[:\s]+([-\d.]+)",
        "max_drawdown": r"(?i)max.*drawdown[:\s]+\$?([-\d.,]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value_str = match.group(1).replace(",", "")
            try:
                metrics[key] = float(value_str) if "." in value_str else int(value_str)
            except:
                pass

    return metrics


def run_selector_with_evolution(
    symbol: str = "BTCUSDT",
    data_dir: str = "./data",
    start: str = "2024-01-01",
    end: str = "2024-06-01",
    run_base: bool = True,
    run_combos: bool = False,
    run_ml: bool = True,
    walkforward: bool = True,
    wf_train_months: float = 3.0,
    wf_val_months: float = 1.0,
    wf_step_months: float = 1.0,
    apply_mode: str = "review",
    enable_evolution: bool = True,
    **extra_args
) -> subprocess.CompletedProcess:
    """
    Executa selector21.py COM auto-evolution.

    Args:
        symbol: Par a tradear (ex: BTCUSDT)
        data_dir: Diretório com dados Parquet
        start: Data inicial (YYYY-MM-DD)
        end: Data final (YYYY-MM-DD)
        run_base: Executar métodos base
        run_combos: Executar combos
        run_ml: Executar ML (XGBoost, RF, LogReg)
        walkforward: Usar walk-forward optimization
        wf_train_months: Meses de treino
        wf_val_months: Meses de validação
        wf_step_months: Step do walk-forward
        apply_mode: Modo auto-evolution (review/interactive/auto)
        enable_evolution: Se True, dispara auto-evolution
        **extra_args: Args adicionais para selector21

    Returns:
        CompletedProcess com resultado da execução
    """

    print("=" * 70)
    print("SELECTOR21 + AUTO EVOLUTION")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Period: {start} → {end}")
    print(f"Walk-Forward: {walkforward} (train={wf_train_months}m, val={wf_val_months}m)")
    print(f"ML Models: XGBoost, RandomForest, LogReg, Ensemble")
    print(f"Auto-Evolution: {enable_evolution} (mode: {apply_mode})")
    print("=" * 70)

    # Construir comando selector21
    cmd = [
        "python3", "selector21.py",
        "--symbol", symbol,
        "--data_dir", data_dir,
        "--start", start,
        "--end", end,
    ]

    if run_base:
        cmd.append("--run_base")
    if run_combos:
        cmd.append("--run_combos")
    if run_ml:
        cmd.append("--run_ml")
        cmd.extend(["--ml_model_kind", "auto"])  # XGB→RF→LogReg

    if walkforward:
        cmd.append("--walkforward")
        cmd.extend([
            "--wf_train_months", str(wf_train_months),
            "--wf_val_months", str(wf_val_months),
            "--wf_step_months", str(wf_step_months),
        ])

    # Args extras
    for key, value in extra_args.items():
        cmd.append(f"--{key}")
        if value is not True:
            cmd.append(str(value))

    print("\n🔬 Executando selector21...")
    print(f"Command: {' '.join(cmd)}\n")

    # Executar selector21
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hora timeout
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Se auto-evolution habilitado, analisar resultados
    if enable_evolution and AUTO_EVOLUTION_AVAILABLE and result.returncode == 0:
        print("\n" + "=" * 70)
        print("🤖 AUTO EVOLUTION ANALYSIS")
        print("=" * 70)

        # Parse métricas do output
        metrics = parse_selector_output(result.stdout)

        print(f"📊 Métricas extraídas:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

        # Criar evento
        event = TradingEvent(
            event_type=EventType.BACKTEST_RESULT,
            timestamp=datetime.now().isoformat(),
            data=metrics,
            context={
                "symbol": symbol,
                "period": f"{start} to {end}",
                "walkforward": walkforward,
                "ml_models": "XGBoost, RandomForest, LogReg, Ensemble",
                "strategy": "selector21_ml",
            }
        )

        # Disparar auto-evolution
        print("\n🧠 Triggering Auto Evolution (Claude + GPT)...")
        evo = AutoEvolutionSystem(apply_mode=apply_mode)
        analysis = evo.intercept_event(event)

        print(f"\n✅ Dual Analysis Complete!")
        print(f"   Confidence: {analysis.confidence:.0%}")
        print(f"   Actions proposed: {len(analysis.actions)}")

        if analysis.actions:
            print(f"\n   Top 3 Actions:")
            for i, action in enumerate(analysis.actions[:3], 1):
                print(f"   {i}. [{action['tipo']}] {action['descrição'][:60]}...")
                print(f"      Priority: {action['prioridade']}/10")

        print(f"\n📝 Learning saved to: claudex/LEARNING_LOG.jsonl")

    return result


def patch_selector21():
    """
    Patch EXPERIMENTAL que intercepta execuções do selector21.

    AVISO: Isso é experimental e pode não funcionar em todos os casos.
    Recomendado usar run_selector_with_evolution() diretamente.
    """
    print("⚠️  Patch automático ainda não implementado.")
    print("   Use run_selector_with_evolution() para integração completa.")
    print()
    print("   Exemplo:")
    print("   from selector21_auto_evolution import run_selector_with_evolution")
    print("   run_selector_with_evolution(symbol='BTCUSDT', start='2024-01-01')")


def create_evolution_roadmap():
    """
    Cria roadmap de evolução gradual do sistema.

    Conforme visão do usuário:
    1. Backtests exigentes (AGORA)
    2. Deep Learning
    3. Paper Trading
    4. Real Trading
    """
    roadmap = {
        "fase_atual": "1_backtests_exigentes",
        "fases": [
            {
                "fase": "1_backtests_exigentes",
                "status": "EM_PROGRESSO",
                "descricao": "Integrar auto-evolution com selector21 backtests",
                "modelos_ml": ["XGBoost", "RandomForest", "LogReg", "Ensemble"],
                "objetivo": "Setup qualificado com análise dual contínua",
                "proximos_passos": [
                    "Rodar walk-forward com auto-evolution",
                    "Analisar logs de aprendizado",
                    "Validar ações propostas por Claude + GPT",
                    "Atingir métricas mínimas (70%+ win rate, Sharpe > 1.5)",
                ]
            },
            {
                "fase": "2_deep_learning",
                "status": "PENDENTE",
                "descricao": "Auto-evolution aprende com DL (GRU, TCN, Transformers)",
                "modelos_dl": ["GRU", "TCN", "Transformers"],
                "objetivo": "IAs aprendem com modelos deep learning",
                "dependencias": ["Fase 1 qualificada"],
            },
            {
                "fase": "3_paper_trading",
                "status": "PENDENTE",
                "descricao": "Auto-evolution em tempo real (paper trading)",
                "objetivo": "Sistema aprende com mercado ao vivo (sem risco)",
                "dependencias": ["Fase 2 qualificada"],
            },
            {
                "fase": "4_real_trading",
                "status": "PENDENTE",
                "descricao": "Auto-evolution em produção (real trading)",
                "objetivo": "Sistema evolui continuamente em produção",
                "dependencias": ["Fase 3 qualificada", "Métricas consistentes"],
            }
        ]
    }

    # Salvar roadmap
    roadmap_path = Path("/opt/botscalpv3/evolution_roadmap.json")
    with open(roadmap_path, 'w') as f:
        json.dump(roadmap, f, indent=2)

    print("📋 Evolution Roadmap criado!")
    print(f"   Arquivo: {roadmap_path}")
    print()
    print("🎯 Fase Atual: 1 - BACKTESTS EXIGENTES")
    print("   Modelos ML preservados: XGBoost, RF, LogReg, Ensemble")
    print("   Próximo: Deep Learning → Paper → Real")

    return roadmap


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("SELECTOR21 + AUTO EVOLUTION INTEGRATION")
    print("🚀" * 35)
    print()
    print("Esta integração adiciona auto-evolution ao selector21.py")
    print("PRESERVANDO todos os modelos ML existentes:")
    print("  • XGBoost (400 estimators)")
    print("  • RandomForest (300 estimators)")
    print("  • Logistic Regression")
    print("  • Ensemble (combina os 3)")
    print()
    print("=" * 70)
    print()

    # Criar roadmap
    create_evolution_roadmap()

    print()
    print("💡 Como usar:")
    print()
    print("1. Execução direta:")
    print("   from selector21_auto_evolution import run_selector_with_evolution")
    print("   run_selector_with_evolution(")
    print("       symbol='BTCUSDT',")
    print("       start='2024-01-01',")
    print("       end='2024-06-01',")
    print("       apply_mode='review'  # ou 'interactive'")
    print("   )")
    print()
    print("2. Via command line:")
    print("   python3 selector21_auto_evolution.py")
    print()
