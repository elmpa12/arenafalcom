#!/usr/bin/env python3
"""
EXEMPLO: Modo Interativo do Auto Evolution System

Demonstra como usar o modo interativo que pergunta antes de aplicar cada mudança.

Uso:
    python3 example_interactive_mode.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from backtest_integration import with_auto_evolution, create_synthetic_backtest_results


def mock_backtest(**kwargs):
    """Mock de backtest para demonstração"""
    print("🔬 Executando backtest mock...")
    return create_synthetic_backtest_results(
        total_trades=kwargs.get("total_trades", 50),
        win_rate=kwargs.get("win_rate", 0.68),
    )


def example_review_mode():
    """
    Modo REVIEW (padrão): Apenas propõe, não aplica
    """
    print("\n" + "="*70)
    print("EXEMPLO 1: Modo REVIEW (apenas propõe)")
    print("="*70)

    from auto_evolution_system import AutoEvolutionSystem, TradingEvent, EventType

    # Evento de teste
    event = TradingEvent(
        event_type=EventType.BACKTEST_RESULT,
        timestamp=datetime.now().isoformat(),
        data={
            "total_trades": 100,
            "win_rate": 0.65,
            "total_pnl": 3500.0,
            "sharpe_ratio": 1.8,
        },
        context={"strategy": "scalping_v1", "timeframe": "5m"}
    )

    # Modo review (padrão)
    evo = AutoEvolutionSystem(apply_mode="review")
    analysis = evo.intercept_event(event)

    print(f"\n✅ Análise completa!")
    print(f"   {len(analysis.actions)} ações propostas")
    print(f"   NENHUMA foi aplicada (modo review)")


def example_interactive_mode():
    """
    Modo INTERACTIVE: Pergunta antes de aplicar cada mudança
    """
    print("\n" + "="*70)
    print("EXEMPLO 2: Modo INTERACTIVE (pergunta antes)")
    print("="*70)
    print()
    print("⚠️  ATENÇÃO: Este modo irá perguntar se você quer aplicar cada mudança!")
    print("   Digite 'S' para Sim, 'n' para Não, ou Enter para Sim (padrão)")
    print()

    input("Pressione Enter para continuar...")

    from auto_evolution_system import AutoEvolutionSystem, TradingEvent, EventType

    # Evento de teste
    event = TradingEvent(
        event_type=EventType.ERROR_OCCURRED,
        timestamp=datetime.now().isoformat(),
        data={
            "error_type": "OrderRejected",
            "message": "Insufficient balance",
            "traceback": "...",
        },
        context={"last_trade": "BUY BTCUSDT @ 43500"}
    )

    # Modo interactive
    evo = AutoEvolutionSystem(apply_mode="interactive")
    analysis = evo.intercept_event(event)

    print(f"\n✅ Processo interativo completo!")
    print(f"   {len(analysis.actions)} ações foram analisadas")
    print(f"   Você escolheu quais aplicar")


def example_auto_mode():
    """
    Modo AUTO: Aplica tudo automaticamente (CUIDADO!)
    """
    print("\n" + "="*70)
    print("EXEMPLO 3: Modo AUTO (aplica automaticamente)")
    print("="*70)
    print()
    print("⚠️  PERIGO: Este modo aplica TODAS as mudanças automaticamente!")
    print("   Use apenas em ambiente de teste controlado")
    print()

    response = input("Tem certeza que quer continuar? [s/N]: ").strip().lower()

    if response not in ["s", "sim", "y", "yes"]:
        print("✅ Cancelado. Boa escolha!")
        return

    from auto_evolution_system import AutoEvolutionSystem, TradingEvent, EventType

    # Evento de teste
    event = TradingEvent(
        event_type=EventType.BACKTEST_RESULT,
        timestamp=datetime.now().isoformat(),
        data={
            "total_trades": 80,
            "win_rate": 0.72,
            "total_pnl": 5000.0,
            "sharpe_ratio": 2.1,
        },
        context={"strategy": "momentum_v2", "timeframe": "15m"}
    )

    # Modo auto (PERIGOSO!)
    evo = AutoEvolutionSystem(apply_mode="auto")
    analysis = evo.intercept_event(event)

    print(f"\n✅ Modo automático executado!")
    print(f"   {len(analysis.actions)} ações foram propostas")
    print(f"   TODAS as code_changes foram aplicadas automaticamente")


def example_backtest_integration_interactive():
    """
    Exemplo 4: Integração com backtest em modo interativo
    """
    print("\n" + "="*70)
    print("EXEMPLO 4: Backtest + Modo Interactive")
    print("="*70)

    # Executar backtest com auto-evolution interativo
    trades = with_auto_evolution(
        backtest_func=mock_backtest,
        strategy_name="scalping_interactive_test",
        timeframe="5m",
        enable_evolution=True,
        total_trades=60,
        win_rate=0.70,
    )

    print(f"\n✅ Backtest completo!")
    print(f"   {len(trades)} trades executados")


def main():
    print("\n" + "🚀" * 35)
    print("AUTO EVOLUTION SYSTEM - MODOS DE APLICAÇÃO")
    print("🚀" * 35)
    print()
    print("Este exemplo demonstra os 3 modos disponíveis:")
    print("  1. REVIEW: Apenas propõe (padrão - SEGURO)")
    print("  2. INTERACTIVE: Pergunta antes de aplicar (RECOMENDADO)")
    print("  3. AUTO: Aplica automaticamente (CUIDADO!)")
    print()

    print("\n📋 MENU DE EXEMPLOS")
    print("="*70)
    print("1. Modo Review (apenas propõe)")
    print("2. Modo Interactive (pergunta antes)")
    print("3. Modo Auto (aplica tudo - CUIDADO!)")
    print("4. Backtest + Interactive")
    print("5. Executar todos")
    print("0. Sair")
    print("="*70)

    while True:
        try:
            choice = input("\nEscolha uma opção [0-5]: ").strip()

            if choice == "0":
                print("\n👋 Até logo!")
                break

            elif choice == "1":
                example_review_mode()

            elif choice == "2":
                example_interactive_mode()

            elif choice == "3":
                example_auto_mode()

            elif choice == "4":
                example_backtest_integration_interactive()

            elif choice == "5":
                example_review_mode()
                print("\n" + "─"*70)
                example_interactive_mode()
                print("\n" + "─"*70)
                example_auto_mode()
                print("\n" + "─"*70)
                example_backtest_integration_interactive()
                break

            else:
                print("⚠️  Opção inválida. Escolha 0-5.")

        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Interrompido pelo usuário. Até logo!")
            break


if __name__ == "__main__":
    main()
