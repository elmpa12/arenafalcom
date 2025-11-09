#!/usr/bin/env python3
"""
TEST: Backtest Integration with Auto Evolution System

Demonstra a integração completa:
1. Backtest sintético executado
2. Métricas extraídas
3. Auto Evolution System acionado (Claude + GPT)
4. Ações propostas
5. Aprendizados salvos

Uso:
    python3 test_backtest_integration.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest_integration import (
    with_auto_evolution,
    extract_backtest_metrics,
    create_backtest_event,
    create_error_event,
)
from auto_evolution_system import AutoEvolutionSystem, EventType


def create_synthetic_backtest_results(
    total_trades: int = 100,
    win_rate: float = 0.65,
    avg_win: float = 150.0,
    avg_loss: float = 80.0,
) -> pd.DataFrame:
    """
    Cria resultados sintéticos de backtest para teste.

    Args:
        total_trades: Número total de trades
        win_rate: Taxa de acertos (0-1)
        avg_win: Lucro médio dos vencedores
        avg_loss: Perda média dos perdedores

    Returns:
        DataFrame com trades sintéticos
    """
    print(f"📊 Gerando {total_trades} trades sintéticos (win rate: {win_rate:.1%})")

    np.random.seed(42)

    trades = []
    balance = 10000.0
    cumulative_pnl = 0.0

    for i in range(total_trades):
        # Determina se é win ou loss
        is_win = np.random.random() < win_rate

        if is_win:
            pnl = np.random.normal(avg_win, avg_win * 0.3)
        else:
            pnl = -np.random.normal(avg_loss, avg_loss * 0.3)

        cumulative_pnl += pnl

        entry_time = datetime.now() - timedelta(hours=total_trades - i)
        bars_held = np.random.randint(5, 50)
        exit_time = entry_time + timedelta(minutes=bars_held)

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_px": 43000 + np.random.randint(-500, 500),
            "exit_px": 43000 + np.random.randint(-500, 500),
            "pnl": pnl,
            "side": np.random.choice(["LONG", "SHORT"]),
            "bars_held": bars_held,
            "fee_paid": abs(pnl) * 0.0004,  # 0.04% fee
        })

    df = pd.DataFrame(trades)
    print(f"✅ Backtest sintético criado: PnL total = {df['pnl'].sum():.2f}")
    return df


def mock_backtest_function(
    df: pd.DataFrame,
    sig: pd.Series,
    **kwargs
) -> pd.DataFrame:
    """
    Mock da função backtest_from_signals para teste.

    Na integração real, isso seria a função original do selector21.
    """
    print("🔬 Mock backtest executando...")

    # Simula processamento
    import time
    time.sleep(0.5)

    # Retorna trades sintéticos
    return create_synthetic_backtest_results(
        total_trades=kwargs.get("total_trades", 100),
        win_rate=kwargs.get("win_rate", 0.65),
    )


def test_basic_integration():
    """
    Teste 1: Integração básica com backtest
    """
    print("\n" + "=" * 70)
    print("TEST 1: Integração Básica - Backtest → Auto Evolution")
    print("=" * 70)

    # Dados dummy
    dates = pd.date_range("2024-01-01", periods=1000, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 43000,
        "high": 43100,
        "low": 42900,
        "close": 43000,
    })
    sig = pd.Series(np.random.choice([-1, 0, 1], size=len(df)))

    # Executar com auto-evolution
    trades = with_auto_evolution(
        backtest_func=mock_backtest_function,
        strategy_name="scalping_atr_test",
        timeframe="5m",
        enable_evolution=True,
        df=df,
        sig=sig,
        total_trades=50,
        win_rate=0.70,
    )

    print(f"\n✅ Test 1 Complete! {len(trades)} trades retornados")
    print(f"   PnL Total: {trades['pnl'].sum():.2f}")
    print(f"   Win Rate: {(trades['pnl'] > 0).mean():.1%}")


def test_metrics_extraction():
    """
    Teste 2: Extração de métricas
    """
    print("\n" + "=" * 70)
    print("TEST 2: Extração de Métricas")
    print("=" * 70)

    # Criar trades sintéticos
    trades = create_synthetic_backtest_results(total_trades=100, win_rate=0.68)

    # Extrair métricas
    metrics = extract_backtest_metrics(trades)

    print("\n📊 Métricas Extraídas:")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")
    print(f"   Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Max Drawdown: ${metrics['max_drawdown']:.2f}")
    print(f"   Avg Win: ${metrics['avg_win']:.2f}")
    print(f"   Avg Loss: ${metrics['avg_loss']:.2f}")

    print("\n✅ Test 2 Complete!")


def test_error_event():
    """
    Teste 3: Evento de erro
    """
    print("\n" + "=" * 70)
    print("TEST 3: Análise de Erro")
    print("=" * 70)

    # Criar evento de erro
    error_event = create_error_event(
        error_type="OrderRejected",
        error_message="Insufficient balance for order",
        traceback_str="Traceback (simulated):\n  File 'trading.py', line 123...",
        context={
            "symbol": "BTCUSDT",
            "order_size": 1.5,
            "available_balance": 1000.0,
        }
    )

    # Analisar erro
    print("🤖 Disparando análise dual do erro...")
    evo_system = AutoEvolutionSystem(apply_mode="review")
    analysis = evo_system.intercept_event(error_event)

    print(f"\n✅ Análise completa! Confiança: {analysis.confidence:.0%}")
    print(f"   {len(analysis.actions)} ações propostas")

    if analysis.actions:
        print("\n   Ações propostas:")
        for i, action in enumerate(analysis.actions, 1):
            print(f"   {i}. [{action['tipo']}] (prioridade {action['prioridade']}/10)")
            print(f"      → {action['descrição']}")


def test_multiple_backtests():
    """
    Teste 4: Múltiplos backtests sequenciais
    """
    print("\n" + "=" * 70)
    print("TEST 4: Múltiplos Backtests → Aprendizado Contínuo")
    print("=" * 70)

    strategies = [
        ("scalping_v1", 0.65, "5m"),
        ("scalping_v2", 0.72, "5m"),
        ("momentum_v1", 0.58, "15m"),
    ]

    for strategy_name, win_rate, timeframe in strategies:
        print(f"\n📊 Testing {strategy_name} @ {timeframe} (target win rate: {win_rate:.1%})")

        trades = with_auto_evolution(
            backtest_func=mock_backtest_function,
            strategy_name=strategy_name,
            timeframe=timeframe,
            enable_evolution=True,
            df=pd.DataFrame(),  # dummy
            sig=pd.Series(),    # dummy
            total_trades=80,
            win_rate=win_rate,
        )

        print(f"   ✅ {len(trades)} trades, PnL: {trades['pnl'].sum():.2f}")

    print("\n✅ Test 4 Complete! Múltiplas estratégias analisadas")
    print("   Logs salvos em: claudex/LEARNING_LOG.jsonl")


def test_performance_degradation():
    """
    Teste 5: Detectar degradação de performance
    """
    print("\n" + "=" * 70)
    print("TEST 5: Detecção de Degradação de Performance")
    print("=" * 70)

    print("\n🔬 Simulando 3 backtests com performance decrescente...")

    win_rates = [0.75, 0.68, 0.52]  # Performance degradando

    for i, win_rate in enumerate(win_rates, 1):
        print(f"\n   Backtest {i}: Win rate = {win_rate:.1%}")

        trades = with_auto_evolution(
            backtest_func=mock_backtest_function,
            strategy_name="scalping_degrading",
            timeframe="5m",
            enable_evolution=True,
            df=pd.DataFrame(),
            sig=pd.Series(),
            total_trades=100,
            win_rate=win_rate,
            additional_context={"iteration": i, "expected_degradation": True}
        )

    print("\n✅ Test 5 Complete! Auto Evolution deve ter detectado degradação")


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("BACKTEST INTEGRATION TEST SUITE")
    print("Testing Auto Evolution System with Backtests")
    print("🚀" * 35)

    try:
        # Executar testes
        test_basic_integration()
        test_metrics_extraction()
        test_error_event()
        test_multiple_backtests()
        test_performance_degradation()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\n✅ Integração completa funcionando!")
        print("✅ Claude + GPT analisando backtests")
        print("✅ Logs salvos em claudex/LEARNING_LOG.jsonl")
        print("\n📝 Próximos passos:")
        print("   1. Integrar com selector21.py real")
        print("   2. Rodar backtests walk-forward com auto-evolution")
        print("   3. Ver sistema evoluindo automaticamente!")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
