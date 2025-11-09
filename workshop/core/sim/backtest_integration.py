#!/usr/bin/env python3
"""
BACKTEST INTEGRATION - BotScalp V3 + Auto Evolution System

Integração entre backtests e auto_evolution_system para aprendizado contínuo.

Arquitetura:
1. Backtest executa normalmente
2. Resultados são interceptados
3. Métricas são extraídas
4. Auto Evolution System analisa (Claude + GPT)
5. Aprendizados são salvos
6. Sistema evolui continuamente

Uso:
    from backtest_integration import with_auto_evolution

    # Wrapper automático que intercepta resultados
    results = with_auto_evolution(
        backtest_func=backtest_from_signals,
        df=df,
        sig=signals,
        strategy_name="scalping_v1",
        timeframe="5m",
        # ... demais params do backtest
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import warnings

# Import auto evolution system
try:
    from auto_evolution_system import AutoEvolutionSystem, TradingEvent, EventType
    AUTO_EVOLUTION_AVAILABLE = True
except ImportError:
    AUTO_EVOLUTION_AVAILABLE = False
    warnings.warn("Auto Evolution System not available. Running in passive mode.")


def extract_backtest_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extrai métricas completas de um DataFrame de trades.

    Args:
        trades_df: DataFrame com colunas [entry_time, exit_time, pnl, side, ...]

    Returns:
        Dict com métricas completas do backtest
    """
    if trades_df is None or len(trades_df) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_duration_minutes": 0.0,
            "total_fees": 0.0,
        }

    # Total trades
    total_trades = len(trades_df)

    # Win rate
    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] < 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # PnL
    total_pnl = trades_df["pnl"].sum()
    avg_pnl = trades_df["pnl"].mean()

    # Wins vs Losses
    winning_trades = trades_df[trades_df["pnl"] > 0]
    losing_trades = trades_df[trades_df["pnl"] < 0]

    avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = abs(losing_trades["pnl"].mean()) if len(losing_trades) > 0 else 0.0
    largest_win = winning_trades["pnl"].max() if len(winning_trades) > 0 else 0.0
    largest_loss = abs(losing_trades["pnl"].min()) if len(losing_trades) > 0 else 0.0

    # Profit Factor
    gross_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0.0
    gross_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Drawdown
    cumulative_pnl = trades_df["pnl"].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()

    # Sharpe Ratio (simplificado - retornos por trade)
    if len(trades_df) > 1 and trades_df["pnl"].std() > 0:
        sharpe_ratio = (trades_df["pnl"].mean() / trades_df["pnl"].std()) * np.sqrt(len(trades_df))
    else:
        sharpe_ratio = 0.0

    # Duration
    if "bars_held" in trades_df.columns:
        avg_duration_bars = trades_df["bars_held"].mean()
    else:
        avg_duration_bars = 0.0

    # Fees
    if "fee_paid" in trades_df.columns:
        total_fees = trades_df["fee_paid"].sum()
    else:
        total_fees = 0.0

    return {
        "total_trades": int(total_trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "avg_pnl": float(avg_pnl),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "largest_win": float(largest_win),
        "largest_loss": float(largest_loss),
        "avg_duration_bars": float(avg_duration_bars),
        "total_fees": float(total_fees),
    }


def create_backtest_event(
    metrics: Dict[str, Any],
    strategy_name: str,
    timeframe: str,
    additional_context: Optional[Dict] = None
) -> TradingEvent:
    """
    Cria um TradingEvent de backtest para análise dual.

    Args:
        metrics: Métricas extraídas do backtest
        strategy_name: Nome da estratégia (ex: "scalping_v1")
        timeframe: Timeframe usado (ex: "5m")
        additional_context: Contexto adicional (params, etc)

    Returns:
        TradingEvent pronto para auto_evolution_system
    """
    context = {
        "strategy": strategy_name,
        "timeframe": timeframe,
        "timestamp": datetime.now().isoformat(),
    }

    if additional_context:
        context.update(additional_context)

    return TradingEvent(
        event_type=EventType.BACKTEST_RESULT,
        timestamp=datetime.now().isoformat(),
        data=metrics,
        context=context
    )


def with_auto_evolution(
    backtest_func: Callable,
    strategy_name: str,
    timeframe: str,
    enable_evolution: bool = True,
    additional_context: Optional[Dict] = None,
    **backtest_kwargs
) -> pd.DataFrame:
    """
    Wrapper que executa backtest e dispara auto-evolution.

    Args:
        backtest_func: Função de backtest (ex: backtest_from_signals)
        strategy_name: Nome da estratégia
        timeframe: Timeframe
        enable_evolution: Se True, chama auto_evolution_system
        additional_context: Contexto adicional
        **backtest_kwargs: Argumentos para a função de backtest

    Returns:
        DataFrame de trades (mesmo retorno da função original)

    Example:
        >>> from selector21 import backtest_from_signals
        >>> from backtest_integration import with_auto_evolution
        >>>
        >>> trades = with_auto_evolution(
        ...     backtest_func=backtest_from_signals,
        ...     strategy_name="scalping_atr",
        ...     timeframe="5m",
        ...     df=df,
        ...     sig=signals,
        ...     max_hold=100,
        ...     fee_perc=0.0002,
        ... )
    """

    # 1. Executar backtest original
    print(f"🔬 Executing backtest: {strategy_name} @ {timeframe}")
    trades_df = backtest_func(**backtest_kwargs)

    # 2. Se auto-evolution desabilitado, retorna direto
    if not enable_evolution or not AUTO_EVOLUTION_AVAILABLE:
        return trades_df

    # 3. Extrair métricas
    metrics = extract_backtest_metrics(trades_df)
    print(f"📊 Backtest complete: {metrics['total_trades']} trades, "
          f"{metrics['win_rate']:.1%} win rate, "
          f"PnL: {metrics['total_pnl']:.2f}")

    # 4. Criar evento
    event = create_backtest_event(
        metrics=metrics,
        strategy_name=strategy_name,
        timeframe=timeframe,
        additional_context=additional_context
    )

    # 5. Disparar auto-evolution (análise dual)
    try:
        print("🤖 Triggering Auto Evolution System (Claude + GPT)...")
        evo_system = AutoEvolutionSystem(apply_mode="review")
        analysis = evo_system.intercept_event(event)

        print(f"✅ Dual analysis complete! Confidence: {analysis.confidence:.0%}")
        print(f"   📝 {len(analysis.actions)} actions proposed")

        if analysis.actions:
            print("   Top 3 actions:")
            for i, action in enumerate(analysis.actions[:3], 1):
                print(f"     {i}. [{action['tipo']}] {action['descrição']} (priority: {action['prioridade']}/10)")

    except Exception as e:
        print(f"⚠️  Auto Evolution failed (non-critical): {e}")
        print("   Backtest results are still valid.")

    # 6. Retornar resultado original
    return trades_df


def create_error_event(
    error_type: str,
    error_message: str,
    traceback_str: str,
    context: Optional[Dict] = None
) -> TradingEvent:
    """
    Cria um evento de erro para análise.

    Args:
        error_type: Tipo do erro (ex: "OrderRejected")
        error_message: Mensagem do erro
        traceback_str: Traceback completo
        context: Contexto adicional

    Returns:
        TradingEvent de erro
    """
    return TradingEvent(
        event_type=EventType.ERROR_OCCURRED,
        timestamp=datetime.now().isoformat(),
        data={
            "error_type": error_type,
            "message": error_message,
            "traceback": traceback_str,
        },
        context=context or {}
    )


# ============================================================================
# HELPER: Patch automático do selector21
# ============================================================================

def patch_selector21_backtest():
    """
    Patch automático que substitui backtest_from_signals por versão com auto-evolution.

    Usage:
        import backtest_integration
        backtest_integration.patch_selector21_backtest()

        # Agora todos os backtests do selector21 disparam auto-evolution!
    """
    try:
        import selector21

        # Guardar função original
        original_backtest = selector21.backtest_from_signals

        def patched_backtest(*args, **kwargs):
            """Versão patchada com auto-evolution"""
            # Extrair contexto dos kwargs
            strategy_name = kwargs.get("strategy_name", "unknown")
            timeframe = kwargs.get("timeframe", "unknown")

            return with_auto_evolution(
                backtest_func=original_backtest,
                strategy_name=strategy_name,
                timeframe=timeframe,
                enable_evolution=True,
                **kwargs
            )

        # Substituir
        selector21.backtest_from_signals = patched_backtest
        print("✅ selector21.backtest_from_signals patched with Auto Evolution!")

    except ImportError:
        print("⚠️  selector21 not available for patching")


if __name__ == "__main__":
    print("=" * 70)
    print("BACKTEST INTEGRATION - Auto Evolution System")
    print("=" * 70)
    print()
    print("This module provides seamless integration between backtests")
    print("and the Auto Evolution System (Claude + GPT).")
    print()
    print("Usage:")
    print("  1. Import: from backtest_integration import with_auto_evolution")
    print("  2. Wrap: trades = with_auto_evolution(backtest_from_signals, ...)")
    print("  3. Auto: Every backtest triggers dual AI analysis!")
    print()
    print("=" * 70)
