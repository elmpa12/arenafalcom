#!/usr/bin/env python3
"""
🔥 HIGH FREQUENCY TRADING - BotScalp v3

Sistema de alta frequência para COMPETIÇÃO:
- 30+ trades POR DIA (objetivo: 60 transações = 30 ciclos completos)
- Execução AUTOMÁTICA (sem confirmação manual)
- Ciclos completos: BUY → SELL automático
- Stop Loss e Take Profit automatizados
- Roda 24/7 em modo daemon

MODO COMPETIÇÃO ATIVADO! 🏆

Uso:
    # Modo automático contínuo (recomendado)
    python3 run_high_frequency_trading.py --auto --target-trades-per-day 30

    # Modo agressivo (mais trades)
    python3 run_high_frequency_trading.py --auto --target-trades-per-day 60 --min-confidence 0.55
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import argparse

try:
    from model_signal_generator import ModelSignalGenerator, TradingSignal
    from paper_trading_executor import PaperTradingExecutor
    from claudex_dual_gpt import DualGPTOrchestrator
except ImportError as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)


class Position:
    """Gerencia posição aberta"""
    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_time: str,
        order_id: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.symbol = symbol
        self.side = side  # "BUY" ou "SELL"
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.order_id = order_id
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.closed = False
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.pnl_pct = 0.0

    def check_exit_conditions(self, current_price: float) -> Tuple[bool, str]:
        """
        Verifica se deve fechar posição

        Returns:
            (should_close: bool, reason: str)
        """
        if self.side == "BUY":
            # Posição long
            if self.take_profit and current_price >= self.take_profit:
                return True, "TAKE_PROFIT"
            if self.stop_loss and current_price <= self.stop_loss:
                return True, "STOP_LOSS"
        else:
            # Posição short
            if self.take_profit and current_price <= self.take_profit:
                return True, "TAKE_PROFIT"
            if self.stop_loss and current_price >= self.stop_loss:
                return True, "STOP_LOSS"

        return False, ""

    def close_position(self, exit_price: float, exit_time: str):
        """Fecha posição e calcula P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.closed = True

        if self.side == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity

        self.pnl_pct = (self.pnl / (self.entry_price * self.quantity)) * 100

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'order_id': self.order_id,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'closed': self.closed
        }


class HighFrequencyTradingSystem:
    """
    Sistema de alta frequência para competição

    Target: 30+ trades/dia, 60 transações (30 ciclos completos)
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        models_dir: str = "./ml_models",
        timeframes: List[str] = ["1m", "5m"],  # Mais rápido: só 1m e 5m
        testnet: bool = True,
        auto_mode: bool = True,
        target_trades_per_day: int = 30,
        min_confidence: float = 0.60,  # Mais permissivo para volume
        atr_stop_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.0,
        position_size_pct: float = 0.02,  # 2% do saldo por trade
        max_position_time_minutes: int = 30  # Força fechar após 30min
    ):
        """
        Initialize high frequency system

        Args:
            auto_mode: Se True, executa sem confirmação manual
            target_trades_per_day: Meta de trades por dia
            min_confidence: Confiança mínima (mais baixo = mais trades)
            position_size_pct: % do saldo por trade
            max_position_time_minutes: Tempo max de posição aberta
        """
        self.symbol = symbol
        self.auto_mode = auto_mode
        self.target_trades_per_day = target_trades_per_day
        self.min_confidence = min_confidence
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.position_size_pct = position_size_pct
        self.max_position_time_minutes = max_position_time_minutes

        # Calcula intervalo entre trades
        minutes_per_day = 24 * 60
        self.check_interval_seconds = max(10, int((minutes_per_day / target_trades_per_day) * 60))

        print("="*80)
        print("🔥 HIGH FREQUENCY TRADING - MODO COMPETIÇÃO")
        print("="*80)
        print(f"Symbol: {symbol}")
        print(f"Mode: {'AUTO (sem confirmação)' if auto_mode else 'MANUAL'}")
        print(f"Target: {target_trades_per_day} trades/dia")
        print(f"Interval: ~{self.check_interval_seconds}s entre verificações")
        print(f"Min Confidence: {min_confidence:.0%}")
        print(f"Position Size: {position_size_pct:.1%} do saldo")
        print(f"Max Position Time: {max_position_time_minutes} min")
        print(f"Timeframes: {timeframes}")
        print("="*80 + "\n")

        # Componentes
        print("📦 Inicializando componentes...\n")

        self.signal_generator = ModelSignalGenerator(
            models_dir=models_dir,
            symbol=symbol,
            timeframes=timeframes
        )

        print("\n   💰 Conectando exchange...")
        self.executor = PaperTradingExecutor(testnet=testnet)

        # Debate apenas se não for auto (muito lento)
        self.debater = None
        if not auto_mode:
            print("\n   💬 Inicializando debate (modo manual)...")
            self.debater = DualGPTOrchestrator()

        print("\n✅ Sistema HIGH FREQUENCY pronto!\n")

        # Estado
        self.current_position: Optional[Position] = None
        self.closed_positions: List[Position] = []
        self.trades_today = 0
        self.day_start = datetime.now().date()
        self.running = False

        # Stats
        self.total_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0

    def calculate_stop_and_tp(
        self,
        entry_price: float,
        side: str,
        atr: float = None
    ) -> Tuple[float, float]:
        """
        Calcula stop loss e take profit baseado em ATR

        Args:
            entry_price: Preço de entrada
            side: "BUY" ou "SELL"
            atr: ATR atual (se None, usa 1% do preço)

        Returns:
            (stop_loss, take_profit)
        """
        if atr is None:
            atr = entry_price * 0.01  # 1% como fallback

        if side == "BUY":
            stop_loss = entry_price - (atr * self.atr_stop_multiplier)
            take_profit = entry_price + (atr * self.atr_tp_multiplier)
        else:
            stop_loss = entry_price + (atr * self.atr_stop_multiplier)
            take_profit = entry_price - (atr * self.atr_tp_multiplier)

        return stop_loss, take_profit

    def open_position(self, signal: TradingSignal) -> bool:
        """
        Abre nova posição baseada no sinal

        Returns:
            True se abriu, False se falhou
        """
        if self.current_position and not self.current_position.closed:
            print("⚠️  Já há posição aberta!")
            return False

        # Obtém preço atual
        current_price = self.executor.get_current_price(self.symbol)

        # Calcula tamanho da posição
        balance = self.executor.get_balance("USDT")
        usdt_to_use = balance['free'] * self.position_size_pct
        quantity = round(usdt_to_use / current_price, 6)

        print(f"\n🔓 ABRINDO POSIÇÃO:")
        print(f"   Side: {signal.signal}")
        print(f"   Price: ${current_price:,.2f}")
        print(f"   Quantity: {quantity}")
        print(f"   Size: ${usdt_to_use:.2f}")
        print(f"   Confidence: {signal.confidence:.1%}")

        # Calcula SL e TP
        atr = signal.features.get('atr', current_price * 0.01)
        stop_loss, take_profit = self.calculate_stop_and_tp(
            current_price, signal.signal, atr
        )

        print(f"   Stop Loss: ${stop_loss:,.2f}")
        print(f"   Take Profit: ${take_profit:,.2f}")

        # Confirmação (só se não for auto)
        if not self.auto_mode:
            print(f"\n⚠️  Confirmar? (y/n)")
            if input("   > ").strip().lower() != 'y':
                print("   ⏸️  Cancelado")
                return False

        # Executa ordem
        proposal = {
            "symbol": self.symbol,
            "proposed_action": signal.signal,
            "quantity": quantity,
            "entry_price": current_price,
            "confidence": signal.confidence
        }

        success, order = self.executor.execute_trade_proposal(proposal)

        if success:
            # Cria posição
            self.current_position = Position(
                symbol=self.symbol,
                side=signal.signal,
                entry_price=current_price,
                quantity=quantity,
                entry_time=datetime.now().isoformat(),
                order_id=order.get('orderId', 0),
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            self.trades_today += 1
            print(f"\n✅ POSIÇÃO ABERTA! (Trade #{self.trades_today} hoje)")
            return True
        else:
            print(f"\n❌ Falha ao abrir posição")
            return False

    def close_position(self, reason: str = "MANUAL") -> bool:
        """
        Fecha posição atual

        Args:
            reason: Motivo do fechamento

        Returns:
            True se fechou, False se falhou
        """
        if not self.current_position or self.current_position.closed:
            print("⚠️  Nenhuma posição aberta!")
            return False

        # Preço atual
        current_price = self.executor.get_current_price(self.symbol)

        # Determina side da ordem de saída (inverso da entrada)
        exit_side = "SELL" if self.current_position.side == "BUY" else "BUY"

        print(f"\n🔒 FECHANDO POSIÇÃO:")
        print(f"   Reason: {reason}")
        print(f"   Entry: ${self.current_position.entry_price:,.2f}")
        print(f"   Exit: ${current_price:,.2f}")

        # Executa ordem de saída
        proposal = {
            "symbol": self.symbol,
            "proposed_action": exit_side,
            "quantity": self.current_position.quantity,
            "entry_price": current_price
        }

        success, order = self.executor.execute_trade_proposal(proposal)

        if success:
            # Fecha posição
            self.current_position.close_position(
                exit_price=current_price,
                exit_time=datetime.now().isoformat()
            )

            # Stats
            self.total_pnl += self.current_position.pnl
            if self.current_position.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            self.closed_positions.append(self.current_position)

            print(f"\n✅ POSIÇÃO FECHADA!")
            print(f"   P&L: ${self.current_position.pnl:.2f} ({self.current_position.pnl_pct:+.2f}%)")
            print(f"   Total P&L: ${self.total_pnl:.2f}")
            print(f"   Win Rate: {self.winning_trades}/{self.winning_trades + self.losing_trades} = {(self.winning_trades/(self.winning_trades + self.losing_trades)*100 if (self.winning_trades + self.losing_trades) > 0 else 0):.1f}%")

            self.current_position = None
            return True
        else:
            print(f"\n❌ Falha ao fechar posição")
            return False

    def check_position_exit(self):
        """Verifica se deve fechar posição atual"""
        if not self.current_position or self.current_position.closed:
            return

        current_price = self.executor.get_current_price(self.symbol)

        # 1. Verifica SL/TP
        should_close, reason = self.current_position.check_exit_conditions(current_price)
        if should_close:
            print(f"\n⚡ Triggered: {reason}")
            self.close_position(reason=reason)
            return

        # 2. Verifica tempo máximo
        entry_time = datetime.fromisoformat(self.current_position.entry_time)
        elapsed_minutes = (datetime.now() - entry_time).total_seconds() / 60

        if elapsed_minutes >= self.max_position_time_minutes:
            print(f"\n⏰ Tempo máximo atingido ({elapsed_minutes:.1f} min)")
            self.close_position(reason="MAX_TIME")
            return

    def trading_cycle(self):
        """Um ciclo completo de trading"""

        # 1. Se há posição aberta, verifica saída
        if self.current_position and not self.current_position.closed:
            self.check_position_exit()
            return  # Espera fechar antes de abrir nova

        # 2. Reset diário
        today = datetime.now().date()
        if today != self.day_start:
            print(f"\n📅 Novo dia! Trades ontem: {self.trades_today}")
            self.trades_today = 0
            self.day_start = today

        # 3. Verifica se atingiu meta do dia
        if self.trades_today >= self.target_trades_per_day:
            print(f"\n✅ Meta do dia atingida! ({self.trades_today}/{self.target_trades_per_day})")
            return

        # 4. Gera sinal
        signal = self.signal_generator.generate_consensus_signal()

        if not signal:
            return

        # 5. Filtros
        if signal.signal == "HOLD":
            return

        if signal.confidence < self.min_confidence:
            print(f"⏭️  Sinal rejeitado: conf {signal.confidence:.1%} < {self.min_confidence:.1%}")
            return

        print(f"\n📊 Sinal: {signal.signal} (conf: {signal.confidence:.1%})")

        # 6. Abre posição
        self.open_position(signal)

    def run_continuous(self):
        """Roda continuamente (modo daemon)"""
        print(f"\n{'='*80}")
        print(f"🔥 INICIANDO MODO HIGH FREQUENCY")
        print(f"   Meta: {self.target_trades_per_day} trades/dia")
        print(f"   Verificação: a cada {self.check_interval_seconds}s")
        print(f"{'='*80}\n")

        self.running = True
        cycle_count = 0

        try:
            while self.running:
                cycle_count += 1
                print(f"\n⏰ Ciclo #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"   Trades hoje: {self.trades_today}/{self.target_trades_per_day}")
                print(f"   P&L total: ${self.total_pnl:.2f}")

                try:
                    self.trading_cycle()
                except Exception as e:
                    print(f"❌ Erro no ciclo: {e}")
                    import traceback
                    traceback.print_exc()

                # Aguarda próximo ciclo
                time.sleep(self.check_interval_seconds)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrompido pelo usuário")
            self.running = False

        # Fecha posição aberta se houver
        if self.current_position and not self.current_position.closed:
            print("\n🔒 Fechando posição aberta...")
            self.close_position(reason="SHUTDOWN")

        # Stats finais
        self.print_final_stats()

    def print_final_stats(self):
        """Imprime estatísticas finais"""
        print(f"\n{'='*80}")
        print("📊 ESTATÍSTICAS FINAIS")
        print(f"{'='*80}")
        print(f"Trades hoje: {self.trades_today}")
        print(f"Trades fechados: {len(self.closed_positions)}")
        print(f"Winning: {self.winning_trades}")
        print(f"Losing: {self.losing_trades}")
        if self.winning_trades + self.losing_trades > 0:
            print(f"Win Rate: {(self.winning_trades/(self.winning_trades + self.losing_trades)*100):.1f}%")
        print(f"Total P&L: ${self.total_pnl:.2f}")

        # Salva log
        log_file = Path(f"hft_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w') as f:
            json.dump({
                "stats": {
                    "trades_today": self.trades_today,
                    "closed_positions": len(self.closed_positions),
                    "winning_trades": self.winning_trades,
                    "losing_trades": self.losing_trades,
                    "total_pnl": self.total_pnl
                },
                "positions": [p.to_dict() for p in self.closed_positions]
            }, f, indent=2)

        print(f"\n📄 Log: {log_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='High Frequency Trading System')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--models-dir', type=str, default='./ml_models')
    parser.add_argument('--auto', action='store_true', help='Modo automático (SEM confirmação)')
    parser.add_argument('--target-trades-per-day', type=int, default=30, help='Meta de trades/dia')
    parser.add_argument('--min-confidence', type=float, default=0.60, help='Confiança mínima')
    parser.add_argument('--position-size', type=float, default=0.02, help='% saldo por trade')
    parser.add_argument('--max-position-time', type=int, default=30, help='Minutos max por posição')
    parser.add_argument('--production', action='store_true', help='⚠️  PRODUÇÃO')

    args = parser.parse_args()

    if not args.auto:
        print("⚠️  Modo manual detectado. Para alta frequência, use --auto")

    system = HighFrequencyTradingSystem(
        symbol=args.symbol,
        models_dir=args.models_dir,
        testnet=not args.production,
        auto_mode=args.auto,
        target_trades_per_day=args.target_trades_per_day,
        min_confidence=args.min_confidence,
        position_size_pct=args.position_size,
        max_position_time_minutes=args.max_position_time
    )

    system.run_continuous()


if __name__ == "__main__":
    main()
