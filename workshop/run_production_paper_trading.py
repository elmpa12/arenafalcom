#!/usr/bin/env python3
"""
🚀 PRODUCTION PAPER TRADING - BotScalp v3

Sistema COMPLETO de paper trading com:
1. Modelos ML/DL treinados (selector21 + dl_heads)
2. Signal Generator (gera sinais BUY/SELL/HOLD)
3. GPT Debate (valida cada sinal antes de executar)
4. Paper Trading Executor (executa na Binance Testnet)

Este é O SISTEMA REAL que você vai usar na competição!

Uso:
    # Primeiro, treine modelos:
    python3 selector21.py --symbol BTCUSDT --run_ml --ml_save_dir ./ml_models \\
        --walkforward --wf_train_months 3 --wf_val_months 1

    # Depois, rode paper trading:
    python3 run_production_paper_trading.py --trades 10
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Imports dos módulos
try:
    from model_signal_generator import ModelSignalGenerator, TradingSignal
    from paper_trading_executor import PaperTradingExecutor
    from claudex_dual_gpt import DualGPTOrchestrator
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    print("   Certifique-se de estar no diretório correto")
    sys.exit(1)


class ProductionPaperTradingSystem:
    """
    Sistema de Paper Trading de PRODUÇÃO

    Integra:
    - Signal Generator (ML/DL models)
    - GPT Debate (validation)
    - Paper Trading Executor (Binance Testnet)
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        models_dir: str = "./ml_models",
        timeframes: list = ["1m", "5m", "15m"],
        testnet: bool = True,
        use_debate: bool = True,
        min_confidence: float = 0.65
    ):
        """
        Initialize production trading system

        Args:
            symbol: Par para negociar
            models_dir: Diretório com modelos treinados
            timeframes: Lista de timeframes
            testnet: Se True, usa Binance Testnet
            use_debate: Se True, GPTs validam cada trade
            min_confidence: Confiança mínima para executar (0.0 a 1.0)
        """
        self.symbol = symbol
        self.testnet = testnet
        self.use_debate = use_debate
        self.min_confidence = min_confidence

        print("="*80)
        print("🚀 BOTSCALP V3 - PRODUCTION PAPER TRADING SYSTEM")
        print("="*80)
        print(f"Symbol: {symbol}")
        print(f"Models Dir: {models_dir}")
        print(f"Timeframes: {timeframes}")
        print(f"Mode: {'TESTNET (Paper)' if testnet else 'PRODUCTION ⚠️'}")
        print(f"Debate: {'ENABLED' if use_debate else 'DISABLED'}")
        print(f"Min Confidence: {min_confidence:.1%}")
        print("="*80 + "\n")

        # Inicializa componentes
        print("📦 Inicializando componentes...\n")

        # 1. Signal Generator (modelos treinados)
        print("   🧠 Carregando modelos ML/DL...")
        self.signal_generator = ModelSignalGenerator(
            models_dir=models_dir,
            symbol=symbol,
            timeframes=timeframes
        )

        # 2. Paper Trading Executor (exchange)
        print("\n   💰 Conectando com Binance Testnet...")
        self.executor = PaperTradingExecutor(testnet=testnet)

        # 3. Debate System (opcional)
        if use_debate:
            print("\n   💬 Inicializando Dual GPT Debate System...")
            self.debater = DualGPTOrchestrator()

        print("\n✅ Sistema pronto para trading!\n")

        # Estatísticas
        self.trades_executed = 0
        self.trades_approved = 0
        self.trades_rejected_low_confidence = 0
        self.trades_rejected_by_debate = 0
        self.session_log = []

    def validate_signal_with_debate(
        self,
        signal: TradingSignal,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        GPT-Strategist vs GPT-Executor debatem se devem executar

        Args:
            signal: Sinal gerado pelos modelos
            current_price: Preço atual

        Returns:
            (should_execute: bool, reasoning: str)
        """
        if not self.use_debate:
            return True, "Debate desabilitado"

        print("\n💬 Iniciando validação por debate GPT...")

        topic = f"""
VALIDAR SINAL DE TRADING:

Sinal gerado pelos modelos ML/DL:
- Ação: {signal.signal}
- Confiança: {signal.confidence:.1%}
- Método: {signal.method}
- Timeframe: {signal.timeframe}
- Symbol: {signal.symbol}
- Preço atual: ${current_price:,.2f}

Features consideradas:
{json.dumps(signal.features, indent=2)}

Prediction bruta: {signal.raw_prediction:.4f}

PERGUNTA: Devemos EXECUTAR este trade?

Considerem:
- A confiança do modelo ({signal.confidence:.1%}) é suficiente?
- O timing está adequado?
- Os riscos são aceitáveis?
- Há sinais conflitantes?

Respondam APENAS: "EXECUTAR" ou "REJEITAR" com breve justificativa (1-2 frases).
"""

        try:
            # Debate rápido (1 round)
            result = self.debater.debate_phase(topic, rounds=1)
            consensus = result.get('consensus', '')

            # Parse resposta
            if 'EXECUTAR' in consensus.upper():
                return True, consensus[:200]
            elif 'REJEITAR' in consensus.upper():
                return False, consensus[:200]
            else:
                # Inconclusivo: rejeita por segurança
                return False, "Consenso inconclusivo, rejeitando por segurança"

        except Exception as e:
            print(f"⚠️  Erro no debate: {e}")
            # Em caso de erro, rejeita por segurança
            return False, f"Debate falhou: {e}"

    def execute_trading_cycle(self) -> bool:
        """
        Executa um ciclo completo de trading:
        1. Gera sinais com modelos ML/DL
        2. Valida confiança mínima
        3. GPTs debatem se executam
        4. Executa na exchange
        5. Registra resultado

        Returns:
            True se executou trade, False caso contrário
        """
        print("\n" + "="*80)
        print(f"🎯 CICLO DE TRADING #{self.trades_executed + self.trades_approved + self.trades_rejected_low_confidence + self.trades_rejected_by_debate + 1}")
        print("="*80)

        # 1. Gera sinal de consenso (multi-timeframe)
        print("\n🧠 Gerando sinal de consenso dos modelos ML/DL...")

        signal = self.signal_generator.generate_consensus_signal()

        if not signal:
            print("⚠️  Nenhum sinal gerado (modelos não carregados ou dados insuficientes)")
            print("\n💡 AÇÃO NECESSÁRIA:")
            print("   1. Treinar modelos primeiro:")
            print("      python3 selector21.py --symbol BTCUSDT --run_ml --ml_save_dir ./ml_models \\")
            print("          --walkforward --wf_train_months 3 --wf_val_months 1 --wf_step_months 1")
            print("   2. Depois rode este script novamente")
            return False

        print(f"\n📋 SINAL GERADO:")
        print(f"   Decisão: {signal.signal}")
        print(f"   Confiança: {signal.confidence:.2%}")
        print(f"   Método: {signal.method}")
        print(f"   Timeframe: {signal.timeframe}")
        print(f"   Prediction: {signal.raw_prediction:.4f}")

        # Mostra features
        if signal.features:
            print(f"\n   Votos multi-timeframe:")
            for key, val in signal.features.items():
                print(f"      {key}: {val}")

        # 2. Verifica confiança mínima
        if signal.confidence < self.min_confidence:
            self.trades_rejected_low_confidence += 1
            print(f"\n❌ REJEITADO: Confiança {signal.confidence:.2%} < mínimo {self.min_confidence:.1%}")
            return False

        # 3. Verifica se é sinal neutro
        if signal.signal == "HOLD":
            print(f"\n⏸️  HOLD: Modelos indicam manter posição atual")
            return False

        # 4. Obtém preço atual
        current_price = self.executor.get_current_price(self.symbol)
        print(f"\n💰 Preço atual: ${current_price:,.2f}")

        # 5. Debate (validação GPT)
        execute, reasoning = self.validate_signal_with_debate(signal, current_price)

        print(f"\n💡 DECISÃO DO DEBATE: {'✅ EXECUTAR' if execute else '❌ REJEITAR'}")
        print(f"   Raciocínio: {reasoning}")

        if not execute:
            self.trades_rejected_by_debate += 1
            return False

        self.trades_approved += 1

        # 6. Prepara ordem
        usdt_balance = self.executor.get_balance("USDT")
        usdt_to_use = min(100.0, usdt_balance['free'] * 0.1)  # 10% do saldo, max $100
        quantity = round(usdt_to_use / current_price, 6)

        print(f"\n⚡ PREPARANDO EXECUÇÃO:")
        print(f"   Ação: {signal.signal}")
        print(f"   Investimento: ${usdt_to_use:.2f} USDT")
        print(f"   Quantidade: {quantity} {self.symbol.replace('USDT', '')}")
        print(f"   Preço: ${current_price:,.2f}")

        # 7. Confirmação final
        print(f"\n⚠️  CONFIRMAR EXECUÇÃO NO TESTNET? (y/n)")
        confirm = input("   > ").strip().lower()

        if confirm != 'y':
            print("   ⏸️  Execução cancelada pelo usuário")
            return False

        # 8. EXECUTA!
        proposal = {
            "symbol": self.symbol,
            "proposed_action": signal.signal,
            "quantity": quantity,
            "entry_price": current_price,
            "confidence": signal.confidence
        }

        success, order = self.executor.execute_trade_proposal(proposal)

        if success:
            self.trades_executed += 1
            print("\n✅ TRADE EXECUTADO COM SUCESSO!")
            print(f"   Order ID: {order.get('orderId')}")
            print(f"   Status: {order.get('status')}")

            # Log completo
            self.session_log.append({
                "timestamp": datetime.now().isoformat(),
                "signal": signal.to_dict(),
                "debate_reasoning": reasoning,
                "order": order,
                "price_at_execution": current_price
            })

            return True
        else:
            print("\n❌ Falha na execução")
            return False

    def run_session(self, num_trades: int = 10, wait_seconds: int = 60):
        """
        Executa sessão de trading com modelos de produção

        Args:
            num_trades: Número máximo de trades
            wait_seconds: Segundos entre verificações
        """
        print(f"\n{'='*80}")
        print(f"📈 INICIANDO SESSÃO DE PRODUCTION PAPER TRADING")
        print(f"   Target: {num_trades} trades executados")
        print(f"   Intervalo: {wait_seconds}s entre ciclos")
        print(f"{'='*80}\n")

        start_time = time.time()
        cycles = 0

        while self.trades_executed < num_trades:
            cycles += 1

            try:
                print(f"\n⏰ Ciclo #{cycles}")
                success = self.execute_trading_cycle()

                # Pausa entre ciclos (a não ser que seja o último)
                if self.trades_executed < num_trades:
                    print(f"\n⏳ Aguardando {wait_seconds}s até próximo ciclo...")
                    time.sleep(wait_seconds)

            except KeyboardInterrupt:
                print("\n\n⚠️  Sessão interrompida pelo usuário")
                break
            except Exception as e:
                print(f"\n❌ Erro no ciclo: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(wait_seconds)
                continue

        # Estatísticas finais
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print("📊 ESTATÍSTICAS DA SESSÃO")
        print(f"{'='*80}")
        print(f"Duração: {elapsed/60:.1f} minutos")
        print(f"Ciclos executados: {cycles}")
        print(f"Trades executados: {self.trades_executed}")
        print(f"Trades aprovados (aguardando execução): {self.trades_approved - self.trades_executed}")
        print(f"Rejeitados (baixa confiança): {self.trades_rejected_low_confidence}")
        print(f"Rejeitados (debate): {self.trades_rejected_by_debate}")

        if self.trades_executed > 0:
            print(f"\nTaxa de aprovação: {(self.trades_approved/(cycles)*100):.1f}%")
            print(f"Taxa de execução: {(self.trades_executed/cycles*100):.1f}%")

        # Salva log
        log_file = Path(f"production_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w') as f:
            json.dump({
                "session_stats": {
                    "symbol": self.symbol,
                    "duration_minutes": elapsed/60,
                    "cycles": cycles,
                    "trades_executed": self.trades_executed,
                    "trades_approved": self.trades_approved,
                    "trades_rejected_low_confidence": self.trades_rejected_low_confidence,
                    "trades_rejected_by_debate": self.trades_rejected_by_debate
                },
                "trades": self.session_log
            }, f, indent=2)

        print(f"\n📄 Log salvo em: {log_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='BotScalp v3 Production Paper Trading')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Par para negociar')
    parser.add_argument('--models-dir', type=str, default='./ml_models', help='Diretório com modelos')
    parser.add_argument('--trades', type=int, default=10, help='Número de trades')
    parser.add_argument('--wait', type=int, default=60, help='Segundos entre ciclos')
    parser.add_argument('--min-confidence', type=float, default=0.65, help='Confiança mínima (0-1)')
    parser.add_argument('--no-debate', action='store_true', help='Desabilita validação por debate')
    parser.add_argument('--production', action='store_true', help='⚠️  USA PRODUÇÃO (CUIDADO!)')

    args = parser.parse_args()

    # Inicializa sistema
    system = ProductionPaperTradingSystem(
        symbol=args.symbol,
        models_dir=args.models_dir,
        timeframes=["1m", "5m", "15m"],
        testnet=not args.production,
        use_debate=not args.no_debate,
        min_confidence=args.min_confidence
    )

    # Roda sessão
    system.run_session(num_trades=args.trades, wait_seconds=args.wait)


if __name__ == "__main__":
    main()
