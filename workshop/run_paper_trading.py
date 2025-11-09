#!/usr/bin/env python3
"""
🚀 PAPER TRADING - BotScalp v3

Sistema completo de paper trading integrando:
- Claudex 2.0 (Claude + GPT debatem cada trade)
- Competitive Trader (análise + proposta)
- Paper Trading Executor (execução real no testnet)

Uso:
    python3 run_paper_trading.py --trades 5
    python3 run_paper_trading.py --symbol ETHUSDT --trades 3
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

# Import nossos módulos
try:
    from competitive_trader import CompetitiveTrader
    from paper_trading_executor import PaperTradingExecutor
    from claudex_dual_gpt import DualGPTOrchestrator
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    print("   Certifique-se de estar no diretório correto")
    sys.exit(1)


class PaperTradingSystem:
    """
    Sistema completo de Paper Trading com IA Dual
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        testnet: bool = True,
        use_debate: bool = True
    ):
        """
        Initialize paper trading system

        Args:
            symbol: Par para negociar (ex: "BTCUSDT")
            testnet: Se True, usa testnet (paper trading)
            use_debate: Se True, Claude + GPT debatem cada trade
        """
        self.symbol = symbol
        self.testnet = testnet
        self.use_debate = use_debate

        print("="*70)
        print("🚀 BOTSCALP V3 - PAPER TRADING SYSTEM")
        print("="*70)
        print(f"Symbol: {symbol}")
        print(f"Mode: {'TESTNET (Paper)' if testnet else 'PRODUCTION ⚠️'}")
        print(f"Debate: {'ENABLED' if use_debate else 'DISABLED'}")
        print("="*70 + "\n")

        # Inicializa componentes
        print("📦 Inicializando componentes...")

        # 1. Competitive Trader (análise + proposta)
        self.trader = CompetitiveTrader(initial_balance=10000.0)
        print("   ✅ Competitive Trader (Claude + GPT análise)")

        # 2. Paper Trading Executor (conexão com exchange)
        self.executor = PaperTradingExecutor(testnet=testnet)
        print("   ✅ Paper Trading Executor (Binance Testnet)")

        # 3. Debate System (opcional)
        if use_debate:
            self.debater = DualGPTOrchestrator()
            print("   ✅ Dual GPT Debate System (verificação dupla)")

        print("\n✅ Sistema pronto!\n")

        # Estatísticas
        self.trades_executed = 0
        self.trades_approved = 0
        self.trades_rejected = 0
        self.session_log = []

    def analyze_market(self) -> Dict:
        """Analisa mercado atual"""
        print(f"📊 Analisando mercado para {self.symbol}...")

        # Obtém preço atual
        current_price = self.executor.get_current_price(self.symbol)
        print(f"   Preço atual: ${current_price:,.2f}")

        # Obtém saldo
        balance = self.executor.get_balance("USDT")
        print(f"   Saldo disponível: ${balance['free']:,.2f} USDT")

        # Simula análise técnica (aqui você integraria selector21, etc)
        analysis = {
            "symbol": self.symbol,
            "current_price": current_price,
            "balance_usdt": balance['free'],
            "timestamp": datetime.now().isoformat(),
            # Métricas técnicas (mock por enquanto)
            "rsi": 55.0,
            "macd": "bullish",
            "volume_trend": "increasing",
            "regime": 1  # 1 = trend, 0 = range
        }

        return analysis

    def debate_trade_decision(self, proposal: Dict) -> tuple[bool, str]:
        """
        Claude + GPT debatem se devem executar o trade

        Returns:
            (execute: bool, reasoning: str)
        """
        if not self.use_debate:
            return True, "Debate desabilitado, executando automaticamente"

        print("\n💬 Iniciando debate GPT-Strategist vs GPT-Executor...")

        topic = f"""
Devemos executar este trade?

Symbol: {proposal['symbol']}
Ação: {proposal['proposed_action']}
Preço: ${proposal.get('entry_price', 0):,.2f}
Quantidade: {proposal.get('quantity', 0)}
Confiança: {proposal.get('confidence', 0)*100}%
Lógica: {proposal.get('entry_logic', 'N/A')}

Considerando:
- Riscos de mercado
- Tamanho da posição
- Timing de entrada
- Gestão de risco

Responda apenas: EXECUTAR ou REJEITAR com breve justificativa.
"""

        try:
            result = self.debater.debate_phase(topic, rounds=1)  # Debate rápido
            consensus = result.get('consensus', '')

            # Parseia consenso
            if 'EXECUTAR' in consensus.upper():
                return True, consensus
            elif 'REJEITAR' in consensus.upper():
                return False, consensus
            else:
                # Se inconclusivo, executa por padrão
                return True, "Consenso inconclusivo, executando por padrão"

        except Exception as e:
            print(f"⚠️  Erro no debate: {e}")
            return True, "Debate falhou, executando por padrão"

    def execute_single_trade(self) -> bool:
        """
        Executa um ciclo completo de trade:
        1. Analisa mercado
        2. Propõe trade
        3. Debate decisão
        4. Executa na exchange
        5. Registra resultado
        """
        print("\n" + "="*70)
        print(f"🎯 TRADE #{self.trades_executed + 1}")
        print("="*70)

        # 1. Análise de mercado
        analysis = self.analyze_market()

        # 2. Proposta de trade (usando competitive_trader)
        print("\n🧠 Gerando proposta de trade (Claude + GPT)...")
        proposal = self.trader.propose_trade(analysis)

        print(f"\n📋 PROPOSTA:")
        print(f"   Ação: {proposal.get('proposed_action')}")
        print(f"   Símbolo: {proposal.get('symbol')}")
        print(f"   Lógica: {proposal.get('entry_logic')}")
        print(f"   Confiança: {proposal.get('confidence', 0)*100:.1f}%")

        # 3. Debate (Claude + GPT decidem juntos)
        execute, reasoning = self.debate_trade_decision(proposal)

        print(f"\n💡 DECISÃO: {'✅ EXECUTAR' if execute else '❌ REJEITAR'}")
        print(f"   Raciocínio: {reasoning[:200]}...")

        if not execute:
            self.trades_rejected += 1
            print("\n⏭️  Trade rejeitado pelo debate\n")
            return False

        self.trades_approved += 1

        # 4. Execução na exchange (REAL!)
        print("\n⚡ Executando trade na exchange...")

        # Adiciona quantidade ao proposal
        current_price = analysis['current_price']
        usdt_to_use = min(100.0, analysis['balance_usdt'] * 0.1)  # Usa 10% do saldo, max $100
        quantity = round(usdt_to_use / current_price, 6)  # Quantidade em BTC/ETH

        proposal['quantity'] = quantity
        proposal['entry_price'] = current_price

        print(f"   Investindo: ${usdt_to_use:.2f} USDT")
        print(f"   Quantidade: {quantity} {self.symbol.replace('USDT', '')}")

        # CONFIRMAÇÃO FINAL
        print(f"\n⚠️  CONFIRMAR EXECUÇÃO REAL NO TESTNET? (y/n)")
        confirm = input("   > ").strip().lower()

        if confirm != 'y':
            print("   ⏸️  Execução cancelada pelo usuário")
            return False

        # EXECUTA!
        success, order = self.executor.execute_trade_proposal(proposal)

        if success:
            self.trades_executed += 1
            print("\n✅ TRADE EXECUTADO COM SUCESSO!")
            print(f"   Order ID: {order.get('orderId')}")
            print(f"   Status: {order.get('status')}")

            # Log
            self.session_log.append({
                "timestamp": datetime.now().isoformat(),
                "proposal": proposal,
                "order": order,
                "debate_reasoning": reasoning
            })

            return True
        else:
            print("\n❌ Falha na execução do trade")
            return False

    def run_session(self, num_trades: int = 5):
        """Executa sessão de paper trading"""
        print(f"\n{'='*70}")
        print(f"📈 INICIANDO SESSÃO DE PAPER TRADING")
        print(f"   Target: {num_trades} trades")
        print(f"{'='*70}\n")

        start_time = time.time()

        for i in range(num_trades):
            try:
                success = self.execute_single_trade()

                # Pausa entre trades
                if i < num_trades - 1:
                    wait_time = 5
                    print(f"\n⏳ Aguardando {wait_time}s até próximo trade...")
                    time.sleep(wait_time)

            except KeyboardInterrupt:
                print("\n\n⚠️  Sessão interrompida pelo usuário")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Estatísticas finais
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print("📊 ESTATÍSTICAS DA SESSÃO")
        print(f"{'='*70}")
        print(f"Duração: {elapsed/60:.1f} minutos")
        print(f"Trades executados: {self.trades_executed}")
        print(f"Trades aprovados: {self.trades_approved}")
        print(f"Trades rejeitados: {self.trades_rejected}")
        print(f"Taxa de aprovação: {(self.trades_approved/(self.trades_approved+self.trades_rejected)*100) if (self.trades_approved+self.trades_rejected) > 0 else 0:.1f}%")

        # Salva log
        log_file = Path(f"paper_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w') as f:
            json.dump({
                "session_stats": {
                    "symbol": self.symbol,
                    "duration_minutes": elapsed/60,
                    "trades_executed": self.trades_executed,
                    "trades_approved": self.trades_approved,
                    "trades_rejected": self.trades_rejected
                },
                "trades": self.session_log
            }, f, indent=2)

        print(f"\n📄 Log salvo em: {log_file}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='BotScalp v3 Paper Trading System')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Par para negociar')
    parser.add_argument('--trades', type=int, default=3, help='Número de trades')
    parser.add_argument('--no-debate', action='store_true', help='Desabilita debate (executa direto)')
    parser.add_argument('--production', action='store_true', help='⚠️  USA PRODUÇÃO (CUIDADO!)')

    args = parser.parse_args()

    # Inicializa sistema
    system = PaperTradingSystem(
        symbol=args.symbol,
        testnet=not args.production,
        use_debate=not args.no_debate
    )

    # Roda sessão
    system.run_session(num_trades=args.trades)


if __name__ == "__main__":
    main()
