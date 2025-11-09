#!/usr/bin/env python3
"""
Paper Trading Executor - Binance Testnet Integration
Conecta o competitive_trader.py com Binance Testnet para paper trading real
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    print("⚠️  Installing python-binance...")
    os.system("pip3 install -q python-binance")
    from binance.client import Client
    from binance.exceptions import BinanceAPIException

load_dotenv()


class PaperTradingExecutor:
    """
    Executor de Paper Trading usando Binance Testnet
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize paper trading executor

        Args:
            testnet: Se True, usa Binance Testnet (padrão)
        """
        self.testnet = testnet

        # Carrega credenciais
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY e BINANCE_API_SECRET devem estar no .env")

        # Conecta com Binance
        if testnet:
            print("🧪 Conectando com Binance TESTNET (paper trading)...")
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = 'https://testnet.binance.vision/api'
        else:
            print("⚠️  ATENÇÃO: Conectando com Binance PRODUÇÃO!")
            self.client = Client(api_key, api_secret)

        # Testa conexão
        try:
            account = self.client.get_account()
            print(f"✅ Conectado! Balances disponíveis:")
            for balance in account['balances'][:5]:
                if float(balance['free']) > 0:
                    print(f"   {balance['asset']}: {balance['free']}")
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")
            raise

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Obtém preço atual do par"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            print(f"❌ Erro ao obter preço: {e}")
            return 0.0

    def get_balance(self, asset: str = "USDT") -> Dict:
        """Obtém saldo disponível"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return {
                        'asset': asset,
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
            return {'asset': asset, 'free': 0.0, 'locked': 0.0, 'total': 0.0}
        except BinanceAPIException as e:
            print(f"❌ Erro ao obter saldo: {e}")
            return {'asset': asset, 'free': 0.0, 'locked': 0.0, 'total': 0.0}

    def place_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" ou "SELL"
        quantity: float
    ) -> Optional[Dict]:
        """
        Coloca ordem a mercado

        Args:
            symbol: Par (ex: "BTCUSDT")
            side: "BUY" ou "SELL"
            quantity: Quantidade (em base asset, ex: BTC)

        Returns:
            Dict com informações da ordem ou None se falhar
        """
        try:
            print(f"📤 Colocando ordem: {side} {quantity} {symbol}")

            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            print(f"✅ Ordem executada! ID: {order['orderId']}")
            print(f"   Status: {order['status']}")
            print(f"   Filled: {order['executedQty']} @ avg price {order.get('price', 'market')}")

            return order

        except BinanceAPIException as e:
            print(f"❌ Erro ao executar ordem: {e}")
            return None

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Optional[Dict]:
        """
        Coloca ordem limite
        """
        try:
            print(f"📤 Colocando ordem LIMIT: {side} {quantity} {symbol} @ {price}")

            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price
            )

            print(f"✅ Ordem limit criada! ID: {order['orderId']}")
            return order

        except BinanceAPIException as e:
            print(f"❌ Erro ao criar ordem limit: {e}")
            return None

    def get_order_status(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Consulta status de ordem"""
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return order
        except BinanceAPIException as e:
            print(f"❌ Erro ao consultar ordem: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancela ordem"""
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            print(f"✅ Ordem {order_id} cancelada")
            return True
        except BinanceAPIException as e:
            print(f"❌ Erro ao cancelar ordem: {e}")
            return False

    def execute_trade_proposal(
        self,
        proposal: Dict
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Executa proposta de trade do competitive_trader

        Args:
            proposal: Dict com {
                "symbol": "BTCUSDT",
                "proposed_action": "BUY",
                "entry_price": 50000.0,
                "quantity": 0.001,
                "confidence": 0.85
            }

        Returns:
            (success, order_info)
        """
        symbol = proposal.get("symbol", "BTCUSDT")
        side = proposal.get("proposed_action", "BUY")
        quantity = proposal.get("quantity", 0.001)

        # Valida saldo antes
        if side == "BUY":
            usdt_balance = self.get_balance("USDT")
            price = self.get_current_price(symbol)
            required = quantity * price

            if usdt_balance['free'] < required:
                print(f"❌ Saldo insuficiente! Precisa {required} USDT, tem {usdt_balance['free']}")
                return False, None

        # Executa ordem
        order = self.place_market_order(symbol, side, quantity)

        if order:
            return True, order
        else:
            return False, None


def test_paper_trading():
    """Testa sistema de paper trading"""
    print("="*70)
    print("🧪 TESTE DE PAPER TRADING - BINANCE TESTNET")
    print("="*70)

    # Inicializa executor
    executor = PaperTradingExecutor(testnet=True)

    # Testa consultas
    print("\n📊 Preço atual BTCUSDT:")
    price = executor.get_current_price("BTCUSDT")
    print(f"   BTC: ${price:,.2f}")

    print("\n💰 Saldo USDT:")
    balance = executor.get_balance("USDT")
    print(f"   Livre: ${balance['free']:,.2f}")
    print(f"   Total: ${balance['total']:,.2f}")

    # Simula proposta de trade
    print("\n🎯 Simulando proposta de trade:")
    proposal = {
        "symbol": "BTCUSDT",
        "proposed_action": "BUY",
        "quantity": 0.001,  # 0.001 BTC
        "confidence": 0.85
    }

    print(f"   Ação: {proposal['proposed_action']}")
    print(f"   Quantidade: {proposal['quantity']} BTC")
    print(f"   Confiança: {proposal['confidence']*100}%")

    # Pergunta confirmação
    print("\n⚠️  Deseja REALMENTE executar este trade no testnet? (y/n)")
    confirm = input("> ").strip().lower()

    if confirm == 'y':
        success, order = executor.execute_trade_proposal(proposal)
        if success:
            print("\n✅ TRADE EXECUTADO COM SUCESSO!")
            print(json.dumps(order, indent=2))
        else:
            print("\n❌ Trade falhou")
    else:
        print("\n⏸️  Trade cancelado pelo usuário")

    print("\n" + "="*70)
    print("✅ Teste concluído!")
    print("="*70)


if __name__ == "__main__":
    test_paper_trading()
