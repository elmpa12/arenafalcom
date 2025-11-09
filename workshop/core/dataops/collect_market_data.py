#!/usr/bin/env python3
"""
================================================================================
MARKET DATA COLLECTOR - Baseado no debate GPT-Strategist vs GPT-Executor
================================================================================

Coleta dados de mercado da Binance Futures:
- Funding Rate (a cada 8h)
- Mark Price vs Spot (1s)
- Open Interest (1m)
- Liquidations (real-time)

Decisões do debate:
- Funding: absoluto E delta
- Mark vs Spot: premium/discount em %
- OI: % change (não absoluto)
- Liquidations: ratio longs/shorts

Uso:
    python3 collect_market_data.py --symbol BTCUSDT --mode live --output-dir ./data/market

Author: Claude + GPT Dual Debate
Date: 2025-11-08
================================================================================
"""

import asyncio
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import pandas as pd
import numpy as np
import websockets
from binance.client import Client
from binance import ThreadedWebsocketManager
import pyarrow as pa
import pyarrow.parquet as pq
import time


class MarketDataCollector:
    """
    Coleta dados de mercado da Binance Futures.

    Schema:
    - timestamp: int64
    - funding_rate: float64
    - funding_rate_delta: float64 (vs anterior)
    - mark_price: float64
    - spot_price: float64 (BTCUSDT spot)
    - spot_premium_pct: float64 (%)
    - open_interest: float64
    - oi_change_pct: float64 (%)
    - liq_long_volume: float64 (último 1min)
    - liq_short_volume: float64
    - liq_ratio: float64 (longs/shorts)
    """

    def __init__(self, symbol: str, output_dir: str):
        self.symbol = symbol.upper()
        self.symbol_spot = symbol.upper()  # BTCUSDT
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = Client()

        # State
        self.last_funding_rate = None
        self.last_open_interest = None
        self.liquidations = deque(maxlen=1000)  # Last 1000 liquidations

        # Buffer
        self.buffer = deque(maxlen=10000)
        self.last_save_time = datetime.now()

        print(f"[MarketDataCollector] Iniciado para {self.symbol}")
        print(f"[MarketDataCollector] Output: {self.output_dir}")

    def get_funding_rate(self) -> dict:
        """
        Obtém funding rate atual.

        Decisão do debate: usar absoluto E delta
        """
        try:
            # Get current funding rate
            funding = self.client.futures_funding_rate(symbol=self.symbol, limit=2)

            if not funding:
                return None

            current = float(funding[0]['fundingRate'])
            previous = float(funding[1]['fundingRate']) if len(funding) > 1 else current

            delta = current - previous

            return {
                'funding_rate': current,
                'funding_rate_delta': delta,
                'funding_time': funding[0]['fundingTime']
            }

        except Exception as e:
            print(f"[ERROR] Funding rate: {e}")
            return None

    def get_mark_and_spot_price(self) -> dict:
        """
        Obtém mark price (futures) e spot price.

        Decisão do debate: premium/discount em %
        """
        try:
            # Mark price (futures)
            mark_price_data = self.client.futures_mark_price(symbol=self.symbol)
            mark_price = float(mark_price_data['markPrice'])

            # Spot price
            spot_ticker = self.client.get_symbol_ticker(symbol=self.symbol_spot)
            spot_price = float(spot_ticker['price'])

            # Premium/discount (%)
            # Positivo = futures mais caro (bullish)
            # Negativo = futures mais barato (bearish)
            spot_premium_pct = ((mark_price - spot_price) / spot_price) * 100

            return {
                'mark_price': mark_price,
                'spot_price': spot_price,
                'spot_premium_pct': spot_premium_pct
            }

        except Exception as e:
            print(f"[ERROR] Mark/Spot price: {e}")
            return None

    def get_open_interest(self) -> dict:
        """
        Obtém Open Interest.

        Decisão do debate: usar % change (não absoluto)
        """
        try:
            oi_data = self.client.futures_open_interest(symbol=self.symbol)
            current_oi = float(oi_data['openInterest'])

            # Calculate % change
            if self.last_open_interest is not None:
                oi_change_pct = ((current_oi - self.last_open_interest) / self.last_open_interest) * 100
            else:
                oi_change_pct = 0.0

            self.last_open_interest = current_oi

            return {
                'open_interest': current_oi,
                'oi_change_pct': oi_change_pct
            }

        except Exception as e:
            print(f"[ERROR] Open Interest: {e}")
            return None

    def process_liquidation(self, msg: dict):
        """
        Processa evento de liquidação.

        Decisão do debate: volume de liq longs vs shorts
        """
        try:
            order = msg['o']

            liq = {
                'timestamp': msg['E'],  # Event time
                'side': order['S'],  # SELL = long liquidation, BUY = short liquidation
                'price': float(order['p']),
                'quantity': float(order['q']),
                'value': float(order['p']) * float(order['q'])
            }

            self.liquidations.append(liq)

        except Exception as e:
            print(f"[ERROR] Liquidation: {e}")

    def get_liquidation_stats(self) -> dict:
        """
        Calcula estatísticas de liquidações (último 1min).

        Decisão do debate: ratio longs/shorts
        """
        if not self.liquidations:
            return {
                'liq_long_volume': 0.0,
                'liq_short_volume': 0.0,
                'liq_ratio': 0.0
            }

        # Filtra last 1min
        now = int(time.time() * 1000)
        one_min_ago = now - 60000

        recent_liqs = [l for l in self.liquidations if l['timestamp'] >= one_min_ago]

        # Separar por lado
        # SELL = long liquidation
        # BUY = short liquidation
        liq_long_vol = sum(l['value'] for l in recent_liqs if l['side'] == 'SELL')
        liq_short_vol = sum(l['value'] for l in recent_liqs if l['side'] == 'BUY')

        # Ratio
        # > 1: mais longs liquidados (bearish)
        # < 1: mais shorts liquidados (bullish)
        liq_ratio = liq_long_vol / liq_short_vol if liq_short_vol > 0 else float('inf')

        return {
            'liq_long_volume': liq_long_vol,
            'liq_short_volume': liq_short_vol,
            'liq_ratio': liq_ratio
        }

    async def collect_live(self):
        """Coleta todos os dados em tempo real"""

        print("[LIVE] Iniciando coleta de market data...")

        # Setup WebSocket for liquidations
        twm = ThreadedWebsocketManager()
        twm.start()

        # Subscribe to liquidation stream
        twm.start_futures_socket(
            callback=lambda msg: self.process_liquidation(msg),
            socket_type='forceOrder',
            symbol=self.symbol.lower()
        )

        print("[WS] Subscribed to liquidation stream")

        # Main loop
        last_funding_check = datetime.now()
        last_oi_check = datetime.now()

        while True:
            try:
                timestamp = int(time.time() * 1000)

                # Collect all data
                data_point = {'timestamp': timestamp}

                # 1. Mark and Spot Price (1s interval)
                mark_spot = self.get_mark_and_spot_price()
                if mark_spot:
                    data_point.update(mark_spot)

                # 2. Open Interest (1m interval)
                now = datetime.now()
                if (now - last_oi_check).total_seconds() >= 60:
                    oi = self.get_open_interest()
                    if oi:
                        data_point.update(oi)
                    last_oi_check = now
                else:
                    # Use last values
                    data_point['open_interest'] = self.last_open_interest or 0.0
                    data_point['oi_change_pct'] = 0.0

                # 3. Funding Rate (8h interval, but check every 1min)
                if (now - last_funding_check).total_seconds() >= 60:
                    funding = self.get_funding_rate()
                    if funding:
                        data_point['funding_rate'] = funding['funding_rate']
                        data_point['funding_rate_delta'] = funding['funding_rate_delta']

                        if self.last_funding_rate != funding['funding_rate']:
                            print(f"[FUNDING] Updated: {funding['funding_rate']:.6f} (delta: {funding['funding_rate_delta']:.6f})")
                            self.last_funding_rate = funding['funding_rate']

                    last_funding_check = now
                else:
                    data_point['funding_rate'] = self.last_funding_rate or 0.0
                    data_point['funding_rate_delta'] = 0.0

                # 4. Liquidation stats
                liq_stats = self.get_liquidation_stats()
                data_point.update(liq_stats)

                # Add to buffer
                self.buffer.append(data_point)

                # Save every 10 seconds
                if (now - self.last_save_time).total_seconds() >= 10.0:
                    await self.save_buffer()
                    self.last_save_time = now

                # Sleep 1s (consenso: mark price a cada 1s)
                await asyncio.sleep(1.0)

            except Exception as e:
                print(f"[ERROR] Main loop: {e}")
                await asyncio.sleep(1.0)

    async def save_buffer(self):
        """Salva buffer em Parquet"""
        if not self.buffer:
            return

        df = pd.DataFrame(list(self.buffer))

        if df.empty:
            return

        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Particionar por hora
        for hour, group in df.groupby(df['dt'].dt.floor('H')):
            year = hour.year
            month = hour.month
            day = hour.day
            hour_val = hour.hour

            partition_path = (
                self.output_dir /
                self.symbol /
                str(year) /
                f"{month:02d}" /
                f"{day:02d}" /
                f"hour={hour_val:02d}"
            )
            partition_path.mkdir(parents=True, exist_ok=True)

            save_df = group.drop(columns=['dt'])

            filepath = partition_path / "data.parquet"

            if filepath.exists():
                existing = pd.read_parquet(filepath)
                save_df = pd.concat([existing, save_df], ignore_index=True)
                save_df = save_df.drop_duplicates(subset=['timestamp'])

            save_df.to_parquet(
                filepath,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

        print(f"[SAVE] {len(df)} market data points salvos")

        self.buffer.clear()


def main():
    parser = argparse.ArgumentParser(description='Market Data Collector - Binance Futures')
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo')
    parser.add_argument('--mode', choices=['live'], default='live',
                        help='Modo de coleta')
    parser.add_argument('--output-dir', default='./data/market',
                        help='Diretório de saída')

    args = parser.parse_args()

    collector = MarketDataCollector(
        symbol=args.symbol,
        output_dir=args.output_dir
    )

    print("[MODE] LIVE - Coletando market data em tempo real...")
    asyncio.run(collector.collect_live())


if __name__ == "__main__":
    main()
