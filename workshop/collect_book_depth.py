#!/usr/bin/env python3
"""
================================================================================
BOOK DEPTH COLLECTOR - Baseado no debate GPT-Strategist vs GPT-Executor
================================================================================

Coleta snapshots do Order Book da Binance Futures e salva em Parquet.

Decisões do debate:
- Frequência: 500ms snapshots
- Levels: 20 levels (bids + asks)
- Storage: Parquet particionado por hora com compressão snappy
- Features: imbalance, spread, slopes, weighted mid price

Uso:
    # Tempo real
    python3 collect_book_depth.py --symbol BTCUSDT --mode live --output-dir ./data/book_depth

    # Histórico (snapshot inicial + reconstruir via diff stream)
    python3 collect_book_depth.py --symbol BTCUSDT --mode historical --hours 2160 --output-dir ./data/book_depth

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
import pyarrow as pa
import pyarrow.parquet as pq
import time


class BookDepthCollector:
    """
    Coleta snapshots do order book a cada 500ms.

    Schema baseado no consenso do debate:
    - timestamp: int64 (Unix ms)
    - last_update_id: int64
    - bids: object (JSON: [[price, qty], ...])
    - asks: object (JSON: [[price, qty], ...])
    - bid_vol_5/10/20: float64
    - ask_vol_5/10/20: float64
    - imbalance_5/10/20: float64
    - spread: float64
    - mid_price: float64
    - weighted_mid: float64
    """

    def __init__(self, symbol: str, output_dir: str, snapshot_interval_ms: int = 500):
        self.symbol = symbol.upper()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval_ms / 1000.0  # Convert to seconds

        # Buffer para snapshots
        self.buffer = deque(maxlen=10000)
        self.last_save_time = datetime.now()

        # Current order book state
        self.order_book = {'bids': [], 'asks': [], 'lastUpdateId': 0}

        print(f"[BookDepthCollector] Iniciado para {self.symbol}")
        print(f"[BookDepthCollector] Snapshot interval: {snapshot_interval_ms}ms")
        print(f"[BookDepthCollector] Output: {self.output_dir}")

    def compute_features(self, book: dict) -> dict:
        """
        Calcula features do order book conforme debate.

        Decisões do debate:
        - Imbalance em 5, 10, 20 levels
        - Spread (absoluto)
        - Mid price (best bid + best ask) / 2
        - Weighted mid price (ponderado por volume)
        """
        if not book['bids'] or not book['asks']:
            return None

        bids = book['bids'][:20]  # Top 20
        asks = book['asks'][:20]  # Top 20

        # Volumes em diferentes depths
        bid_vol_5 = sum(float(b[1]) for b in bids[:5])
        ask_vol_5 = sum(float(a[1]) for a in asks[:5])
        bid_vol_10 = sum(float(b[1]) for b in bids[:10])
        ask_vol_10 = sum(float(a[1]) for a in asks[:10])
        bid_vol_20 = sum(float(b[1]) for b in bids[:20])
        ask_vol_20 = sum(float(a[1]) for a in asks[:20])

        # Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        # Consenso do debate: esta é a fórmula correta
        def calc_imbalance(bvol, avol):
            total = bvol + avol
            return (bvol - avol) / total if total > 0 else 0.0

        imbalance_5 = calc_imbalance(bid_vol_5, ask_vol_5)
        imbalance_10 = calc_imbalance(bid_vol_10, ask_vol_10)
        imbalance_20 = calc_imbalance(bid_vol_20, ask_vol_20)

        # Prices
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])

        # Spread
        spread = best_ask - best_bid

        # Mid price
        mid_price = (best_bid + best_ask) / 2.0

        # Weighted mid price (ponderado por volume no best bid/ask)
        best_bid_vol = float(bids[0][1])
        best_ask_vol = float(asks[0][1])
        weighted_mid = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

        return {
            'bid_vol_5': bid_vol_5,
            'ask_vol_5': ask_vol_5,
            'bid_vol_10': bid_vol_10,
            'ask_vol_10': ask_vol_10,
            'bid_vol_20': bid_vol_20,
            'ask_vol_20': ask_vol_20,
            'imbalance_5': imbalance_5,
            'imbalance_10': imbalance_10,
            'imbalance_20': imbalance_20,
            'spread': spread,
            'mid_price': mid_price,
            'weighted_mid': weighted_mid,
        }

    async def collect_live(self):
        """Coleta order book em tempo real via WebSocket"""
        # Primeiro, get snapshot inicial
        client = Client()
        depth = client.futures_order_book(symbol=self.symbol, limit=20)

        self.order_book = {
            'bids': depth['bids'],
            'asks': depth['asks'],
            'lastUpdateId': depth['lastUpdateId']
        }

        print(f"[SNAPSHOT] Initial book loaded. LastUpdateId: {depth['lastUpdateId']}")

        # WebSocket para diff updates
        uri = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@depth@100ms"

        print(f"[WS] Conectando a {uri}...")

        async with websockets.connect(uri) as websocket:
            print("[WS] Conectado! Recebendo depth updates...")

            last_snapshot_time = time.time()

            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    # Update order book with diff
                    # (simplified - in production, need proper update logic)
                    if 'b' in data and 'a' in data:
                        # Update bids and asks
                        # This is simplified - real implementation needs proper merging
                        pass

                    # Take snapshot every 500ms (consenso do debate)
                    current_time = time.time()
                    if current_time - last_snapshot_time >= self.snapshot_interval:
                        # Get fresh snapshot from REST API (more reliable)
                        depth = client.futures_order_book(symbol=self.symbol, limit=20)

                        timestamp = int(time.time() * 1000)

                        # Compute features
                        features = self.compute_features(depth)

                        if features:
                            snapshot = {
                                'timestamp': timestamp,
                                'last_update_id': depth['lastUpdateId'],
                                'bids': json.dumps(depth['bids'][:20]),  # Store as JSON string
                                'asks': json.dumps(depth['asks'][:20]),
                                **features
                            }

                            self.buffer.append(snapshot)

                        last_snapshot_time = current_time

                        # Save every 10 seconds
                        now = datetime.now()
                        if (now - self.last_save_time).total_seconds() >= 10.0:
                            await self.save_buffer()
                            self.last_save_time = now

                except Exception as e:
                    print(f"[ERROR] {e}")
                    await asyncio.sleep(0.5)

    async def save_buffer(self):
        """Salva buffer em Parquet (particionado por hora)"""
        if not self.buffer:
            return

        df = pd.DataFrame(list(self.buffer))

        if df.empty:
            return

        # Adicionar datetime column
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

            # Append mode
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

        print(f"[SAVE] {len(df)} snapshots salvos")

        self.buffer.clear()

    def collect_snapshots_historical(self, hours: int = 2160):
        """
        Coleta snapshots históricos.

        NOTA: Binance não oferece histórico de book depth.
        Esta função coleta snapshots atuais a cada 500ms por N horas.

        Args:
            hours: Número de horas para coletar (default: 2160 = 90 dias)
        """
        print(f"[HISTORICAL] Coletando snapshots por {hours} horas...")
        print("[WARNING] Book depth histórico não está disponível na Binance.")
        print("[INFO] Para dados históricos, use modo LIVE por período prolongado.")

        # Para dados históricos reais, precisaria de:
        # 1. Dados de outro provider (como Kaiko, CryptoCompare)
        # 2. Ou reconstruir via trade data (menos preciso)

        print("[INFO] Iniciando coleta em modo LIVE...")
        asyncio.run(self.collect_live())


def main():
    parser = argparse.ArgumentParser(description='Book Depth Collector - Binance Futures')
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo')
    parser.add_argument('--mode', choices=['live'], default='live',
                        help='Modo de coleta (apenas live disponível)')
    parser.add_argument('--snapshot-interval', type=int, default=500,
                        help='Intervalo entre snapshots em ms (default: 500)')
    parser.add_argument('--output-dir', default='./data/book_depth',
                        help='Diretório de saída')

    args = parser.parse_args()

    collector = BookDepthCollector(
        symbol=args.symbol,
        output_dir=args.output_dir,
        snapshot_interval_ms=args.snapshot_interval
    )

    print("[MODE] LIVE - Coletando snapshots em tempo real...")
    asyncio.run(collector.collect_live())


if __name__ == "__main__":
    main()
