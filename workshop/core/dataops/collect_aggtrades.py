#!/usr/bin/env python3
"""
================================================================================
AGGTRADES COLLECTOR - Baseado no debate GPT-Strategist vs GPT-Executor
================================================================================

Coleta aggtrades da Binance Futures via WebSocket e salva em Parquet.

Decisões do debate:
- Frequência: 1s aggregate
- Storage: Parquet particionado por hora com compressão snappy
- Features: CVD, VWAP, buy/sell pressure, trade intensity, large trades

Uso:
    python3 collect_aggtrades.py --symbol BTCUSDT --days 90 --output-dir ./data/aggtrades

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


class AggTradesCollector:
    """
    Coleta aggtrades da Binance Futures e salva em Parquet particionado.

    Schema baseado no consenso do debate:
    - timestamp: int64 (Unix ms)
    - trade_id: int64
    - price: float64
    - quantity: float64
    - is_buyer_maker: bool (True = sell, False = buy)
    - first_trade_id: int64
    - last_trade_id: int64
    """

    def __init__(self, symbol: str, output_dir: str):
        self.symbol = symbol.upper()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffer para agregar a cada 1s
        self.buffer = deque(maxlen=10000)
        self.last_save_time = datetime.now()

        print(f"[AggTradesCollector] Iniciado para {self.symbol}")
        print(f"[AggTradesCollector] Output: {self.output_dir}")

    async def collect_live(self):
        """Coleta trades em tempo real via WebSocket"""
        uri = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"

        print(f"[WS] Conectando a {uri}...")

        async with websockets.connect(uri) as websocket:
            print("[WS] Conectado! Recebendo trades...")

            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    # Parse aggTrade
                    trade = {
                        'timestamp': data['T'],  # Trade time
                        'trade_id': data['a'],   # Aggregate trade ID
                        'price': float(data['p']),
                        'quantity': float(data['q']),
                        'is_buyer_maker': data['m'],  # True = sell
                        'first_trade_id': data['f'],
                        'last_trade_id': data['l'],
                    }

                    self.buffer.append(trade)

                    # Save a cada 1 segundo (consenso do debate)
                    now = datetime.now()
                    if (now - self.last_save_time).total_seconds() >= 1.0:
                        await self.save_buffer()
                        self.last_save_time = now

                except Exception as e:
                    print(f"[ERROR] {e}")
                    await asyncio.sleep(1)

    async def save_buffer(self):
        """Salva buffer em Parquet (particionado por hora)"""
        if not self.buffer:
            return

        # Converter buffer para DataFrame
        df = pd.DataFrame(list(self.buffer))

        if df.empty:
            return

        # Adicionar datetime column para particionamento
        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Particionar por hora
        for hour, group in df.groupby(df['dt'].dt.floor('H')):
            year = hour.year
            month = hour.month
            day = hour.day
            hour_val = hour.hour

            # Caminho: /aggtrades/BTCUSDT/2024/11/08/hour=14/data.parquet
            partition_path = (
                self.output_dir /
                self.symbol /
                str(year) /
                f"{month:02d}" /
                f"{day:02d}" /
                f"hour={hour_val:02d}"
            )
            partition_path.mkdir(parents=True, exist_ok=True)

            # Remover coluna datetime antes de salvar
            save_df = group.drop(columns=['dt'])

            # Salvar em Parquet com compressão snappy (consenso do debate)
            filepath = partition_path / "data.parquet"

            # Append mode
            if filepath.exists():
                existing = pd.read_parquet(filepath)
                save_df = pd.concat([existing, save_df], ignore_index=True)
                save_df = save_df.drop_duplicates(subset=['trade_id'])

            save_df.to_parquet(
                filepath,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

        print(f"[SAVE] {len(df)} trades salvos em {len(df.groupby(df['dt'].dt.floor('H')))} partition(s)")

        # Limpar buffer
        self.buffer.clear()

    def collect_historical(self, days: int = 90):
        """
        Coleta dados históricos via REST API

        Args:
            days: Número de dias para baixar (padrão 90, conforme contexto do debate)
        """
        print(f"[HISTORICAL] Baixando {days} dias de aggtrades...")

        client = Client()

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        current_time = start_time

        while current_time < end_time:
            # Binance limita a 1000 trades por request
            # Vamos buscar em chunks de 1 hora
            chunk_end = min(current_time + timedelta(hours=1), end_time)

            print(f"[FETCH] {current_time} → {chunk_end}")

            try:
                trades = client.futures_aggregate_trades(
                    symbol=self.symbol,
                    startTime=int(current_time.timestamp() * 1000),
                    endTime=int(chunk_end.timestamp() * 1000),
                    limit=1000
                )

                if trades:
                    # Converter para buffer
                    for t in trades:
                        self.buffer.append({
                            'timestamp': t['T'],
                            'trade_id': t['a'],
                            'price': float(t['p']),
                            'quantity': float(t['q']),
                            'is_buyer_maker': t['m'],
                            'first_trade_id': t['f'],
                            'last_trade_id': t['l'],
                        })

                    # Salvar a cada hora
                    asyncio.run(self.save_buffer())

                current_time = chunk_end

            except Exception as e:
                print(f"[ERROR] {e}")
                current_time += timedelta(minutes=1)

        print(f"[HISTORICAL] Concluído! {days} dias baixados.")


def main():
    parser = argparse.ArgumentParser(description='AggTrades Collector - Binance Futures')
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo (default: BTCUSDT)')
    parser.add_argument('--mode', choices=['live', 'historical'], default='historical',
                        help='Modo de coleta (default: historical)')
    parser.add_argument('--days', type=int, default=90,
                        help='Dias para baixar no modo historical (default: 90)')
    parser.add_argument('--output-dir', default='./data/aggtrades',
                        help='Diretório de saída (default: ./data/aggtrades)')

    args = parser.parse_args()

    collector = AggTradesCollector(
        symbol=args.symbol,
        output_dir=args.output_dir
    )

    if args.mode == 'live':
        print("[MODE] LIVE - Coletando em tempo real...")
        asyncio.run(collector.collect_live())
    else:
        print(f"[MODE] HISTORICAL - Baixando {args.days} dias...")
        collector.collect_historical(days=args.days)


if __name__ == "__main__":
    main()