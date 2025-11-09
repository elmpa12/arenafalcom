#!/usr/bin/env python3
"""
Download Binance Historical Data

Baixa klines (OHLCV) da Binance e salva em parquet para uso no selector21.py

Uso:
    # Baixar 3 meses de BTCUSDT 1m
    python3 download_binance_data.py --symbol BTCUSDT --timeframe 1m --days 90

    # Baixar ano completo, múltiplos timeframes
    python3 download_binance_data.py --symbol BTCUSDT --timeframe 1m,5m,15m --days 365
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

try:
    from binance.client import Client
except ImportError:
    print("⚠️  Installing python-binance...")
    os.system("pip3 install -q python-binance")
    from binance.client import Client

try:
    import pyarrow.parquet as pq
except ImportError:
    print("⚠️  Installing pyarrow...")
    os.system("pip3 install -q pyarrow")
    import pyarrow.parquet as pq


class BinanceDataDownloader:
    """
    Download historical klines from Binance and save as parquet
    """

    TIMEFRAME_MAPPING = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
    }

    def __init__(self, output_dir: str = "./data"):
        """
        Initialize downloader

        Args:
            output_dir: Diretório para salvar parquets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cliente Binance (public API, sem keys necessárias)
        self.client = Client("", "")  # Public endpoints

        print(f"📁 Dados serão salvos em: {self.output_dir.absolute()}")

    def download_klines(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Download klines do Binance

        Args:
            symbol: Par (ex: "BTCUSDT")
            timeframe: Timeframe (ex: "1m", "5m", "15m")
            start_date: Data inicial
            end_date: Data final

        Returns:
            DataFrame com OHLCV
        """
        print(f"\n📥 Baixando {symbol} {timeframe}...")
        print(f"   Período: {start_date.date()} → {end_date.date()}")

        interval = self.TIMEFRAME_MAPPING.get(timeframe)
        if not interval:
            raise ValueError(f"Timeframe inválido: {timeframe}")

        # Baixa em chunks (Binance limita a 1000 klines por request)
        all_klines = []
        current_start = start_date

        chunk_count = 0
        while current_start < end_date:
            try:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=int(current_start.timestamp() * 1000),
                    end_str=int(end_date.timestamp() * 1000),
                    limit=1000
                )

                if not klines:
                    break

                all_klines.extend(klines)
                chunk_count += 1

                # Atualiza start para próximo chunk
                last_time = klines[-1][0]
                current_start = datetime.fromtimestamp(last_time / 1000)

                # Rate limiting
                time.sleep(0.2)

                # Progress
                days_done = (current_start - start_date).days
                total_days = (end_date - start_date).days
                progress = (days_done / total_days * 100) if total_days > 0 else 100
                print(f"   Progress: {progress:.1f}% ({len(all_klines)} klines)", end='\r')

            except Exception as e:
                print(f"\n   ⚠️  Erro no chunk {chunk_count}: {e}")
                time.sleep(5)  # Wait e retry
                continue

        print(f"\n   ✅ Baixados {len(all_klines)} klines em {chunk_count} chunks")

        # Converte para DataFrame
        if not all_klines:
            print("   ⚠️  Nenhum dado baixado!")
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # Converte tipos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove duplicatas
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        # Ordena por timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Path:
        """
        Salva DataFrame em parquet

        Args:
            df: DataFrame com klines
            symbol: Par
            timeframe: Timeframe

        Returns:
            Path do arquivo salvo
        """
        if df.empty:
            print("   ⚠️  DataFrame vazio, não salvando")
            return None

        # Nome do arquivo: BTCUSDT_1m.parquet
        filename = f"{symbol}_{timeframe}.parquet"
        filepath = self.output_dir / filename

        print(f"\n💾 Salvando em: {filepath}")

        # Seleciona colunas relevantes
        df_save = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Salva
        df_save.to_parquet(
            filepath,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"   ✅ Salvo! ({file_size_mb:.2f} MB, {len(df)} rows)")

        return filepath

    def download_and_save(
        self,
        symbol: str,
        timeframe: str,
        days: int = 90
    ) -> Optional[Path]:
        """
        Download e salva em um comando

        Args:
            symbol: Par (ex: "BTCUSDT")
            timeframe: Timeframe (ex: "1m")
            days: Dias para baixar (padrão: 90)

        Returns:
            Path do arquivo salvo
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = self.download_klines(symbol, timeframe, start_date, end_date)

        if df.empty:
            return None

        filepath = self.save_to_parquet(df, symbol, timeframe)
        return filepath

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos básicos (RSI, MACD, ATR)

        Args:
            df: DataFrame com OHLCV

        Returns:
            DataFrame com indicadores
        """
        print("\n📊 Calculando indicadores técnicos...")

        df = df.copy()

        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)

        print("   ✅ Indicadores adicionados: RSI, MACD, ATR, BB")

        return df


def main():
    parser = argparse.ArgumentParser(description='Download Binance Historical Data')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol (ex: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1m', help='Timeframe(s) (ex: 1m ou 1m,5m,15m)')
    parser.add_argument('--days', type=int, default=90, help='Dias para baixar (padrão: 90)')
    parser.add_argument('--output-dir', type=str, default='./data', help='Diretório output')
    parser.add_argument('--with-indicators', action='store_true', help='Adiciona indicadores técnicos')

    args = parser.parse_args()

    print("="*80)
    print("📥 BINANCE DATA DOWNLOADER")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe(s): {args.timeframe}")
    print(f"Days: {args.days}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Inicializa downloader
    downloader = BinanceDataDownloader(output_dir=args.output_dir)

    # Processa múltiplos timeframes se fornecido
    timeframes = [tf.strip() for tf in args.timeframe.split(',')]

    for tf in timeframes:
        try:
            filepath = downloader.download_and_save(
                symbol=args.symbol,
                timeframe=tf,
                days=args.days
            )

            if filepath and args.with_indicators:
                # Recarrega, adiciona indicadores, salva novamente
                df = pd.read_parquet(filepath)
                df = downloader.calculate_indicators(df)
                df.to_parquet(filepath, engine='pyarrow', compression='snappy', index=False)
                print(f"   ✅ Indicadores salvos em {filepath}")

        except Exception as e:
            print(f"\n❌ Erro ao baixar {tf}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("✅ DOWNLOAD CONCLUÍDO!")
    print("="*80)
    print(f"\nArquivos salvos em: {Path(args.output_dir).absolute()}")
    print("\nPróximo passo:")
    print(
        f"  python3 -m core.selectors.selector21 --symbol {args.symbol} --data_dir {args.output_dir} \\"
    )
    print(f"      --run_ml --ml_save_dir ./ml_models --walkforward")
    print("="*80)


if __name__ == "__main__":
    main()
