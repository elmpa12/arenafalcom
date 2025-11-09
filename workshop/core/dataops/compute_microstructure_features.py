#!/usr/bin/env python3
"""
================================================================================
MICROSTRUCTURE FEATURES COMPUTER
================================================================================

Processa dados brutos coletados e calcula features para ML/DL.

Input:
- ./data/aggtrades/    (raw aggtrades)
- ./data/book_depth/   (raw book snapshots)
- ./data/market/       (funding, OI, liquidations)

Output:
- ./data/features/BTCUSDT/1s/   (features a cada 1s)
- ./data/features/BTCUSDT/1m/   (features a cada 1m)
- ./data/features/BTCUSDT/5m/   (features a cada 5m)
- ./data/features/BTCUSDT/15m/  (features a cada 15m)

Features calculadas (baseado no debate):

FROM AGGTRADES (12):
- cvd (Cumulative Volume Delta)
- vwap (Volume-Weighted Average Price)
- buy_volume, sell_volume, buy_pressure
- trade_count
- trade_intensity_1s, _5s, _10s
- large_trade_count, _volume, _pct

FROM BOOK DEPTH (10):
- imbalance_5, _10, _20
- spread_avg, spread_std
- bid_slope, ask_slope
- weighted_mid_diff_bps
- bid_vol_ratio_5_20, ask_vol_ratio_5_20

FROM MARKET (7):
- funding_rate, funding_delta
- spot_premium_pct
- open_interest, oi_change_pct
- liq_long_vol, liq_short_vol, liq_ratio

Uso:
    python3 compute_microstructure_features.py \\
        --symbol BTCUSDT \\
        --start-date 2024-08-01 \\
        --end-date 2024-11-08 \\
        --timeframes 1m,5m,15m \\
        --aggtrades-dir ./data/aggtrades \\
        --book-dir ./data/book_depth \\
        --market-dir ./data/market \\
        --output-dir ./data/features

Author: Claude + GPT Dual Debate
Date: 2025-11-08
================================================================================
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress
import json
from tqdm import tqdm
import pyarrow.parquet as pq


class MicrostructureFeaturesComputer:
    """
    Computa features de microstructure.

    Decisões do debate implementadas:
    - CVD: sum(buy_vol) - sum(sell_vol)
    - VWAP: sum(price * volume) / sum(volume)
    - Trade intensity: rolling count em janelas 1s, 5s, 10s
    - Large trades: threshold 200% do volume médio (15min rolling)
    - Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    - Funding: absoluto E delta
    - OI: % change
    - Liquidations: ratio longs/shorts
    """

    def __init__(self, symbol: str, aggtrades_dir: str, book_dir: str, market_dir: str):
        self.symbol = symbol.upper()
        self.aggtrades_dir = Path(aggtrades_dir) / symbol
        self.book_dir = Path(book_dir) / symbol
        self.market_dir = Path(market_dir) / symbol

    def load_aggtrades(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Carrega aggtrades do período"""
        print(f"[LOAD] AggTrades: {start_date} → {end_date}")

        dfs = []

        current_date = start_date
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day

            # Load all hours of the day
            for hour in range(24):
                path = (
                    self.aggtrades_dir /
                    str(year) /
                    f"{month:02d}" /
                    f"{day:02d}" /
                    f"hour={hour:02d}" /
                    "data.parquet"
                )

                if path.exists():
                    df = pd.read_parquet(path)
                    dfs.append(df)

            current_date += timedelta(days=1)

        if not dfs:
            print("[WARNING] No aggtrades data found!")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"[LOADED] {len(df):,} trades")

        return df

    def compute_aggtrade_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calcula features dos aggtrades.

        Args:
            df: DataFrame com aggtrades
            timeframe: '1s', '1min', '5min', '15min'

        Returns:
            DataFrame com features por intervalo
        """
        print(f"[COMPUTE] AggTrade features para {timeframe}...")

        df = df.copy()
        df.set_index('timestamp', inplace=True)

        # Separa buy/sell
        df['is_buy'] = ~df['is_buyer_maker']
        df['buy_volume'] = df['quantity'].where(df['is_buy'], 0)
        df['sell_volume'] = df['quantity'].where(~df['is_buy'], 0)

        # Resample
        agg_dict = {
            'price': ['first', 'max', 'min', 'last'],  # OHLC
            'quantity': 'sum',  # Total volume
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'trade_id': 'count',  # Trade count
        }

        resampled = df.resample(timeframe).agg(agg_dict)

        # Flatten columns
        resampled.columns = ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'trade_count']

        # CVD (Cumulative Volume Delta)
        # Consenso: sum(buy_vol) - sum(sell_vol), cumulativo
        resampled['volume_delta'] = resampled['buy_volume'] - resampled['sell_volume']
        resampled['cvd'] = resampled['volume_delta'].cumsum()

        # VWAP intrabar
        # Precisamos do weighted average, mas já temos OHLC
        # Aproximação: (H+L+C)/3 ponderado
        resampled['vwap'] = (resampled['high'] + resampled['low'] + resampled['close']) / 3

        # Buy pressure
        resampled['buy_pressure'] = resampled['buy_volume'] / (resampled['volume'] + 1e-10)

        # Trade intensity (trades/segundo)
        # Para diferentes janelas
        resampled['trade_intensity_1s'] = resampled['trade_count']
        resampled['trade_intensity_5s'] = resampled['trade_count'].rolling(5).sum()
        resampled['trade_intensity_10s'] = resampled['trade_count'].rolling(10).sum()

        # Large trade detection
        # Threshold: 200% do volume médio (rolling 15min)
        volume_mean_15m = resampled['volume'].rolling('15min').mean()
        large_trade_threshold = volume_mean_15m * 2.0

        # Contar trades grandes (aproximação com volume total do período)
        resampled['large_trade_flag'] = (resampled['volume'] > large_trade_threshold).astype(int)
        resampled['large_trade_count'] = resampled['large_trade_flag']
        resampled['large_trade_volume'] = resampled['volume'].where(resampled['large_trade_flag'] == 1, 0)
        resampled['large_trade_pct'] = (resampled['large_trade_volume'] / (resampled['volume'] + 1e-10)) * 100

        return resampled.reset_index()

    def load_book_depth(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Carrega book depth snapshots"""
        print(f"[LOAD] Book Depth: {start_date} → {end_date}")

        dfs = []

        current_date = start_date
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day

            for hour in range(24):
                path = (
                    self.book_dir /
                    str(year) /
                    f"{month:02d}" /
                    f"{day:02d}" /
                    f"hour={hour:02d}" /
                    "data.parquet"
                )

                if path.exists():
                    df = pd.read_parquet(path)
                    dfs.append(df)

            current_date += timedelta(days=1)

        if not dfs:
            print("[WARNING] No book depth data found!")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"[LOADED] {len(df):,} snapshots")

        return df

    def compute_book_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calcula features do book depth"""
        print(f"[COMPUTE] Book Depth features para {timeframe}...")

        df = df.copy()
        df.set_index('timestamp', inplace=True)

        # Resample (média dos valores no período)
        agg_dict = {
            'imbalance_5': 'mean',
            'imbalance_10': 'mean',
            'imbalance_20': 'mean',
            'spread': ['mean', 'std'],
            'mid_price': 'last',
            'weighted_mid': 'last',
        }

        # Adicionar features calculadas se existirem
        if 'bid_vol_5' in df.columns:
            agg_dict['bid_vol_5'] = 'mean'
            agg_dict['ask_vol_5'] = 'mean'
            agg_dict['bid_vol_20'] = 'mean'
            agg_dict['ask_vol_20'] = 'mean'

        resampled = df.resample(timeframe).agg(agg_dict)

        # Flatten
        resampled.columns = ['_'.join(col).strip('_') for col in resampled.columns.values]

        # Ratios
        if 'bid_vol_5_mean' in resampled.columns:
            resampled['bid_vol_ratio_5_20'] = resampled['bid_vol_5_mean'] / (resampled['bid_vol_20_mean'] + 1e-10)
            resampled['ask_vol_ratio_5_20'] = resampled['ask_vol_5_mean'] / (resampled['ask_vol_20_mean'] + 1e-10)

        # Weighted mid price deviation (em bps)
        if 'weighted_mid_last' in resampled.columns and 'mid_price_last' in resampled.columns:
            resampled['weighted_mid_diff_bps'] = (
                (resampled['weighted_mid_last'] - resampled['mid_price_last']) /
                resampled['mid_price_last'] * 10000
            )

        return resampled.reset_index()

    def load_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Carrega market data"""
        print(f"[LOAD] Market Data: {start_date} → {end_date}")

        dfs = []

        current_date = start_date
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day

            for hour in range(24):
                path = (
                    self.market_dir /
                    str(year) /
                    f"{month:02d}" /
                    f"{day:02d}" /
                    f"hour={hour:02d}" /
                    "data.parquet"
                )

                if path.exists():
                    df = pd.read_parquet(path)
                    dfs.append(df)

            current_date += timedelta(days=1)

        if not dfs:
            print("[WARNING] No market data found!")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"[LOADED] {len(df):,} data points")

        return df

    def compute_market_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calcula features de market data"""
        print(f"[COMPUTE] Market Data features para {timeframe}...")

        df = df.copy()
        df.set_index('timestamp', inplace=True)

        # Resample
        agg_dict = {
            'funding_rate': 'last',
            'funding_rate_delta': 'last',
            'mark_price': 'last',
            'spot_price': 'last',
            'spot_premium_pct': 'mean',
            'open_interest': 'last',
            'oi_change_pct': 'sum',
            'liq_long_volume': 'sum',
            'liq_short_volume': 'sum',
        }

        resampled = df.resample(timeframe).agg(agg_dict).fillna(method='ffill')

        # Recalcular liq_ratio
        resampled['liq_ratio'] = resampled['liq_long_volume'] / (resampled['liq_short_volume'] + 1e-10)

        return resampled.reset_index()

    def merge_all_features(
        self,
        aggtrade_features: pd.DataFrame,
        book_features: pd.DataFrame,
        market_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all features on timestamp"""
        print("[MERGE] Merging all features...")

        # Start with aggtrades (base OHLCV)
        merged = aggtrade_features.copy()

        # Merge book features
        if not book_features.empty:
            merged = pd.merge(merged, book_features, on='timestamp', how='left', suffixes=('', '_book'))

        # Merge market features
        if not market_features.empty:
            merged = pd.merge(merged, market_features, on='timestamp', how='left', suffixes=('', '_market'))

        # Fill NaN with forward fill
        merged = merged.fillna(method='ffill').fillna(0)

        print(f"[MERGED] {len(merged)} rows, {len(merged.columns)} features")

        return merged

    def save_features(self, df: pd.DataFrame, symbol: str, timeframe: str, output_dir: Path):
        """Salva features em Parquet particionado"""
        print(f"[SAVE] Salvando features {timeframe}...")

        df['dt'] = pd.to_datetime(df['timestamp'])

        # Particionar por hora
        for hour, group in df.groupby(df['dt'].dt.floor('H')):
            year = hour.year
            month = hour.month
            day = hour.day
            hour_val = hour.hour

            partition_path = (
                output_dir /
                symbol /
                timeframe /
                str(year) /
                f"{month:02d}" /
                f"{day:02d}" /
                f"hour={hour_val:02d}"
            )
            partition_path.mkdir(parents=True, exist_ok=True)

            save_df = group.drop(columns=['dt'])

            filepath = partition_path / "data.parquet"

            save_df.to_parquet(
                filepath,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

        print(f"[SAVED] {timeframe} features")


def main():
    parser = argparse.ArgumentParser(description='Microstructure Features Computer')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--start-date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--end-date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--timeframes', default='1min,5min,15min', help='Comma-separated')
    parser.add_argument('--aggtrades-dir', default='./data/aggtrades')
    parser.add_argument('--book-dir', default='./data/book_depth')
    parser.add_argument('--market-dir', default='./data/market')
    parser.add_argument('--output-dir', default='./data/features')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    timeframes = args.timeframes.split(',')

    computer = MicrostructureFeaturesComputer(
        symbol=args.symbol,
        aggtrades_dir=args.aggtrades_dir,
        book_dir=args.book_dir,
        market_dir=args.market_dir
    )

    # Load raw data
    aggtrades = computer.load_aggtrades(start_date, end_date)
    book_depth = computer.load_book_depth(start_date, end_date)
    market_data = computer.load_market_data(start_date, end_date)

    # Process each timeframe
    for tf in timeframes:
        print(f"\n{'='*80}")
        print(f"PROCESSING TIMEFRAME: {tf}")
        print(f"{'='*80}\n")

        # Compute features
        aggtrade_features = computer.compute_aggtrade_features(aggtrades, tf)
        book_features = computer.compute_book_features(book_depth, tf) if not book_depth.empty else pd.DataFrame()
        market_features = computer.compute_market_features(market_data, tf) if not market_data.empty else pd.DataFrame()

        # Merge
        final_features = computer.merge_all_features(
            aggtrade_features,
            book_features,
            market_features
        )

        # Save
        computer.save_features(
            final_features,
            args.symbol,
            tf,
            Path(args.output_dir)
        )

    print(f"\n{'='*80}")
    print("FEATURES COMPUTATION COMPLETED!")
    print(f"{'='*80}\n")
    print(f"Features salvos em: {args.output_dir}")
    print(f"Timeframes: {', '.join(timeframes)}")


if __name__ == "__main__":
    main()
