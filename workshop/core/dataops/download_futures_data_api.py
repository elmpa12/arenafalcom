#!/usr/bin/env python3
"""
Download Futures Data via Binance API
Baixa dados de futuros que NÃO estão disponíveis no Binance Vision.

Dados disponíveis via API:
- Funding Rate (histórico)
- Open Interest (histórico)
- Long/Short Ratio
- Taker Buy/Sell Volume

IMPORTANTE: API tem rate limits! Use com moderação.
"""

import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import time

BASE_URL = "https://fapi.binance.com"

class BinanceFuturesAPI:
    """Cliente para API de Futures da Binance"""

    def __init__(self):
        self.session = requests.Session()
        self.rate_limit_delay = 0.2  # 200ms entre requests

    def get_funding_rate(self, symbol: str, start_time: int, end_time: int, limit: int = 1000):
        """
        Get funding rate history.

        API: GET /fapi/v1/fundingRate
        Rate: Weight 1

        Returns:
            List of dict with: symbol, fundingTime, fundingRate, markPrice
        """
        url = f"{BASE_URL}/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }

        time.sleep(self.rate_limit_delay)

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] Funding rate: {e}")
            return []

    def get_open_interest_hist(self, symbol: str, period: str, start_time: int, end_time: int, limit: int = 500):
        """
        Get open interest history.

        API: GET /futures/data/openInterestHist
        Rate: Weight 1

        Args:
            period: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"

        Returns:
            List of dict with: symbol, sumOpenInterest, sumOpenInterestValue, timestamp
        """
        url = f"{BASE_URL}/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }

        time.sleep(self.rate_limit_delay)

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] Open interest: {e}")
            return []

    def get_long_short_ratio(self, symbol: str, period: str, start_time: int, end_time: int, limit: int = 500):
        """
        Get top trader long/short ratio.

        API: GET /futures/data/topLongShortAccountRatio
        Rate: Weight 1

        Args:
            period: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"

        Returns:
            List of dict with: symbol, longShortRatio, longAccount, shortAccount, timestamp
        """
        url = f"{BASE_URL}/futures/data/topLongShortAccountRatio"
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }

        time.sleep(self.rate_limit_delay)

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] Long/Short ratio: {e}")
            return []


def download_funding_rate(symbol: str, start_date: datetime, end_date: datetime, output_dir: Path):
    """
    Download funding rate history (happens every 8 hours).

    Binance funding times: 00:00, 08:00, 16:00 UTC
    """
    print(f"[FUNDING RATE] {symbol} from {start_date.date()} to {end_date.date()}")

    api = BinanceFuturesAPI()
    all_data = []

    # API limit: 1000 records por request
    # Funding rate acontece a cada 8h = 3 por dia
    # 1000 registros = ~333 dias

    current_start = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    total_days = (end_date - start_date).days
    pbar = tqdm(total=total_days, desc="Downloading")

    while current_start < end_ts:
        # Request 1000 registros (max)
        data = api.get_funding_rate(
            symbol=symbol,
            start_time=current_start,
            end_time=end_ts,
            limit=1000
        )

        if not data:
            print("[WARNING] No data returned, stopping")
            break

        all_data.extend(data)

        # Atualizar timestamp para próximo batch
        last_time = data[-1]['fundingTime']
        current_start = last_time + 1

        # Atualizar progress bar
        days_downloaded = (datetime.fromtimestamp(last_time/1000) - start_date).days
        pbar.update(max(0, days_downloaded - pbar.n))

        # Rate limit safety
        time.sleep(0.5)

    pbar.close()

    if not all_data:
        print("[ERROR] No funding rate data downloaded!")
        return

    # Converter para DataFrame
    df = pd.DataFrame(all_data)
    df['fundingRate'] = df['fundingRate'].astype(float)
    df['markPrice'] = df['markPrice'].astype(float) if 'markPrice' in df.columns else 0.0

    print(f"[TOTAL] {len(df):,} funding rate records")

    # Salvar
    output_path = output_dir / symbol
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / f"{symbol}_fundingRate_{start_date.date()}_{end_date.date()}.csv"
    df.to_csv(csv_file, index=False)
    print(f"[SAVED] {csv_file}")


def download_open_interest(symbol: str, start_date: datetime, end_date: datetime, output_dir: Path):
    """
    Download open interest history (5min intervals).
    """
    print(f"[OPEN INTEREST] {symbol} from {start_date.date()} to {end_date.date()}")

    api = BinanceFuturesAPI()
    all_data = []

    # API limit: 500 records por request
    # Period: 5m = 288 por dia
    # 500 registros = ~1.7 dias

    current_start = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    total_days = (end_date - start_date).days
    pbar = tqdm(total=total_days, desc="Downloading OI")

    while current_start < end_ts:
        data = api.get_open_interest_hist(
            symbol=symbol,
            period="5m",
            start_time=current_start,
            end_time=end_ts,
            limit=500
        )

        if not data:
            print("[WARNING] No data returned, stopping")
            break

        all_data.extend(data)

        # Atualizar timestamp
        last_time = data[-1]['timestamp']
        current_start = last_time + 1

        # Progress
        days_downloaded = (datetime.fromtimestamp(last_time/1000) - start_date).days
        pbar.update(max(0, days_downloaded - pbar.n))

        time.sleep(0.3)

    pbar.close()

    if not all_data:
        print("[ERROR] No open interest data!")
        return

    # DataFrame
    df = pd.DataFrame(all_data)
    df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
    df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)

    print(f"[TOTAL] {len(df):,} OI records")

    # Salvar em Parquet (muitos dados)
    output_path = output_dir / symbol
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"{symbol}_openInterest_{start_date.date()}_{end_date.date()}.parquet"
    df.to_parquet(parquet_file, compression='zstd', index=False)
    print(f"[SAVED] {parquet_file}")


def download_long_short_ratio(symbol: str, start_date: datetime, end_date: datetime, output_dir: Path):
    """
    Download top trader long/short ratio (5min intervals).
    """
    print(f"[LONG/SHORT RATIO] {symbol} from {start_date.date()} to {end_date.date()}")

    api = BinanceFuturesAPI()
    all_data = []

    current_start = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    total_days = (end_date - start_date).days
    pbar = tqdm(total=total_days, desc="Downloading L/S Ratio")

    while current_start < end_ts:
        data = api.get_long_short_ratio(
            symbol=symbol,
            period="5m",
            start_time=current_start,
            end_time=end_ts,
            limit=500
        )

        if not data:
            break

        all_data.extend(data)
        last_time = data[-1]['timestamp']
        current_start = last_time + 1

        days_downloaded = (datetime.fromtimestamp(last_time/1000) - start_date).days
        pbar.update(max(0, days_downloaded - pbar.n))

        time.sleep(0.3)

    pbar.close()

    if not all_data:
        print("[ERROR] No long/short ratio data!")
        return

    df = pd.DataFrame(all_data)
    df['longShortRatio'] = df['longShortRatio'].astype(float)
    df['longAccount'] = df['longAccount'].astype(float)
    df['shortAccount'] = df['shortAccount'].astype(float)

    print(f"[TOTAL] {len(df):,} L/S ratio records")

    output_path = output_dir / symbol
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / f"{symbol}_longShortRatio_{start_date.date()}_{end_date.date()}.parquet"
    df.to_parquet(parquet_file, compression='zstd', index=False)
    print(f"[SAVED] {parquet_file}")


def main():
    parser = argparse.ArgumentParser(description='Download Futures Data via Binance API')
    parser.add_argument('--data-type', required=True,
                        choices=['fundingRate', 'openInterest', 'longShortRatio', 'all'],
                        help='Tipo de dado')
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo')
    parser.add_argument('--start-date', required=True, help='Data inicial (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='Data final (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='./data', help='Diretório de saída')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    output_dir = Path(args.output_dir)

    print("="*80)
    print("DOWNLOAD FUTURES DATA VIA API")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {start_date.date()} → {end_date.date()}")
    print(f"Type: {args.data_type}")
    print("="*80)
    print()

    if args.data_type == 'fundingRate' or args.data_type == 'all':
        download_funding_rate(args.symbol, start_date, end_date, output_dir / 'fundingRate')
        print()

    if args.data_type == 'openInterest' or args.data_type == 'all':
        download_open_interest(args.symbol, start_date, end_date, output_dir / 'openInterest')
        print()

    if args.data_type == 'longShortRatio' or args.data_type == 'all':
        download_long_short_ratio(args.symbol, start_date, end_date, output_dir / 'longShortRatio')
        print()

    print("="*80)
    print("DOWNLOAD CONCLUÍDO!")
    print("="*80)


if __name__ == "__main__":
    main()
