#!/usr/bin/env python3
"""
Download Binance Public Data - TURBO MODE
Download paralelo de dados históricos da Binance Vision (10-20x mais rápido!)

Features:
- Downloads paralelos (10-20 workers)
- Progress bar em tempo real
- Auto-conversão para Parquet otimizado
- Retry automático
- Resume capability
"""

import os
import sys
import argparse
import requests
import zipfile
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Dtypes otimizados (do script original)
KLINES_DTYPE = {
    'open_time': 'int64',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    'close_time': 'int64',
    'quote_volume': 'float64',
    'trades': 'int32',
    'taker_buy_base': 'float64',
    'taker_buy_quote': 'float64',
    'ignore': 'float64',
}

KLINES_NAMES = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
]

AGGTRADES_DTYPE = {
    'trade_id': 'int64',
    'price': 'float64',
    'quantity': 'float64',
    'first_trade_id': 'int64',
    'last_trade_id': 'int64',
    'timestamp': 'int64',
    'is_buyer_maker': 'bool'
}

AGGTRADES_NAMES = [
    'trade_id', 'price', 'quantity',
    'first_trade_id', 'last_trade_id',
    'timestamp', 'is_buyer_maker'
]


def generate_date_range(start: str, end: str) -> List[str]:
    """Gera lista de datas entre start e end"""
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


def build_url(market: str, data_type: str, symbol: str, date: str, interval: Optional[str] = None) -> str:
    """Constrói URL de download da Binance Vision"""
    base = "https://data.binance.vision/data"

    if market == "spot":
        market_path = "spot"
    else:
        market_path = "futures/um"

    if data_type == "klines":
        return f"{base}/{market_path}/daily/{data_type}/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
    elif data_type == "aggTrades":
        return f"{base}/{market_path}/daily/{data_type}/{symbol}/{symbol}-{data_type}-{date}.zip"
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")


def download_and_convert(
    url: str,
    output_file: Path,
    data_type: str,
    max_retries: int = 3
) -> Tuple[bool, str]:
    """
    Baixa arquivo ZIP, extrai CSV, converte para Parquet

    Returns:
        (success: bool, message: str)
    """
    # Verificar se já existe
    if output_file.exists():
        return True, f"Skip (exists): {output_file.name}"

    # Criar diretório se necessário
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Download com retry
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 404:
                return False, f"Not found: {url.split('/')[-1]}"

            response.raise_for_status()

            # Extrair ZIP em memória
            zip_path = output_file.with_suffix('.zip')
            zip_path.write_bytes(response.content)

            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_name = z.namelist()[0]
                csv_data = z.read(csv_name)

            # Remover ZIP
            zip_path.unlink()

            # Detectar se CSV tem header (primeira linha começa com letra ao invés de número)
            first_line = csv_data.decode('utf-8').split('\n')[0]
            has_header = not first_line[0].isdigit()

            # Converter para Parquet
            if data_type == "klines":
                df = pd.read_csv(
                    pd.io.common.BytesIO(csv_data),
                    skiprows=1 if has_header else 0,
                    header=None,
                    names=KLINES_NAMES,
                    dtype=KLINES_DTYPE
                )
            elif data_type == "aggTrades":
                df = pd.read_csv(
                    pd.io.common.BytesIO(csv_data),
                    skiprows=1 if has_header else 0,
                    header=None,
                    names=AGGTRADES_NAMES,
                    dtype=AGGTRADES_DTYPE
                )

            # Salvar como Parquet com Zstd compression
            df.to_parquet(
                output_file,
                engine='pyarrow',
                compression='zstd',
                compression_level=3,
                index=False
            )

            return True, f"✓ {output_file.name}"

        except Exception as e:
            if attempt < max_retries:
                continue
            else:
                return False, f"✗ {output_file.name}: {str(e)[:50]}"

    return False, f"✗ Failed after {max_retries} retries"


def download_parallel(
    tasks: List[Tuple[str, Path, str]],
    max_workers: int = 15,
    description: str = "Downloading"
) -> Tuple[int, int]:
    """
    Download paralelo com progress bar

    Args:
        tasks: Lista de (url, output_file, data_type)
        max_workers: Número de workers paralelos
        description: Descrição para progress bar

    Returns:
        (success_count, failed_count)
    """
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit todas as tarefas
        futures = {
            executor.submit(download_and_convert, url, output, dtype): (url, output)
            for url, output, dtype in tasks
        }

        # Progress bar
        with tqdm(total=len(tasks), desc=description, unit="file") as pbar:
            for future in as_completed(futures):
                success, message = future.result()

                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    if "Skip" not in message:
                        logger.warning(message)

                pbar.update(1)
                pbar.set_postfix({"✓": success_count, "✗": failed_count})

    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance Public Data - TURBO MODE (10-20x faster!)"
    )

    parser.add_argument("--market", default="futures", choices=["spot", "futures"])
    parser.add_argument("--data-type", required=True, choices=["klines", "aggTrades"])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--intervals", help="Intervalos separados por vírgula (ex: 1m,5m,15m)")
    parser.add_argument("--start-date", required=True, help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="Data final (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="./data", help="Diretório de saída")
    parser.add_argument("--workers", type=int, default=15, help="Número de workers paralelos (default: 15)")

    args = parser.parse_args()

    # Banner
    print("=" * 70)
    print("🚀 BINANCE DATA DOWNLOADER - TURBO MODE")
    print("=" * 70)
    print(f"Market: {args.market}")
    print(f"Data Type: {args.data_type}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start_date} → {args.end_date}")
    print(f"Parallel Workers: {args.workers}")
    print("=" * 70)
    print()

    # Gerar datas
    dates = generate_date_range(args.start_date, args.end_date)
    logger.info(f"Total de {len(dates)} dias para baixar")

    output_dir = Path(args.output_dir)

    # Preparar tarefas
    if args.data_type == "klines":
        if not args.intervals:
            logger.error("--intervals é obrigatório para klines")
            sys.exit(1)

        intervals = args.intervals.split(",")

        for interval in intervals:
            logger.info(f"Preparando download de klines {interval}...")

            tasks = []
            for date in dates:
                url = build_url(args.market, "klines", args.symbol, date, interval)
                output_file = output_dir / "klines" / interval / args.symbol / f"{args.symbol}_{interval}_{date}.parquet"
                tasks.append((url, output_file, "klines"))

            logger.info(f"Baixando {len(tasks)} arquivos de klines {interval}...")
            success, failed = download_parallel(
                tasks,
                max_workers=args.workers,
                description=f"Klines {interval}"
            )

            logger.info(f"Klines {interval}: {success} sucessos, {failed} falhas")
            print()

    elif args.data_type == "aggTrades":
        logger.info("Preparando download de aggTrades...")

        tasks = []
        for date in dates:
            url = build_url(args.market, "aggTrades", args.symbol, date)
            output_file = output_dir / "aggTrades" / args.symbol / f"{args.symbol}_aggTrades_{date}.parquet"
            tasks.append((url, output_file, "aggTrades"))

        logger.info(f"Baixando {len(tasks)} arquivos de aggTrades...")
        success, failed = download_parallel(
            tasks,
            max_workers=args.workers,
            description="AggTrades"
        )

        logger.info(f"AggTrades: {success} sucessos, {failed} falhas")
        print()

    print("=" * 70)
    print("✅ DOWNLOAD COMPLETO!")
    print("=" * 70)
    print(f"Dados salvos em: {output_dir}")
    print()


if __name__ == "__main__":
    main()
