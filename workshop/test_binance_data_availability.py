#!/usr/bin/env python3
"""
Test Binance Data Availability
Verifica se os dados estão disponíveis no Binance Vision antes de baixar.
"""

import requests
from datetime import datetime, timedelta

BASE_URL = "https://data.binance.vision/data"
MARKET = "futures/um"

# Testa 1 dia recente (ontem)
TEST_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

DATA_TYPES = {
    "aggTrades": f"{BASE_URL}/{MARKET}/daily/aggTrades/{{symbol}}/{{symbol}}-aggTrades-{TEST_DATE}.zip",
    "klines_1m": f"{BASE_URL}/{MARKET}/daily/klines/{{symbol}}/1m/{{symbol}}-1m-{TEST_DATE}.zip",
    "klines_1h": f"{BASE_URL}/{MARKET}/daily/klines/{{symbol}}/1h/{{symbol}}-1h-{TEST_DATE}.zip",
    "klines_4h": f"{BASE_URL}/{MARKET}/daily/klines/{{symbol}}/4h/{{symbol}}-4h-{TEST_DATE}.zip",
    "klines_1d": f"{BASE_URL}/{MARKET}/daily/klines/{{symbol}}/1d/{{symbol}}-1d-{TEST_DATE}.zip",
    "fundingRate": f"{BASE_URL}/{MARKET}/daily/fundingRate/{{symbol}}/{{symbol}}-fundingRate-{TEST_DATE}.zip",
    "openInterest": f"{BASE_URL}/{MARKET}/daily/openInterest/{{symbol}}/{{symbol}}-openInterest-{TEST_DATE}.zip",
    "liquidationSnapshot": f"{BASE_URL}/{MARKET}/daily/liquidationSnapshot/{{symbol}}/{{symbol}}-liquidationSnapshot-{TEST_DATE}.zip",
}

def test_url(url):
    """Testa se URL está acessível (HEAD request)"""
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 80)
    print(f"TESTANDO DISPONIBILIDADE DOS DADOS - {TEST_DATE}")
    print("=" * 80)
    print()

    results = {}

    for symbol in SYMBOLS:
        print(f"\n{'='*80}")
        print(f"Testando {symbol}")
        print(f"{'='*80}")

        results[symbol] = {}

        for data_type, url_template in DATA_TYPES.items():
            url = url_template.format(symbol=symbol)
            available = test_url(url)
            results[symbol][data_type] = available

            status = "✅ DISPONÍVEL" if available else "❌ INDISPONÍVEL"
            print(f"  {data_type:20s} {status}")

    # Resumo
    print(f"\n{'='*80}")
    print("RESUMO")
    print(f"{'='*80}\n")

    for data_type in DATA_TYPES.keys():
        available_symbols = [s for s in SYMBOLS if results[s][data_type]]
        status = "✅" if len(available_symbols) == len(SYMBOLS) else "⚠️"
        print(f"{status} {data_type:25s} {len(available_symbols)}/{len(SYMBOLS)} símbolos")
        if len(available_symbols) < len(SYMBOLS):
            missing = [s for s in SYMBOLS if not results[s][data_type]]
            print(f"   Faltando: {', '.join(missing)}")

    print("\n" + "="*80)
    print("IMPORTANTE:")
    print("  - Se algum dado estiver indisponível, o download vai pular esses arquivos")
    print("  - Funding Rate e OI podem ter atraso de 1-2 dias")
    print("  - Liquidations podem não estar sempre disponíveis")
    print("  - Book depth geralmente NÃO está disponível no Binance Vision")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
