#!/usr/bin/env python3
"""
Script para exportar resultados de backtest do Selector21 para formato visual replay.

Reconstrói frames a partir de:
- Klines BTCUSDT 15m (data/BTCUSDT/)
- Trades executados (best_trades.csv)
- Configuração do backtest (runtime_config.json)
- Métricas (leaderboard_base.csv)

Saída: visual/data/{backtest_id}/
  - frames.jsonl: Frame-by-frame replay data
  - trades.jsonl: Complete trade list
  - meta.json: Backtest metadata
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd


def load_klines(symbol: str, start: str, end: str, timeframe: str = "15m") -> pd.DataFrame:
    """Carrega klines do símbolo no período especificado."""
    data_dir = Path(f"/opt/botscalpv3/data/klines/{timeframe}") / symbol

    # Para 15m, cada arquivo tem 96 candles/dia (24h * 4)
    # Precisamos agregar de 1m para 15m se necessário
    print(f"🔍 Buscando dados: {symbol} {timeframe} de {start} até {end}")

    # Tenta carregar arquivo parquet agregado se existir
    parquet_path = data_dir / f"{symbol.lower()}_{timeframe}.parquet"
    if parquet_path.exists():
        print(f"✅ Carregando {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df = df[(df['open_time'] >= start) & (df['open_time'] < end)]
        return df

    # Senão, carrega CSVs individuais
    all_files = sorted(data_dir.glob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {data_dir}")

    print(f"📦 Carregando {len(all_files)} arquivos CSV...")
    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df = df[(df['open_time'] >= start) & (df['open_time'] < end)]
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Erro ao ler {f}: {e}")
            continue

    if not dfs:
        raise ValueError(f"Nenhum dado encontrado no período {start} - {end}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('open_time').reset_index(drop=True)

    # Agrega para timeframe desejado se necessário
    if timeframe != "1m":
        df = aggregate_to_timeframe(df, timeframe)

    print(f"✅ {len(df)} candles carregados")
    return df


def aggregate_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Agrega candles de 1m para timeframe maior."""
    # Mapeia timeframe para minutos
    tf_map = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1H", "4h": "4H", "1d": "1D"}
    freq = tf_map.get(timeframe)
    if not freq:
        raise ValueError(f"Timeframe {timeframe} não suportado")

    print(f"🔄 Agregando de 1m para {timeframe}...")
    df = df.set_index('open_time')

    agg_df = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna().reset_index()

    return agg_df


def load_trades(result_dir: Path) -> List[Dict[str, Any]]:
    """Carrega trades executados do CSV."""
    trades_path = result_dir / "best_trades.csv"
    if not trades_path.exists():
        print(f"⚠️ Arquivo {trades_path} não encontrado")
        return []

    trades = []
    with open(trades_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                'entry_time': row['entry_time'],
                'exit_time': row['exit_time'],
                'entry_price': float(row['entry_price']),
                'exit_price': float(row['exit_price']),
                'side': row['side'],
                'bars_held': int(row['bars_held']),
            })

    print(f"✅ {len(trades)} trades carregados")
    return trades


def load_config(result_dir: Path) -> Dict[str, Any]:
    """Carrega runtime config."""
    config_path = result_dir / "runtime_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} não encontrado")

    with open(config_path, 'r') as f:
        return json.load(f)


def load_leaderboard(result_dir: Path) -> Dict[str, Any]:
    """Carrega métricas da leaderboard."""
    lb_path = result_dir / "leaderboard_base.csv"
    if not lb_path.exists():
        raise FileNotFoundError(f"{lb_path} não encontrado")

    with open(lb_path, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)  # Apenas primeira linha
        return {
            'method': row['method'],
            'timeframe': row['timeframe'],
            'expectancy': float(row['expectancy']),
            'hit': float(row['hit']),
            'sharpe': float(row['sharpe']),
            'n_trades': int(row['n_trades']),
            'total_pnl': float(row['total_pnl']),
            'maxdd': float(row['maxdd']),
            'payoff': float(row['payoff']),
        }


def generate_frames(klines: pd.DataFrame, trades: List[Dict], config: Dict) -> List[Dict[str, Any]]:
    """Gera frames para replay a partir de klines e trades."""
    print("🎬 Gerando frames...")

    # Indexa trades por tempo de entrada e saída (convertendo para formato ISO)
    from datetime import datetime
    trades_by_entry = {datetime.fromisoformat(t['entry_time']).isoformat(): t for t in trades}
    trades_by_exit = {datetime.fromisoformat(t['exit_time']).isoformat(): t for t in trades}

    # Estado de trading
    equity = 10000.0  # Capital inicial (placeholder)
    open_trades = []
    frames = []

    for idx, row in klines.iterrows():
        timestamp = row['open_time'].isoformat()

        # Frame base com OHLCV
        frame = {
            'bar': {
                't': timestamp,
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
                'v': float(row['volume']),
            },
            'signals': [],
            'trades_open': [],
            'trades_closed': [],
            'equity': equity,
        }

        # Verifica se há nova entrada neste timestamp
        if timestamp in trades_by_entry:
            trade = trades_by_entry[timestamp]
            open_trade = {
                'entry_t': trade['entry_time'],
                'entry_px': trade['entry_price'],
                'side': trade['side'],
                'exit_t': None,
                'exit_px': None,
                'pnl': None,
                'pipeline': config['base_methods'][config['timeframes'][0]][0]['name'],
            }
            open_trades.append(open_trade)

            # Sinal de entrada
            frame['signals'].append({
                't': timestamp,
                'type': 'LONG' if trade['side'] == 'LONG' else 'SHORT',
                'px': trade['entry_price'],
                'label': f"Entry {trade['side']}",
            })

        # Verifica se há saída neste timestamp
        if timestamp in trades_by_exit:
            trade = trades_by_exit[timestamp]
            # Encontra trade aberto correspondente
            for ot in open_trades[:]:
                if ot['entry_t'] == trade['entry_time']:
                    # Calcula PnL (simplificado, sem fees/slippage)
                    if trade['side'] == 'LONG':
                        pnl = (trade['exit_price'] - trade['entry_price']) * config['execution']['contracts'] * config['execution']['contract_value']
                    else:
                        pnl = (trade['entry_price'] - trade['exit_price']) * config['execution']['contracts'] * config['execution']['contract_value']

                    equity += pnl

                    closed_trade = {
                        'entry_t': ot['entry_t'],
                        'entry_px': ot['entry_px'],
                        'side': ot['side'],
                        'exit_t': timestamp,
                        'exit_px': trade['exit_price'],
                        'pnl': pnl,
                        'pipeline': ot['pipeline'],
                    }
                    frame['trades_closed'].append(closed_trade)
                    open_trades.remove(ot)

                    # Sinal de saída
                    frame['signals'].append({
                        't': timestamp,
                        'type': 'EXIT',
                        'px': trade['exit_price'],
                        'label': f"Exit ({'+' if pnl > 0 else ''}{pnl:.2f})",
                    })
                    break

        # Trades ainda abertos
        frame['trades_open'] = [dict(t) for t in open_trades]
        frame['equity'] = equity

        frames.append(frame)

    print(f"✅ {len(frames)} frames gerados")
    return frames


def export_to_visual(backtest_id: str, result_dir: Path, output_dir: Path):
    """Exporta backtest completo para formato visual."""
    print(f"\n{'='*60}")
    print(f"🎯 Exportando: {backtest_id}")
    print(f"📂 Origem: {result_dir}")
    print(f"📂 Destino: {output_dir}")
    print(f"{'='*60}\n")

    # 1. Carrega configuração
    print("📋 [1/5] Carregando configuração...")
    config = load_config(result_dir)
    leaderboard = load_leaderboard(result_dir)

    symbol = config['meta']['symbol']
    start = config['meta']['range_used']['start']
    end = config['meta']['range_used']['end']
    timeframe = config['timeframes'][0]

    # 2. Carrega klines
    print(f"\n📊 [2/5] Carregando klines...")
    klines = load_klines(symbol, start, end, timeframe)

    # 3. Carrega trades
    print(f"\n💼 [3/5] Carregando trades...")
    trades = load_trades(result_dir)

    # 4. Gera frames
    print(f"\n🎬 [4/5] Gerando frames...")
    frames = generate_frames(klines, trades, config)

    # 5. Exporta arquivos
    print(f"\n💾 [5/5] Exportando arquivos...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # frames.jsonl
    frames_path = output_dir / "frames.jsonl"
    with open(frames_path, 'w') as f:
        for frame in frames:
            f.write(json.dumps(frame) + '\n')
    print(f"  ✅ {frames_path.name} ({len(frames)} frames)")

    # trades.jsonl - Convert format to match backend expectations
    trades_path = output_dir / "trades.jsonl"
    with open(trades_path, 'w') as f:
        for trade in trades:
            # Convert field names: entry_time -> entry_t, entry_price -> entry_px, etc.
            converted_trade = {
                'entry_t': trade['entry_time'],
                'exit_t': trade['exit_time'],
                'entry_px': trade['entry_price'],
                'exit_px': trade['exit_price'],
                'side': trade['side'],
                'size': None,  # Not available in CSV
                'pnl': None,   # Will be calculated by frontend
                'pipeline': None,
            }
            f.write(json.dumps(converted_trade) + '\n')
    print(f"  ✅ {trades_path.name} ({len(trades)} trades)")

    # meta.json
    meta = {
        'id': backtest_id,
        'symbol': symbol,
        'timeframe': timeframe,
        'n_frames': len(frames),
        'n_trades': len(trades),
        'period': {
            'start': start,
            'end': end,
        },
        'pipelines': [
            {
                'id': config['base_methods'][timeframe][0]['name'],
                'type': 'base',
                'edge': leaderboard['expectancy'] / leaderboard['n_trades'],  # edge por trade
                'kpis': {
                    'winrate': leaderboard['hit'],
                    'expectancy': leaderboard['expectancy'],
                    'sharpe': leaderboard['sharpe'],
                    'total_pnl': leaderboard['total_pnl'],
                    'maxdd': leaderboard['maxdd'],
                    'payoff': leaderboard['payoff'],
                },
            }
        ],
        'kpis': {
            'winrate': leaderboard['hit'],
            'expectancy': leaderboard['expectancy'],
            'sharpe': leaderboard['sharpe'],
            'total_pnl': leaderboard['total_pnl'],
            'maxdd': leaderboard['maxdd'],
            'payoff': leaderboard['payoff'],
        },
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ {meta_path.name}")

    print(f"\n{'='*60}")
    print(f"✅ EXPORTAÇÃO COMPLETA!")
    print(f"{'='*60}\n")
    print(f"📊 Estatísticas:")
    print(f"  • Frames: {len(frames)}")
    print(f"  • Trades: {len(trades)}")
    print(f"  • Win rate: {leaderboard['hit']*100:.1f}%")
    print(f"  • Total PnL: ${leaderboard['total_pnl']:,.2f}")
    print(f"  • Sharpe: {leaderboard['sharpe']:.2f}")
    print(f"\n🌐 Para visualizar:")
    print(f"  cd /opt/botscalpv3/visual")
    print(f"  python3 -m backend.app")
    print(f"  # Acesse http://localhost:8000")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 export_to_visual.py <backtest_id> [result_dir] [output_dir]")
        print("\nExemplo:")
        print("  python3 export_to_visual.py gen3_15m_macd")
        print("  python3 export_to_visual.py gen3_15m_macd ./resultados/gen3/gen3_15m_macd ./visual/data/gen3_15m_macd")
        sys.exit(1)

    backtest_id = sys.argv[1]
    result_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"./resultados/gen3/{backtest_id}")
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"./visual/data/{backtest_id}")

    if not result_dir.exists():
        print(f"❌ Diretório {result_dir} não encontrado")
        sys.exit(1)

    export_to_visual(backtest_id, result_dir, output_dir)
