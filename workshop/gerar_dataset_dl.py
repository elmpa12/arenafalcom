#!/usr/bin/env python3
"""
Gerar Dataset para Deep Learning
Extrai features e labels dos 4 setups validados para treinar LSTM/Transformer
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🧠 Gerando Dataset para Deep Learning\n")

# Configuração
SETUPS_VALIDADOS = [
    {"method": "ema_crossover", "tf": "15m", "name": "EMA Cross 15m"},
    {"method": "ema_crossover", "tf": "5m", "name": "EMA Cross 5m"},
    {"method": "macd_trend", "tf": "15m", "name": "MACD Trend 15m"},
    {"method": "keltner_breakout", "tf": "15m", "name": "Keltner Breakout 15m"}
]

def load_ohlcv_data(data_root, symbol, timeframe, start_date, end_date):
    """Carrega dados OHLCV do data_monthly"""
    data_path = Path(data_root) / f"{symbol}_{timeframe}.parquet"

    if not data_path.exists():
        print(f"  ⚠️  Arquivo não encontrado: {data_path}")
        return None

    df = pd.read_parquet(data_path)

    # Filtrar datas
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    return df

def calculate_technical_indicators(df):
    """Calcula indicadores técnicos"""
    features = df.copy()

    # EMAs
    for period in [9, 21, 50, 200]:
        features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    features['bb_upper'] = sma_20 + (std_20 * 2)
    features['bb_lower'] = sma_20 - (std_20 * 2)
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    features['atr'] = true_range.rolling(14).mean()

    # Volume features
    features['volume_sma'] = df['volume'].rolling(window=20).mean()
    features['volume_ratio'] = df['volume'] / features['volume_sma']

    # Price momentum
    for period in [5, 10, 20]:
        features[f'momentum_{period}'] = df['close'].pct_change(period)

    # Volatility
    features['volatility_20'] = df['close'].rolling(window=20).std()

    # Distance from EMAs
    for period in [9, 21, 50]:
        features[f'dist_ema_{period}'] = (df['close'] - features[f'ema_{period}']) / features[f'ema_{period}']

    return features

def generate_labels_from_future_returns(df, horizon=10):
    """
    Gera labels baseado em retornos futuros
    Label = 1 se retorno futuro > 0.1%, 0 caso contrário
    """
    # Calcular retorno futuro
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1

    # Label: 1 se lucro > 0.1%, 0 caso contrário (threshold conservador)
    df['label'] = (df['future_return'] > 0.001).astype(int)

    return df

def create_sequences(df, lookback=128):
    """
    Cria sequências de lookback barras para LSTM
    Input: DataFrame com features
    Output: Arrays de sequências X e labels y
    """
    feature_cols = [col for col in df.columns if col not in
                   ['timestamp', 'label', 'future_return', 'open', 'high', 'low', 'close', 'volume']]

    X_sequences = []
    y_labels = []
    timestamps = []

    for i in range(lookback, len(df)):
        # Pegar lookback barras anteriores
        sequence = df.iloc[i-lookback:i][feature_cols].values

        # Verificar se tem NaN
        if np.isnan(sequence).any():
            continue

        # Label da barra atual
        label = df.iloc[i]['label']
        if pd.isna(label):
            continue

        X_sequences.append(sequence)
        y_labels.append(label)
        timestamps.append(df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i)

    return np.array(X_sequences), np.array(y_labels), timestamps

def main():
    parser = argparse.ArgumentParser(description='Gerar dataset DL dos 4 setups validados')
    parser.add_argument('--data_root', default='./data_monthly', help='Diretório com dados parquet')
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo')
    parser.add_argument('--start', default='2022-01-01', help='Data início')
    parser.add_argument('--end', default='2024-10-31', help='Data fim')
    parser.add_argument('--lookback', type=int, default=128, help='Número de barras no passado')
    parser.add_argument('--horizon', type=int, default=10, help='Barras no futuro para label')
    parser.add_argument('--out_dir', default='./dl_data', help='Diretório de saída')

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"📅 Período: {args.start} a {args.end}")
    print(f"🔍 Lookback: {args.lookback} barras")
    print(f"🎯 Horizon: {args.horizon} barras\n")
    print("="*70 + "\n")

    all_X = []
    all_y = []
    all_timestamps = []
    all_setups = []

    for setup in SETUPS_VALIDADOS:
        print(f"📊 Processando {setup['name']}...")

        # Carregar dados
        df = load_ohlcv_data(args.data_root, args.symbol, setup['tf'], args.start, args.end)

        if df is None or len(df) == 0:
            print(f"  ❌ Sem dados para {setup['name']}\n")
            continue

        print(f"  ✅ Carregou {len(df):,} barras")

        # Calcular features
        df = calculate_technical_indicators(df)
        print(f"  ✅ Calculou indicadores técnicos")

        # Gerar labels
        df = generate_labels_from_future_returns(df, horizon=args.horizon)
        print(f"  ✅ Gerou labels (retornos futuros)")

        # Criar sequências
        X, y, timestamps = create_sequences(df, lookback=args.lookback)

        if len(X) == 0:
            print(f"  ❌ Sem sequências válidas\n")
            continue

        print(f"  ✅ Criou {len(X):,} sequências")

        # Estatísticas de labels
        label_1_pct = (y == 1).mean() * 100
        print(f"  📊 Label=1 (bons trades): {label_1_pct:.1f}%")
        print(f"  📊 Label=0 (ruins): {100-label_1_pct:.1f}%\n")

        all_X.append(X)
        all_y.append(y)
        all_timestamps.extend(timestamps)
        all_setups.extend([setup['name']] * len(X))

    if len(all_X) == 0:
        print("❌ Nenhum dado gerado!")
        return

    # Concatenar todos os setups
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    print("="*70)
    print(f"\n📦 DATASET FINAL:\n")
    print(f"  Total de sequências: {len(X_combined):,}")
    print(f"  Shape de X: {X_combined.shape}")
    print(f"  Shape de y: {y_combined.shape}")
    print(f"  Label=1: {(y_combined == 1).sum():,} ({(y_combined == 1).mean()*100:.1f}%)")
    print(f"  Label=0: {(y_combined == 0).sum():,} ({(y_combined == 0).mean()*100:.1f}%)\n")

    # Salvar datasets
    print("💾 Salvando datasets...\n")

    np.save(out_dir / 'X_sequences.npy', X_combined)
    np.save(out_dir / 'y_labels.npy', y_combined)

    # Salvar metadados
    metadata = {
        'n_sequences': len(X_combined),
        'sequence_length': X_combined.shape[1],
        'n_features': X_combined.shape[2],
        'lookback': args.lookback,
        'horizon': args.horizon,
        'label_1_count': int((y_combined == 1).sum()),
        'label_0_count': int((y_combined == 0).sum()),
        'setups': SETUPS_VALIDADOS,
        'period': f"{args.start} to {args.end}",
        'generated_at': datetime.now().isoformat()
    }

    import json
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ X_sequences.npy ({X_combined.shape})")
    print(f"  ✅ y_labels.npy ({y_combined.shape})")
    print(f"  ✅ metadata.json\n")

    print("="*70)
    print("\n✅ DATASET PRONTO PARA TREINO!\n")
    print("📊 Próximo passo:")
    print("   python3 treinar_dl.py --data_dir ./dl_data\n")

if __name__ == '__main__':
    main()
