# Próximos Passos - Após Downloads Completarem

## ✅ Verificar Downloads

```bash
# 1. Ver estrutura baixada
find data -type d | sort

# 2. Contar arquivos
echo "AggTrades:" && find data/aggTrades -name "*.parquet" | wc -l
echo "Klines:" && find data/klines -name "*.parquet" | wc -l

# 3. Ver tamanho total
du -sh data/*

# 4. Verificar logs finais
tail -20 /tmp/download_aggtrades_BTCUSDT.log
tail -20 /tmp/download_klines_BTCUSDT.log
```

---

## 🧪 Testar Dados Baixados

```python
import pandas as pd
from pathlib import Path

# Testar aggTrades
print("Testing aggTrades...")
bt_trades = list(Path('data/aggTrades/BTCUSDT').glob('*.parquet'))
print(f"  Files: {len(bt_trades)}")

if bt_trades:
    df = pd.read_parquet(bt_trades[0])
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Rows: {len(df):,}")
    print(f"  Date: {pd.to_datetime(df['timestamp'].iloc[0], unit='ms')}")

# Testar klines
print("\nTesting klines...")
for tf in ['1m', '5m', '15m', '1h', '4h', '1d']:
    files = list(Path(f'data/klines/{tf}/BTCUSDT').glob('*.parquet'))
    if files:
        df = pd.read_parquet(files[0])
        print(f"  {tf}: {len(files)} files, {len(df)} rows")
```

---

## 📊 Próximos Passos - Desenvolvimento

### 1. Criar Pipeline de Dados

```bash
# Features básicas
python3 create_features.py \
    --input data/aggTrades/BTCUSDT \
    --output features/btc_basic.parquet \
    --timeframe 1m
```

### 2. Multi-Timeframe Features

```python
# Combinar múltiplos timeframes
from feature_engineering import MultiTimeframeFeatures

mtf = MultiTimeframeFeatures(
    klines_dir='data/klines',
    symbol='BTCUSDT',
    timeframes=['1m', '5m', '15m', '1h']
)

features = mtf.create_features()
features.to_parquet('features/btc_mtf.parquet')
```

### 3. Backtesting

```python
from backtester import Backtester

bt = Backtester(
    data_dir='data',
    symbol='BTCUSDT',
    timeframe='5m',
    initial_capital=10000
)

results = bt.run(
    strategy='your_strategy',
    start_date='2023-01-01',
    end_date='2024-01-01'
)

print(f"Sharpe: {results['sharpe']}")
print(f"Max DD: {results['max_drawdown']}")
print(f"Win Rate: {results['win_rate']}")
```

### 4. Walk-Forward Optimization

```python
from wf_optimizer import WalkForwardOptimizer

wf = WalkForwardOptimizer(
    data_dir='data/klines/5m/BTCUSDT',
    train_window=180,  # dias
    test_window=30,
    anchored=False
)

results = wf.optimize(
    strategy_params={
        'rsi_period': [10, 14, 20],
        'rsi_overbought': [65, 70, 75],
        'rsi_oversold': [25, 30, 35]
    }
)

wf.plot_results()
```

### 5. ML/DL Training

```python
from ml_trainer import MLTrainer

trainer = MLTrainer(
    features_dir='features',
    target='returns_5m',
    test_size=0.2
)

# Train model
model = trainer.train(
    model_type='xgboost',  # ou 'lightgbm', 'lstm', 'transformer'
    cv_folds=5
)

# Evaluate
metrics = trainer.evaluate(model)
print(f"Accuracy: {metrics['accuracy']}")
print(f"Precision: {metrics['precision']}")
print(f"AUC: {metrics['auc']}")

# Save
trainer.save_model(model, 'models/xgb_btc_v1.pkl')
```

---

## 📚 Estrutura de Projeto Recomendada

```
/opt/botscalpv3/
├── data/                    # Dados brutos (já tem!)
│   ├── aggTrades/
│   └── klines/
│
├── features/                # Features processadas
│   ├── btc_basic.parquet
│   ├── eth_basic.parquet
│   └── mtf_features.parquet
│
├── models/                  # Modelos treinados
│   ├── xgb_btc_v1.pkl
│   └── lstm_eth_v1.h5
│
├── backtests/               # Resultados de backtests
│   ├── strategy_a/
│   └── strategy_b/
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   └── 03_modeling.ipynb
│
└── src/                     # Código fonte
    ├── feature_engineering.py
    ├── backtester.py
    ├── ml_trainer.py
    └── strategy.py
```

---

## 🎯 Plano de 7 Dias

### Dia 1-2: Exploração
- Carregar dados
- Análise exploratória
- Verificar qualidade
- Identificar padrões

### Dia 3-4: Features
- Criar features técnicas
- Multi-timeframe aggregation
- Order flow features (aggTrades)
- Cross-market features (BTC vs ETH)

### Dia 5-6: Backtesting
- Estratégia base
- Walk-forward validation
- Parameter optimization
- Análise de resultados

### Dia 7: ML/DL
- Feature selection
- Train/test split
- Model training
- Evaluation

---

## 📖 Recursos Úteis

### Bibliotecas Python:
```bash
pip install vectorbt pandas-ta ta-lib scikit-learn xgboost lightgbm
pip install tensorflow pytorch optuna  # ML/DL
```

### Documentação:
- VectorBT: https://vectorbt.dev/
- TA-Lib: https://ta-lib.org/
- XGBoost: https://xgboost.readthedocs.io/
- Optuna: https://optuna.org/

---

## 🔍 Debugging

Se algo deu errado:

```bash
# Ver processos que falharam
grep -i error /tmp/download_*.log

# Rerun para completar arquivos faltantes
./DOWNLOAD_2_ANOS_COMPLETO.sh  # Vai pular já baixados

# Verificar integridade
python3 verify_data_integrity.py
```

---

**Quando downloads terminarem, comece pela exploração dos dados!**
