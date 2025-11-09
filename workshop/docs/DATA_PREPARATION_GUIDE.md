# 📊 GUIA DE PREPARAÇÃO DE DADOS - BotScalp v3

**Passo a passo completo para preparar dados e rodar Walk-Forward backtest**

---

## 🎯 OBJETIVO

Antes de rodar o HFT em produção, você precisa:
1. ✅ **Baixar dados históricos** da Binance (klines/OHLCV)
2. ✅ **Organizar em parquets** para selector21.py
3. ✅ **Rodar Walk-Forward backtest** (treinar + validar)
4. ✅ **Treinar modelos ML** (XGBoost, RF, LogReg)
5. ✅ **Validar performance** (win rate, sharpe, etc)

---

## 📥 PASSO 1: BAIXAR DADOS DA BINANCE

### **Opção A: Script Automático (Recomendado)**

```bash
# Baixar 3 meses de BTCUSDT em 1m, 5m, 15m
python3 download_binance_data.py \
    --symbol BTCUSDT \
    --timeframe 1m,5m,15m \
    --days 90 \
    --output-dir ./data \
    --with-indicators
```

**O que isso faz:**
- ✅ Baixa klines (OHLCV) da Binance
- ✅ Converte para parquet otimizado
- ✅ Adiciona indicadores (RSI, MACD, ATR, BB)
- ✅ Salva em `./data/BTCUSDT_1m.parquet`, etc

**Tempo:** ~5-10 minutos para 90 dias

**Parâmetros:**
- `--days 90`: 3 meses (mínimo recomendado para WF)
- `--days 180`: 6 meses (melhor)
- `--days 365`: 1 ano (ideal para produção)
- `--with-indicators`: Adiciona RSI, MACD, ATR

---

### **Opção B: Já Tem Parquets? (Recuperar do Backup)**

Se você já tinha parquets antes:

```bash
# Listar backups disponíveis
find /root /home /opt -name "*.parquet" -o -name "*.pq" 2>/dev/null

# Copiar para diretório correto
mkdir -p ./data
cp /caminho/para/seus/parquets/*.parquet ./data/

# Verificar
ls -lh ./data/*.parquet
```

---

### **Opção C: Download Manual do Binance Vision**

Se preferir baixar direto do site oficial:

1. Acesse: https://data.binance.vision/
2. Navegue: `data/spot/monthly/klines/BTCUSDT/1m/`
3. Baixe ZIP dos meses desejados
4. Descompacte e converta para parquet (script abaixo)

```python
# convert_csv_to_parquet.py
import pandas as pd
from pathlib import Path

csv_files = Path('./binance_data').glob('*.csv')
for csv_file in csv_files:
    df = pd.read_csv(csv_file, names=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    output = f"./data/{csv_file.stem}.parquet"
    df.to_parquet(output, engine='pyarrow', compression='snappy', index=False)
    print(f"✅ {output}")
```

---

## 🔬 PASSO 2: RODAR WALK-FORWARD BACKTEST

Agora que você tem os dados, rode o backtest Walk-Forward:

### **2.1: Backtest Rápido (Teste)**

```bash
# Teste com 1 mês de dados
python3 selector21.py \
    --symbol BTCUSDT \
    --data_dir ./data \
    --start 2024-11-01 \
    --end 2024-11-30 \
    --exec_rules "1m,5m,15m" \
    --run_base \
    --print_top10
```

**Tempo:** ~2-5 minutos
**Output:** Top 10 melhores métodos base

---

### **2.2: Walk-Forward COMPLETO (Produção)**

```bash
# Walk-Forward: 3 meses train, 1 mês val, step 1 mês
python3 selector21.py \
    --symbol BTCUSDT \
    --data_dir ./data \
    --start 2024-06-01 \
    --end 2024-12-01 \
    --exec_rules "1m,5m,15m" \
    --walkforward \
    --wf_train_months 3 \
    --wf_val_months 1 \
    --wf_step_months 1 \
    --run_base \
    --run_combos \
    --run_ml \
    --ml_model_kind auto \
    --ml_save_dir ./ml_models \
    --ml_use_agg \
    --ml_use_depth \
    --ml_opt_thr \
    --use_atr_stop \
    --use_atr_tp \
    --hard_stop_usd "60,80,100" \
    --hard_tp_usd "300,360,400" \
    --print_top10
```

**Tempo:** ~30-60 minutos (depende dos dados)

**O que isso faz:**
1. ✅ Divide dados em janelas Walk-Forward
2. ✅ Treina em 3 meses, valida em 1 mês
3. ✅ Step de 1 mês (avança janela)
4. ✅ Testa métodos base + combos
5. ✅ Treina modelos ML (XGBoost, RF, LogReg)
6. ✅ Otimiza threshold de decisão
7. ✅ Salva melhores modelos em `./ml_models/`
8. ✅ Gera relatório completo

**Output esperado:**
```
./ml_models/
├── model_BTCUSDT_1m_xgb_wf0.pkl
├── scaler_BTCUSDT_1m_xgb_wf0.pkl
├── model_BTCUSDT_5m_rf_wf0.pkl
├── scaler_BTCUSDT_5m_rf_wf0.pkl
├── model_BTCUSDT_15m_logreg_wf0.pkl
└── scaler_BTCUSDT_15m_logreg_wf0.pkl

./wf_results/
└── BTCUSDT_wf_report.json
```

---

## 📊 PASSO 3: ANALISAR RESULTADOS

### **3.1: Ver Top 10 Métodos**

```bash
# Já mostrado no final do selector21.py
# Procure por:
#
# TOP 10 METHODS:
# 1. combo_xgb_1m_5m: Win Rate 62.5%, Sharpe 1.85
# 2. ml_rf_5m: Win Rate 58.3%, Sharpe 1.42
# ...
```

### **3.2: Validar Modelos ML**

```bash
# Testa se modelos foram salvos corretamente
python3 model_signal_generator.py
```

**Output esperado:**
```
🔍 Carregando modelos de: ml_models
   Encontrados: 6 modelos ML, 6 scalers
   ✅ Loaded: 1m xgb
   ✅ Loaded: 5m rf
   ✅ Loaded: 15m logreg

✅ Modelos carregados!
```

---

## 🎯 PASSO 4: VALIDAÇÃO FINAL

Antes de rodar o HFT, valide que está tudo pronto:

### **Checklist de Validação:**

```bash
# 1. Dados parquet existem?
ls -lh ./data/*.parquet

# 2. Modelos ML foram treinados?
ls -lh ./ml_models/*.pkl

# 3. Signal generator funciona?
python3 model_signal_generator.py

# 4. Paper trading funciona?
python3 run_production_paper_trading.py --trades 1
```

Se TODOS passarem: ✅ **Pronto para HFT!**

---

## 🚀 PASSO 5: RODAR HFT

Agora sim, com modelos treinados e validados:

```bash
# HFT com 30 trades/dia
python3 run_high_frequency_trading.py \
    --auto \
    --target-trades-per-day 30 \
    --min-confidence 0.60 \
    --models-dir ./ml_models
```

---

## 📐 ESTRUTURA DE DIRETÓRIOS FINAL

```
/opt/botscalpv3/
├── data/                           # Dados históricos
│   ├── BTCUSDT_1m.parquet         # 1 minuto
│   ├── BTCUSDT_5m.parquet         # 5 minutos
│   └── BTCUSDT_15m.parquet        # 15 minutos
│
├── ml_models/                      # Modelos treinados
│   ├── model_BTCUSDT_1m_xgb_wf0.pkl
│   ├── scaler_BTCUSDT_1m_xgb_wf0.pkl
│   ├── model_BTCUSDT_5m_rf_wf0.pkl
│   ├── scaler_BTCUSDT_5m_rf_wf0.pkl
│   └── ...
│
├── wf_results/                     # Resultados Walk-Forward
│   ├── BTCUSDT_wf_report.json
│   └── leaderboard.csv
│
├── selector21.py                   # Backtest engine
├── download_binance_data.py        # Data downloader
├── model_signal_generator.py       # Signal generator
├── run_production_paper_trading.py # Production trading
└── run_high_frequency_trading.py   # HFT mode
```

---

## ⏱️ RESUMO DE TEMPO

| Etapa | Tempo Estimado |
|-------|---------------|
| Download dados (90 dias) | ~5-10 min |
| Walk-Forward backtest | ~30-60 min |
| Treinar modelos ML | ~10-20 min |
| Validação | ~5 min |
| **TOTAL** | **~1-2 horas** |

---

## 🔧 TROUBLESHOOTING

### **"No module named 'pyarrow'"**
```bash
pip3 install pyarrow
```

### **"BinanceAPIException: Invalid symbol"**
→ Símbolo incorreto. Use `BTCUSDT`, não `BTC/USDT`

### **"FileNotFoundError: data/BTCUSDT_1m.parquet"**
→ Execute primeiro o download_binance_data.py

### **selector21.py muito lento**
→ Reduza período:
```bash
--start 2024-10-01 --end 2024-11-30  # 2 meses ao invés de 6
```

### **Modelos não foram salvos**
→ Verifique se usou `--ml_save_dir ./ml_models`

---

## 📞 COMANDOS COMPLETOS - COPIAR E COLAR

### **Para ter TUDO rodando hoje:**

```bash
# 1. Baixar dados (90 dias)
python3 download_binance_data.py \
    --symbol BTCUSDT \
    --timeframe 1m,5m,15m \
    --days 90 \
    --output-dir ./data \
    --with-indicators

# 2. Walk-Forward backtest + treinar ML
python3 selector21.py \
    --symbol BTCUSDT \
    --data_dir ./data \
    --start 2024-08-01 \
    --end 2024-11-08 \
    --exec_rules "1m,5m,15m" \
    --walkforward \
    --wf_train_months 2 \
    --wf_val_months 1 \
    --wf_step_months 1 \
    --run_ml \
    --ml_save_dir ./ml_models \
    --ml_model_kind auto \
    --ml_opt_thr \
    --use_atr_stop \
    --use_atr_tp \
    --print_top10

# 3. Validar modelos
python3 model_signal_generator.py

# 4. Rodar HFT!
python3 run_high_frequency_trading.py \
    --auto \
    --target-trades-per-day 30 \
    --min-confidence 0.60
```

**Total:** ~1-2 horas e você está rodando HFT com modelos reais! 🚀

---

**ESTE É O CAMINHO!** Siga passo a passo e em algumas horas terá o sistema completo funcionando! 💪
