# 🚀 DEEP LEARNING QUICKSTART

**Status**: Scripts prontos para uso
**Objetivo**: Melhorar win rate de 75% → 80-85% usando LSTM

---

## ⚡ QUICK START (3 comandos)

```bash
# 1. Gerar dataset (2-4 horas)
python3 gerar_dataset_dl.py --start 2022-01-01 --end 2024-10-31

# 2. Treinar LSTM (4-8 horas com GPU, 12-24h CPU)
python3 treinar_dl.py --epochs 50 --batch_size 256

# 3. Validar modelo
python3 validar_dl.py --confidence_threshold 0.7
```

**Expectativa**: Accuracy >= 80% no test set após filtro de confidence

---

## 📊 O QUE OS SCRIPTS FAZEM

### 1. `gerar_dataset_dl.py` - Preparação dos Dados

**Input**:
- Dados históricos dos 4 setups validados (data_monthly/)
- Período: 2022-2024 (2-3 anos de dados)

**Output** (`./dl_data/`):
- `X_sequences.npy` - Sequências de 128 barras × ~50 features
- `y_labels.npy` - Labels (1=trade bom, 0=trade ruim)
- `metadata.json` - Metadados do dataset

**Features extraídas** (~50 total):
- OHLCV básico
- EMAs (9, 21, 50, 200)
- RSI (14)
- MACD + Signal + Histogram
- Bollinger Bands + Width
- ATR (14)
- Volume ratio
- Momentum (5, 10, 20 períodos)
- Volatility (20)
- Distance from EMAs

**Labels**:
- Baseado em retornos futuros
- Label=1 se retorno futuro > 0.1% (trade bom)
- Label=0 caso contrário (trade ruim)

**Comandos opcionais**:
```bash
# Dataset personalizado
python3 gerar_dataset_dl.py \
  --data_root ./data_monthly \
  --symbol BTCUSDT \
  --start 2022-01-01 \
  --end 2024-10-31 \
  --lookback 128 \
  --horizon 10 \
  --out_dir ./dl_data
```

---

### 2. `treinar_dl.py` - Treinamento do LSTM

**Arquitetura**:
```
Input: [batch, 128 barras, ~50 features]
  ↓
LSTM Layer 1 (256 units) + Dropout(0.3)
  ↓
LSTM Layer 2 (128 units) + Dropout(0.3)
  ↓
Fully Connected (64 units) + ReLU + Dropout(0.3)
  ↓
Output Layer (1 unit) + Sigmoid
  ↓
Output: [batch, 1] - Confidence [0-1]
```

**Treinamento**:
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.001)
- Batch size: 256
- Epochs: 50 (default)
- Train/Val split: 80/20
- Early stopping baseado em F1-Score

**Output** (`./dl_models/`):
- `trading_lstm_best.pth` - Melhor modelo (baseado em F1)
- `training_history.json` - Histórico de todas as epochs

**Comandos opcionais**:
```bash
# Treinamento personalizado
python3 treinar_dl.py \
  --data_dir ./dl_data \
  --hidden_size 256 \
  --dropout 0.3 \
  --batch_size 256 \
  --epochs 100 \
  --lr 0.001 \
  --val_split 0.2 \
  --out_dir ./dl_models
```

**Uso de GPU**:
- Auto-detecta CUDA se disponível
- GPU acelera 10-50x o treinamento
- CPU funciona mas é mais lento (12-24h vs 4-8h)

---

### 3. `validar_dl.py` - Validação do Modelo

**Testa** o modelo treinado em dados não vistos (últimos 20% cronologicamente)

**Métricas calculadas**:
- Accuracy, Precision, Recall, F1-Score
- Métricas COM e SEM filtro de confidence
- TP, FP, FN, TN
- Taxa de filtragem (% de sinais mantidos)

**Output** (`./dl_models/`):
- `validation_results.json` - Métricas detalhadas

**Comandos opcionais**:
```bash
# Validação personalizada
python3 validar_dl.py \
  --model_dir ./dl_models \
  --data_dir ./dl_data \
  --test_split 0.2 \
  --confidence_threshold 0.7
```

**Interpretação**:
```
Accuracy >= 80% após filtro → ✅ META ATINGIDA
Accuracy >= 75% após filtro → ⚡ BOM resultado
Accuracy < 75% após filtro → ⚠️ Retreinar ou ajustar threshold
```

---

## 🎯 WORKFLOW COMPLETO

```
1. PREPARAÇÃO
├── Carregar dados históricos dos 4 setups
├── Calcular ~50 features técnicas
├── Gerar labels baseado em retornos futuros
└── Salvar X_sequences.npy, y_labels.npy

2. TREINAMENTO
├── Carregar dataset
├── Split 80/20 (train/val)
├── Treinar LSTM por 50 epochs
├── Salvar melhor modelo (F1-Score)
└── Gerar training_history.json

3. VALIDAÇÃO
├── Carregar modelo treinado
├── Testar em últimos 20% dos dados
├── Calcular métricas com threshold 0.7
└── Salvar validation_results.json

4. INTEGRAÇÃO (Próximo passo)
├── Aplicar DL filter aos 4 setups
├── Filtrar apenas sinais com confidence > 0.7
├── Comparar win rate COM vs SEM DL
└── Se win rate >= 80% → Produção!
```

---

## 📊 EXPECTATIVA DE RESULTADOS

### Dataset:
- Sequências: ~50K-100K (depende do período)
- Label=1: ~40-60% (trades bons)
- Label=0: ~40-60% (trades ruins)

### Modelo Treinado:
- Train Accuracy: 75-85%
- Val Accuracy: 70-80%
- F1-Score: 0.75-0.85

### Após Filtro (confidence > 0.7):
- Sinais mantidos: 60-80%
- Accuracy (filtrado): **80-85%** ← META
- Ganho: +5-10% vs sem filtro

### Win Rate nos 4 Setups:
- **SEM DL**: 75% (atual)
- **COM DL**: 80-85% (esperado)
- **Ganho**: +5-10% win rate
- **PnL**: +19-38% mensal

---

## 🔧 REQUISITOS

### Python packages:
```bash
pip install torch torchvision torchaudio  # PyTorch
pip install numpy pandas pyarrow          # Data handling
```

### Hardware:
- RAM: 16GB+ (32GB recomendado)
- GPU: Opcional mas acelera 10-50x
- CPU: Funciona mas mais lento

### Dados:
- Arquivos parquet em `data_monthly/`
- BTCUSDT_5m.parquet
- BTCUSDT_15m.parquet

---

## ⚠️ TROUBLESHOOTING

### Erro: "X_sequences.npy not found"
→ Execute `gerar_dataset_dl.py` primeiro

### Erro: "trading_lstm_best.pth not found"
→ Execute `treinar_dl.py` primeiro

### Accuracy muito baixa (< 70%)
→ Possíveis causas:
- Dataset pequeno demais (< 10K sequências)
- Labels desbalanceadas (> 80% de uma classe)
- Overfitting (usar dropout maior: 0.4-0.5)
- Underfitting (treinar mais epochs: 100+)

### GPU out of memory
→ Reduzir batch_size: 128 ou 64

### Treinamento muito lento em CPU
→ Considere usar GPU ou cloud (Google Colab, AWS)

---

## 📈 PRÓXIMO PASSO: INTEGRAÇÃO

Depois de validar o modelo com accuracy >= 75%:

1. **Criar script de integração** que:
   - Carrega modelo treinado
   - Roda os 4 setups no selector21
   - Aplica DL filter aos sinais
   - Só executa trades com confidence > 0.7

2. **Backtest COM DL**:
   - Comparar PnL, win rate, Sharpe
   - COM vs SEM DL filter

3. **Paper Trading COM DL**:
   - Rodar por 2-4 semanas
   - Monitorar win rate real

4. **Produção**:
   - Se win rate >= 78% → Aprovar para prod!

---

## 💡 DICAS

1. **Comece pequeno**: Teste com 1 mês de dados primeiro
2. **GPU é opcional**: CPU funciona, só demora mais
3. **Monitore overfitting**: Val loss deve acompanhar train loss
4. **Ajuste threshold**: Teste 0.6, 0.7, 0.8 para ver trade-off
5. **Re-treinar mensalmente**: Modelo pode ficar desatualizado

---

**Status**: ✅ Scripts prontos para uso
**Tempo estimado**: 1-2 dias (preparação + treinamento + validação)
**Próximo arquivo**: `integrar_dl_selector.py` (quando modelo validado)
