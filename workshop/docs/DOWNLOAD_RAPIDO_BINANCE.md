# 🚀 DOWNLOAD RÁPIDO - Binance Public Data

**Método OFICIAL e MUITO MAIS RÁPIDO que API!**

Fonte: https://data.binance.vision/ (dados públicos oficiais da Binance)

---

## ⚡ POR QUE É MAIS RÁPIDO?

### Método ANTIGO (API):
- ❌ Limite de 1000 trades por request
- ❌ Rate limit: 1200 requests/minuto
- ❌ Precisa fazer milhares de requests
- ❌ **Tempo: 2-4 horas** para 90 dias

### Método NOVO (Binance Vision):
- ✅ Arquivos diários completos (ZIP)
- ✅ SEM rate limit
- ✅ Download paralelo possível
- ✅ **Tempo: 5-15 minutos** para 90 dias!

**~20x MAIS RÁPIDO!** 🔥

---

## 📥 DOWNLOAD RÁPIDO - 90 DIAS

### 1. AggTrades (Recommended!)

```bash
python3 download_binance_public_data.py \
    --data-type aggTrades \
    --symbol BTCUSDT \
    --market futures \
    --start-date 2024-08-01 \
    --end-date 2024-11-08 \
    --output-dir ./data
```

**Tempo:** ~5-10 minutos
**Tamanho:** ~500MB-1GB comprimido
**Output:** `./data/aggTrades/BTCUSDT/2024/MM/DD/hour=HH/data.parquet`

---

### 2. Klines (OHLCV) - Múltiplos Timeframes

```bash
python3 download_binance_public_data.py \
    --data-type klines \
    --symbol BTCUSDT \
    --market futures \
    --intervals 1m,5m,15m \
    --start-date 2024-08-01 \
    --end-date 2024-11-08 \
    --output-dir ./data
```

**Tempo:** ~10-15 minutos (para 3 timeframes)
**Output:**
```
./data/klines/
├── 1m/BTCUSDT/2024/...
├── 5m/BTCUSDT/2024/...
└── 15m/BTCUSDT/2024/...
```

---

## 🎯 WORKFLOW COMPLETO (90 DIAS)

### PASSO 1: Download dos Dados (10-15 min)

```bash
# AggTrades (para features microstructure)
python3 download_binance_public_data.py \
    --data-type aggTrades \
    --symbol BTCUSDT \
    --start-date 2024-08-01 \
    --end-date 2024-11-08 \
    --output-dir ./data

# Klines (para OHLCV base)
python3 download_binance_public_data.py \
    --data-type klines \
    --symbol BTCUSDT \
    --intervals 1m,5m,15m \
    --start-date 2024-08-01 \
    --end-date 2024-11-08 \
    --output-dir ./data
```

---

### PASSO 2: Computar Features (30-60 min)

```bash
python3 compute_microstructure_features.py \
    --symbol BTCUSDT \
    --start-date 2024-08-01 \
    --end-date 2024-11-08 \
    --timeframes 1min,5min,15min \
    --aggtrades-dir ./data/aggTrades \
    --output-dir ./data/features
```

---

### PASSO 3: Walk-Forward (1-2 horas)

```bash
bash COMANDO_WF_OTIMIZADO.sh
```

---

## 📊 DADOS DISPONÍVEIS

### ✅ DISPONÍVEL (Histórico Completo)

| Tipo | Descrição | Uso |
|------|-----------|-----|
| **aggTrades** | Aggregated trades | ✅ CVD, VWAP, pressure, intensity |
| **klines** | OHLCV candles | ✅ Base para todos os indicadores |
| **trades** | Individual trades | ⚠️ Muito pesado, use aggTrades |
| **fundingRate** | Funding rate (Futures) | ✅ 8h intervals |
| **markPriceKlines** | Mark price (Futures) | ✅ Premium/discount |
| **indexPriceKlines** | Index price | ✅ Spot vs futures |

### ❌ NÃO DISPONÍVEL (Histórico)

| Tipo | Alternativa |
|------|-------------|
| **bookDepth** | ❌ Histórico não disponível. Use tempo real ou ignore |
| **liquidations** | ❌ Não público. Use proxy (funding + OI) |

---

## 🔥 COMPARAÇÃO DE PERFORMANCE

### Cenário: 90 dias de BTCUSDT AggTrades

| Método | Tempo | Requests | Complexidade |
|--------|-------|----------|--------------|
| **API (antigo)** | 2-4 horas | ~100,000+ | 🔴 Alta |
| **Binance Vision** | **5-10 min** | ~90 | 🟢 Baixa |

**Speedup: 20-40x!** 🚀

---

## 💡 TIPS & TRICKS

### Download Paralelo (Ainda Mais Rápido!)

```bash
# Download em paralelo (múltiplos símbolos)
python3 download_binance_public_data.py --symbol BTCUSDT --data-type aggTrades --start-date 2024-08-01 --end-date 2024-11-08 &
python3 download_binance_public_data.py --symbol ETHUSDT --data-type aggTrades --start-date 2024-08-01 --end-date 2024-11-08 &
wait
```

### Monthly vs Daily

```bash
# Daily: Mais granular, melhor para períodos curtos (<30 dias)
--frequency daily

# Monthly: Arquivos maiores, melhor para períodos longos (>30 dias)
--frequency monthly
```

### Verificar Arquivos Baixados

```bash
# Ver estrutura
tree -L 5 ./data/aggTrades/

# Ver tamanho
du -sh ./data/aggTrades/*

# Ver primeiras linhas de um parquet
python3 -c "import pandas as pd; df = pd.read_parquet('./data/aggTrades/BTCUSDT/2024/11/08/hour=00/data.parquet'); print(df.head())"
```

---

## 🐛 TROUBLESHOOTING

### Erro: "File not found (404)"

**Problema:** Dia específico não tem dados (exchange offline, etc)

**Solução:** Normal! O script pula automaticamente.

```
[SKIP] File not found: https://data.binance.vision/data/.../2024-11-10.zip
```

---

### Download Interrompido

**Solução:** Re-rodar o script. Ele detecta arquivos existentes e continua de onde parou!

```bash
# Re-executar
python3 download_binance_public_data.py \
    --data-type aggTrades \
    --symbol BTCUSDT \
    --start-date 2024-08-01 \
    --end-date 2024-11-08
```

---

### Espaço em Disco

**90 dias de dados:**
- AggTrades: ~1-2GB (comprimido Parquet)
- Klines (1m+5m+15m): ~500MB
- Features computadas: ~2-3GB
- **Total: ~5GB**

---

## 📋 CHECKLIST RÁPIDO

- [ ] Instalar dependências: `pip install requests pandas pyarrow tqdm`
- [ ] Baixar AggTrades: `python3 download_binance_public_data.py --data-type aggTrades ...`
- [ ] Baixar Klines: `python3 download_binance_public_data.py --data-type klines --intervals 1m,5m,15m ...`
- [ ] Verificar download: `tree -L 5 ./data/`
- [ ] Computar features: `python3 compute_microstructure_features.py ...`
- [ ] Walk-Forward: `bash COMANDO_WF_OTIMIZADO.sh`
- [ ] 🚀 Deploy HFT!

---

## 🎯 RESULTADO FINAL

### Após download + compute:

```
./data/
├── aggTrades/           ← Dados brutos (1-2GB)
│   └── BTCUSDT/
├── klines/              ← OHLCV (500MB)
│   ├── 1m/
│   ├── 5m/
│   └── 15m/
└── features/            ← Features ML (~3GB)
    └── BTCUSDT/
        ├── 1min/        ← ~50 features
        ├── 5min/        ← ~50 features
        └── 15min/       ← ~50 features
```

**Total:** ~5GB, **~50 features** por timeframe

**Pronto para treinar modelos ML/DL!** 🎉

---

## 🔗 REFERÊNCIAS

- Binance Vision: https://data.binance.vision/
- Binance Public Data: https://github.com/binance/binance-public-data
- Debate Microstructure: `DEBATE_MICROSTRUCTURE_DATA.md`
- Guia Completo: `GUIA_COMPLETO_COLETA_DADOS.md`

---

**🚀 AGORA SIM! Download de 90 dias em 10 minutos!**

Obrigado pela dica! 💪
