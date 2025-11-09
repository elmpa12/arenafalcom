# 🚀 INSTRUÇÕES - Download 2 Anos no SEU Servidor (root@lab)

**Execute no seu servidor `root@lab`**

---

## 📋 PASSO A PASSO

### 1. Conectar ao seu servidor

```bash
ssh root@lab
```

---

### 2. Ir para o diretório do projeto

```bash
cd /opt/botscalpv3
```

---

### 3. Pull do GitHub (pegar scripts atualizados)

```bash
git pull origin claude/review-botscalpv3-project-011CUun7aaeuet1uVKikS4vS
```

---

### 4. Instalar dependências (se necessário)

```bash
pip3 install pandas pyarrow requests tqdm
```

---

### 5. EXECUTAR DOWNLOAD (Escolha uma opção)

#### **Opção A: Script Automático (RECOMENDADO)**

```bash
chmod +x DOWNLOAD_2_ANOS_COMPLETO.sh
./DOWNLOAD_2_ANOS_COMPLETO.sh
```

#### **Opção B: Comandos Individuais**

```bash
# AggTrades (2 anos)
nohup python3 download_binance_public_data.py \
    --data-type aggTrades \
    --symbol BTCUSDT \
    --market futures \
    --start-date 2022-11-08 \
    --end-date 2024-11-08 \
    --output-dir ./data \
    > /tmp/download_aggtrades.log 2>&1 &

# Klines (1m, 5m, 15m - 2 anos)
nohup python3 download_binance_public_data.py \
    --data-type klines \
    --symbol BTCUSDT \
    --market futures \
    --intervals 1m,5m,15m \
    --start-date 2022-11-08 \
    --end-date 2024-11-08 \
    --output-dir ./data \
    > /tmp/download_klines.log 2>&1 &
```

---

### 6. Acompanhar Progresso

```bash
# Ver logs em tempo real
tail -f /tmp/download_aggtrades.log
tail -f /tmp/download_klines.log

# Ou com watch (atualiza a cada 5s)
watch -n 5 'tail -3 /tmp/download_aggtrades.log; echo ""; tail -3 /tmp/download_klines.log'

# Ver PIDs dos processos
ps aux | grep download_binance
```

---

### 7. Verificar Dados Baixados

```bash
# Ver estrutura
tree -L 4 ./data/

# Ver tamanho
du -sh ./data/*

# Contar arquivos
find ./data -name "*.parquet" | wc -l
```

---

## ⏱️ TEMPO ESTIMADO

| Dados | Arquivos | Tamanho | Tempo |
|-------|----------|---------|-------|
| **AggTrades** | ~730 | ~10-12GB | ~10-15 min |
| **Klines 1m** | ~730 | ~2-3GB | ~8-10 min |
| **Klines 5m** | ~730 | ~1-2GB | ~6-8 min |
| **Klines 15m** | ~730 | ~500MB-1GB | ~5-7 min |
| **TOTAL** | ~2920 | **~15-20GB** | **~25-35 min** |

---

## 📊 ESTRUTURA FINAL

```
/opt/botscalpv3/data/
├── aggTrades/
│   └── BTCUSDT/
│       ├── 2022/
│       │   └── 11/
│       │       ├── 08/hour=00/data.parquet
│       │       ├── 08/hour=01/data.parquet
│       │       └── ...
│       ├── 2023/
│       └── 2024/
│
└── klines/
    ├── 1m/
    │   └── BTCUSDT/
    │       ├── 2022/11/...
    │       ├── 2023/...
    │       └── 2024/...
    ├── 5m/
    │   └── BTCUSDT/...
    └── 15m/
        └── BTCUSDT/...
```

---

## 🔍 MONITORAMENTO

### Ver progresso AggTrades:
```bash
tail -10 /tmp/download_aggtrades.log
```

Exemplo de output:
```
Downloading:  45%|████▌     | 330/732 [05:12<06:20,  1.06it/s]
[DOWNLOAD] https://data.binance.vision/.../BTCUSDT-aggTrades-2023-08-15.zip
[TOTAL] 8,234,567 trades
```

### Ver progresso Klines:
```bash
tail -10 /tmp/download_klines.log
```

---

## ⚠️ TROUBLESHOOTING

### Erro: "No module named 'pandas'"
```bash
pip3 install pandas pyarrow requests tqdm
```

### Erro: "File not found (404)"
**Normal!** Alguns dias podem não ter dados. O script pula automaticamente.

### Download travou?
```bash
# Ver se ainda está rodando
ps aux | grep download_binance

# Se travou, matar e reiniciar
pkill -f download_binance
./DOWNLOAD_2_ANOS_COMPLETO.sh
```

### Espaço em disco insuficiente?
```bash
# Ver espaço disponível
df -h /opt

# Precisa de ~20GB livres
```

---

## ✅ APÓS O DOWNLOAD

Quando terminar (~30 min), você terá:

✅ **2 anos de dados** (2022-11-08 → 2024-11-08)
✅ **~2920 arquivos** Parquet
✅ **~15-20GB** de dados históricos
✅ **Pronto para ML/DL!**

### Próximo passo:

```bash
# Computar features microstructure
python3 compute_microstructure_features.py \
    --symbol BTCUSDT \
    --start-date 2022-11-08 \
    --end-date 2024-11-08 \
    --timeframes 1min,5min,15min \
    --aggtrades-dir ./data/aggTrades \
    --output-dir ./data/features
```

---

## 🎯 RESUMO DOS COMANDOS

```bash
# No seu servidor root@lab:
cd /opt/botscalpv3
git pull
chmod +x DOWNLOAD_2_ANOS_COMPLETO.sh
./DOWNLOAD_2_ANOS_COMPLETO.sh

# Acompanhar
tail -f /tmp/download_aggtrades.log

# Verificar quando terminar
ls -lh ./data/aggTrades/BTCUSDT/2024/11/07/
```

---

**⏱️ Aguarde ~30 minutos e você terá 2 ANOS de dados prontos!** 🚀
