# Guia de Download de Dados da Binance

## Sumário

Este projeto possui dois métodos de download de dados históricos:

1. **Binance Vision** (data.binance.vision) - Dados históricos em arquivos ZIP
2. **Binance API REST** - Dados via API pública (com restrições geográficas)

---

## 📊 Dados Disponíveis

### ✅ Via Binance Vision (Recomendado)

**Vantagens:**
- Muito rápido (download paralelo de arquivos)
- Sem rate limits
- Funciona de qualquer localização

**Dados disponíveis:**
- ✅ **AggTrades** (trades agregados, tick-by-tick)
- ✅ **Klines** (OHLCV em todos os timeframes: 1m, 5m, 15m, 1h, 4h, 1d, etc)
- ✅ **Trades** individuais (não implementado ainda, use aggTrades)

**Script:** `download_binance_public_data.py`

### ⚠️ Via Binance API REST (Com Restrições)

**Vantagens:**
- Dados de futuros específicos (funding, OI, etc)
- Atualização mais recente

**Desvantagens:**
- ⚠️ **BLOQUEADO EM CERTAS REGIÕES** (EUA, etc)
- Rate limited (max ~1200 req/min)
- Mais lento

**Dados disponíveis (se não estiver bloqueado):**
- 💰 **Funding Rate** (a cada 8h)
- 📊 **Open Interest** (a cada 5min)
- 📈 **Long/Short Ratio** (a cada 5min)

**Script:** `download_futures_data_api.py`

---

## 🚀 Como Usar

### Método 1: Script Completo (Recomendado)

Baixa TODOS os dados de uma vez (2 anos, 3 símbolos):

```bash
chmod +x DOWNLOAD_2_ANOS_COMPLETO.sh
./DOWNLOAD_2_ANOS_COMPLETO.sh
```

**O que será baixado:**
- BTCUSDT, ETHUSDT, SOLUSDT
- AggTrades (2 anos)
- Klines (1m, 5m, 15m, 1h, 4h, 1d)
- Funding Rate* (se não bloqueado)
- Open Interest* (se não bloqueado)
- Long/Short Ratio* (se não bloqueado)

**Tempo estimado:** 60-90 minutos
**Espaço em disco:** ~40-50GB

### Método 2: Downloads Individuais

#### Baixar AggTrades:

```bash
python3 download_binance_public_data.py \
    --data-type aggTrades \
    --symbol BTCUSDT \
    --market futures \
    --start-date 2022-11-08 \
    --end-date 2024-11-08 \
    --output-dir ./data
```

#### Baixar Klines:

```bash
python3 download_binance_public_data.py \
    --data-type klines \
    --symbol BTCUSDT \
    --market futures \
    --intervals 1m,5m,15m,1h,4h,1d \
    --start-date 2022-11-08 \
    --end-date 2024-11-08 \
    --output-dir ./data
```

#### Baixar Funding Rate (requer acesso à API):

```bash
python3 download_futures_data_api.py \
    --data-type fundingRate \
    --symbol BTCUSDT \
    --start-date 2024-01-01 \
    --end-date 2024-11-08 \
    --output-dir ./data
```

---

## ⚠️ Problema: API Bloqueada por Região

### Erro 451 - "Service unavailable from a restricted location"

A Binance bloqueia acessos da API de futuros em certas regiões (EUA, etc).

**Sintomas:**
```
[ERROR] Funding rate: 451 Client Error
Service unavailable from a restricted location according to 'b. Eligibility'
```

### Soluções:

#### Opção 1: Usar apenas Binance Vision (Recomendado)

Binance Vision **funciona de qualquer lugar** e tem 95% dos dados necessários:

```bash
# Baixar apenas dados do Vision (sem funding/OI)
python3 download_binance_public_data.py \
    --data-type aggTrades \
    --symbol BTCUSDT \
    --market futures \
    --start-date 2022-11-08 \
    --end-date 2024-11-08 \
    --output-dir ./data

python3 download_binance_public_data.py \
    --data-type klines \
    --symbol BTCUSDT \
    --market futures \
    --intervals 1m,5m,15m,1h,4h,1d \
    --start-date 2022-11-08 \
    --end-date 2024-11-08 \
    --output-dir ./data
```

#### Opção 2: VPN/Proxy

Use um VPN para país não restrito (Brasil, Europa, Ásia):

```bash
# Conectar VPN para Brasil/Europa
# Depois rodar:
python3 download_futures_data_api.py --data-type all --symbol BTCUSDT ...
```

#### Opção 3: Calcular Funding Rate Aproximado

Você pode **estimar** funding rate usando mark price vs spot price:

```python
# funding_rate_approx = (mark_price - spot_price) / spot_price
# Não é perfeito, mas serve para backtesting
```

#### Opção 4: Usar dados de outra exchange

Alternativas sem restrição geográfica:
- **Bybit** (API aberta globalmente)
- **OKX** (API aberta)
- **Deribit** (sem restrições)

---

## 📁 Estrutura de Dados

Após o download completo:

```
./data/
├── aggTrades/
│   ├── BTCUSDT/
│   │   ├── BTCUSDT_aggTrades_2022-11-08.parquet
│   │   ├── BTCUSDT_aggTrades_2022-11-09.parquet
│   │   └── ...
│   ├── ETHUSDT/
│   └── SOLUSDT/
├── klines/
│   ├── 1m/
│   │   ├── BTCUSDT/
│   │   ├── ETHUSDT/
│   │   └── SOLUSDT/
│   ├── 5m/
│   ├── 15m/
│   ├── 1h/
│   ├── 4h/
│   └── 1d/
├── fundingRate/          # Apenas se API funcionar
│   ├── BTCUSDT/
│   │   └── BTCUSDT_fundingRate_2022-11-08_2024-11-08.csv
│   ├── ETHUSDT/
│   └── SOLUSDT/
├── openInterest/         # Apenas se API funcionar
│   └── ...
└── longShortRatio/       # Apenas se API funcionar
    └── ...
```

---

## 🔍 Verificar Disponibilidade

Antes de iniciar download completo, teste:

```bash
python3 test_binance_data_availability.py
```

Isso verifica:
- ✅ Quais dados estão disponíveis no Binance Vision
- ⚠️ Se a API está bloqueada na sua região

---

## 💡 Recomendações

### Para Backtesting de Scalping:

**Dados essenciais:**
- ✅ AggTrades (1s-1m granularidade)
- ✅ Klines 1m, 5m

**Dados opcionais (mas úteis):**
- Funding Rate (para regime detection)
- Open Interest (para confirmar tendências)

### Para ML/DL:

**Dados essenciais:**
- ✅ AggTrades
- ✅ Klines (múltiplos timeframes: 1m, 5m, 15m, 1h, 4h, 1d)

**Dados muito úteis:**
- Funding Rate (feature importante!)
- Open Interest (trend strength)
- Long/Short Ratio (sentiment)

### Para Walk-Forward Optimization:

**Dados essenciais:**
- ✅ Klines (15m, 1h, 4h, 1d)
- ✅ AggTrades (para fill simulation)

---

## 📊 Estatísticas de Download

Baseado em testes com servidor de 1Gbps:

| Tipo | Símbolo | Período | Tamanho | Tempo |
|------|---------|---------|---------|-------|
| AggTrades | BTCUSDT | 2 anos | ~8GB | 15-20min |
| Klines 1m | BTCUSDT | 2 anos | ~50MB | 2-3min |
| Klines 1h | BTCUSDT | 2 anos | ~5MB | <1min |
| Funding* | BTCUSDT | 2 anos | ~500KB | 5-10min |
| OI* | BTCUSDT | 2 anos | ~100MB | 10-15min |

\* Via API (se disponível)

---

## ❓ FAQ

### Preciso mesmo de 2 anos de dados?

Para **backtesting robusto com WF**, sim:
- Train: 1 ano
- Validation: 6 meses
- Test OOS: 6 meses

### Posso usar dados de spot ao invés de futures?

Sim! Mude `--market futures` para `--market spot`:

```bash
python3 download_binance_public_data.py \
    --data-type aggTrades \
    --symbol BTCUSDT \
    --market spot \
    ...
```

**Diferenças:**
- Spot: menor alavancagem, sem funding rate
- Futures: maior volume, dados adicionais (funding, OI)

### Por que SOL e não outras altcoins?

- Alta liquidez (~$2-5B volume/dia)
- Comportamento diferente de BTC/ETH
- Testa robustez em diferentes regimes de mercado

**Alternativas boas:**
- BNBUSDT
- XRPUSDT
- ADAUSDT
- DOGEUSDT

---

## 🆘 Troubleshooting

### "Connection timeout"

```bash
# Aumentar timeout no código ou retry
# Ou rodar de novo (downloads são incrementais)
./DOWNLOAD_2_ANOS_COMPLETO.sh
```

### "Disk space full"

```bash
# Verificar espaço
df -h

# Limpar downloads antigos
rm -rf /tmp/download_*.log
```

### API rate limited

```bash
# Aumentar delay no código:
# rate_limit_delay = 0.5  # 500ms entre requests
```

---

## 📚 Referências

- [Binance Vision](https://github.com/binance/binance-public-data)
- [Binance Futures API Docs](https://binance-docs.github.io/apidocs/futures/en/)
- [Restricted Locations](https://www.binance.com/en/terms)

---

**Dúvidas?** Abra uma issue ou consulte a documentação oficial da Binance.
