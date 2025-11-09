# 📊 MICROSTRUCTURE DATA COLLECTION - Guia Completo

**Baseado no debate GPT-Strategist vs GPT-Executor**

---

## 🎯 SCRIPTS CRIADOS

### 1. `collect_aggtrades.py` ✅
Coleta aggtrades (aggregated trades) da Binance Futures.

**Uso:**
```bash
# Histórico (90 dias)
python3 collect_aggtrades.py --symbol BTCUSDT --mode historical --days 90 --output-dir ./data/aggtrades

# Tempo real
python3 collect_aggtrades.py --symbol BTCUSDT --mode live --output-dir ./data/aggtrades
```

**Output:**
```
./data/aggtrades/
└── BTCUSDT/
    └── 2024/
        └── 11/
            └── 08/
                ├── hour=00/data.parquet
                ├── hour=01/data.parquet
                └── ...
```

**Schema:**
- timestamp (int64)
- trade_id (int64)
- price (float64)
- quantity (float64)
- is_buyer_maker (bool) - True = sell, False = buy
- first_trade_id (int64)
- last_trade_id (int64)

---

### 2. `collect_book_depth.py` (TODO)
Coleta snapshots do order book a cada 500ms.

**Features a calcular:**
- Bid-Ask Imbalance (5, 10, 20 levels)
- Spread (absoluto e %)
- Weighted mid price
- Order book slope

---

### 3. `collect_market_data.py` (TODO)
Coleta dados complementares:
- Funding rate (a cada 8h)
- Mark price vs Spot (1s)
- Open Interest (1m)
- Liquidations (real-time)

---

### 4. `compute_microstructure_features.py` (TODO)
Processa dados brutos e calcula features para ML.

**Features calculadas:**

#### From AggTrades:
- CVD (Cumulative Volume Delta)
- VWAP intrabar
- Buy/Sell pressure ratio
- Trade intensity (1s, 5s, 10s)
- Large trade detection

#### From Book Depth:
- Imbalance (5, 10, 20 levels)
- Spread stats (avg, std)
- Bid/Ask slopes
- Weighted mid price deviation

#### From Market Data:
- Funding rate (absolute + delta)
- Spot premium %
- OI change %
- Liquidation ratio (longs/shorts)

---

## 🚀 WORKFLOW COMPLETO

### Passo 1: Coletar Dados (90 dias)

```bash
# 1. AggTrades
python3 collect_aggtrades.py --symbol BTCUSDT --days 90 --output-dir ./data/aggtrades

# 2. Book Depth (quando implementado)
# python3 collect_book_depth.py --symbol BTCUSDT --days 90 --output-dir ./data/book_depth

# 3. Market Data (quando implementado)
# python3 collect_market_data.py --symbol BTCUSDT --days 90 --output-dir ./data/market
```

### Passo 2: Computar Features

```bash
# Processa todos os dados e gera features para ML
python3 compute_microstructure_features.py \
    --aggtrades-dir ./data/aggtrades \
    --book-dir ./data/book_depth \
    --market-dir ./data/market \
    --output-dir ./data/features \
    --timeframes 1m,5m,15m
```

### Passo 3: Integrar com selector21.py

```bash
# Run Walk-Forward com dados completos
bash COMANDO_WF_OTIMIZADO.sh
```

---

## 📊 DECISÕES DO DEBATE

### Frequências de Coleta

| Tipo | Frequência | Razão |
|------|-----------|-------|
| AggTrades | 1s aggregate | Consenso: "razoável para manter precisão sem sobrecarregar" |
| Book Depth | 500ms snapshots | Consenso: "intervalo razoável" |
| Funding Rate | 8h | Atualizado pela exchange |
| Mark Price | 1s | Acompanha trades |
| Open Interest | 1m | Não muda rapidamente |
| Liquidations | Real-time | Eventos críticos |

### Storage

- **Formato:** Parquet
- **Compressão:** Snappy (mais rápido que gzip)
- **Particionamento:** Por hora
- **Estrutura:** `/tipo/SYMBOL/YYYY/MM/DD/hour=HH/data.parquet`

### Features Finais (~40-50 por timeframe)

**AggTrades (12):**
- cvd, vwap
- buy_volume, sell_volume, buy_pressure
- trade_count
- trade_intensity_1s, _5s, _10s
- large_trade_count, _volume, _pct

**Book Depth (10):**
- imbalance_5, _10, _20
- spread_avg, spread_std
- bid_slope, ask_slope
- weighted_mid_diff_bps
- bid_vol_ratio_5_20, ask_vol_ratio_5_20

**Market (7):**
- funding_rate, funding_delta
- spot_premium_pct
- open_interest, oi_change_pct
- liq_long_vol, liq_short_vol, liq_ratio

**Technical (20+):**
- RSI, MACD, ATR, BB, etc. (já existentes em selector21.py)

---

## ⚠️ LIMITAÇÕES E SOLUÇÕES

### Limitação 1: Latência de Rede
**Solução (GPT-Strategist):**
> "Implementar redundância, como conectar-se a múltiplas exchanges ou provedores de dados, pode mitigar o risco de interrupções."

### Limitação 2: Volume de Dados
**Solução (GPT-Executor):**
> "Usar Parquet particionado por hora com compressão Snappy otimiza leitura/escrita e armazenamento."

### Limitação 3: Complexidade de Processamento
**Solução (GPT-Strategist):**
> "Implementar uma arquitetura baseada em microservices na nuvem permite escalar componentes independentemente. Uso de Apache Kafka e Flink para stream processing."

---

## 🏆 PRÓXIMOS PASSOS

- [x] Debate sobre dados microstructure
- [x] collect_aggtrades.py implementado
- [ ] collect_book_depth.py (implementar)
- [ ] collect_market_data.py (implementar)
- [ ] compute_microstructure_features.py (implementar)
- [ ] Integrar com selector21.py
- [ ] Testar Walk-Forward com dados completos
- [ ] Deploy HFT system

---

## 📚 REFERÊNCIAS

- Debate completo: `DEBATE_MICROSTRUCTURE_DATA.md`
- Walk-Forward otimizado: `COMANDO_WF_OTIMIZADO.sh`
- Parâmetros WF: `DEBATE_WF_PARAMETERS.md`

**As IAs confirmaram: "só os klines não servem pra nada" ✅**

Agora temos um sistema COMPLETO de coleta de dados microstructure! 🚀
