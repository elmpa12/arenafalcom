# 🎭 DEBATE: Microstructure Data - Beyond Klines

**Data:** 2025-11-08
**Participantes:** GPT-Strategist vs GPT-Executor
**Tema:** "Só os klines não servem pra nada" - Dados necessários para HFT

---

## 🚨 PROBLEMA IDENTIFICADO PELO USUÁRIO

> **"só os klines não servem pra nada. Temos que ter os aggtrades, os bookdepth e alguns outros dados tbm para ajudar os modelos a tomarem decisão."**

**O usuário está CORRETO!** As IAs confirmam:

---

## 💬 CONSENSO GERAL (Round 1)

### Por que klines (OHLCV) são insuficientes para HFT?

**GPT-Executor:**
> *"Os klines oferecem uma visão simplificada do mercado, resumindo informações em intervalos de tempo fixos. Para estratégias de high-frequency trading (HFT), que executam 30-60 trades por dia, é necessário um nível mais granular de dados para capturar movimentos rápidos e microestruturas do mercado. Os klines não fornecem informações sobre a sequência de trades ou a profundidade do mercado, que são cruciais para decisões rápidas."*

**Klines NÃO revelam:**
- ❌ Sequência de trades (momentum intrabar)
- ❌ Direção do fluxo (buyers vs sellers)
- ❌ Profundidade do livro de ordens
- ❌ Pressão de compra/venda
- ❌ Liquidez disponível
- ❌ Large trades (whales)

---

## 📊 DADOS NECESSÁRIOS

### 1. **AGGTRADES** (Aggregated Trades)

**O que revelam:**
- Sequência e direção dos trades (buyer-initiated vs seller-initiated)
- Volume por segundo/minuto
- Padrões de execução e momentum
- Detecção de large trades (baleias)

**Features a calcular:**

#### **CVD (Cumulative Volume Delta)**
```python
# Fórmula:
CVD = sum(volume_buy) - sum(volume_sell)

# Implementação:
for trade in aggtrades:
    if trade['is_buyer_maker']:  # Sell
        cvd -= trade['quantity']
    else:  # Buy
        cvd += trade['quantity']
```

#### **VWAP Intrabar**
```python
# Fórmula:
VWAP = sum(price * volume) / sum(volume)

# Por barra de 1m:
vwap_1m = df['price'].multiply(df['volume']).sum() / df['volume'].sum()
```

#### **Trade Intensity (trades/segundo)**
```python
# Janelas ideais: 1s, 5s, 10s
# GPT-Executor: "janelas móveis de 1s, 5s e 10s para capturar diferentes granularidades"

trade_intensity_1s = df.rolling('1s')['trade_id'].count()
trade_intensity_5s = df.rolling('5s')['trade_id'].count()
trade_intensity_10s = df.rolling('10s')['trade_id'].count()
```

#### **Buy/Sell Pressure Ratio**
```python
# Fórmula:
buy_pressure = volume_buy / total_volume

# Rolling window de 1 minuto:
df['is_buy'] = ~df['is_buyer_maker']
df['buy_vol'] = df['quantity'].where(df['is_buy'], 0)
df['sell_vol'] = df['quantity'].where(~df['is_buy'], 0)

buy_pressure = df['buy_vol'].rolling('1min').sum() / df['quantity'].rolling('1min').sum()
```

#### **Large Trade Detection**
```python
# Threshold: % do volume médio
# GPT-Executor: "threshold baseado em um percentual do volume médio, calculado por um
# rolling mean sobre um período pré-definido, como 15 minutos ou 1 hora"

volume_mean_15m = df['quantity'].rolling('15min').mean()
large_trade_threshold = volume_mean_15m * 2.0  # 200% do volume médio

df['is_large_trade'] = df['quantity'] > large_trade_threshold
large_trades = df[df['is_large_trade']]
```

---

### 2. **BOOK DEPTH** (Order Book)

**O que revela:**
- Liquidez disponível em cada nível de preço
- Desequilíbrio entre compra/venda (pressure)
- Bid-ask spread (custo de execução)
- "Paredes" de ordens (grandes ordens limitadas)

**Features a calcular:**

#### **Bid-Ask Imbalance**
```python
# Fórmula CORRETA (confirmada pelo GPT-Executor):
bid_ask_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

# Implementação:
def calc_imbalance(book_snapshot):
    bid_vol = sum([level['quantity'] for level in book_snapshot['bids']])
    ask_vol = sum([level['quantity'] for level in book_snapshot['asks']])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)
```

#### **Depth Imbalance em N Levels**
```python
# N ideal: 10 ou 20 levels (GPT-Executor)
# "a escolha de N níveis deve ser baseada na profundidade típica do mercado,
# com 10 ou 20 sendo comuns"

def depth_imbalance(book, n_levels=10):
    bid_vol = sum([level['quantity'] for level in book['bids'][:n_levels]])
    ask_vol = sum([level['quantity'] for level in book['asks'][:n_levels]])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

imbalance_5 = depth_imbalance(book, 5)
imbalance_10 = depth_imbalance(book, 10)
imbalance_20 = depth_imbalance(book, 20)
```

#### **Order Book Slope** (Regressão Linear)
```python
# GPT-Executor: "capturar snapshots do book depth e aplicar uma regressão
# nos níveis de preço e volume"

from scipy.stats import linregress

def book_slope(book_side, n_levels=20):
    """
    book_side: book['bids'] ou book['asks']
    Retorna: slope da regressão linear (volume vs distance from mid)
    """
    prices = [float(level['price']) for level in book_side[:n_levels]]
    volumes = [float(level['quantity']) for level in book_side[:n_levels]]

    # Distance from best price
    distances = [abs(p - prices[0]) for p in prices]

    slope, intercept, r_value, p_value, std_err = linregress(distances, volumes)
    return slope

bid_slope = book_slope(book['bids'])
ask_slope = book_slope(book['asks'])
```

#### **Weighted Mid Price vs Last Price**
```python
# Diferença em basis points (bps)

def weighted_mid_price(book):
    """
    Weighted by volume at best bid/ask
    """
    best_bid = float(book['bids'][0]['price'])
    best_ask = float(book['asks'][0]['price'])
    bid_vol = float(book['bids'][0]['quantity'])
    ask_vol = float(book['asks'][0]['quantity'])

    wmp = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
    return wmp

wmp = weighted_mid_price(book)
last_price = float(aggtrades[-1]['price'])

diff_bps = ((wmp - last_price) / last_price) * 10000  # basis points
```

#### **Time-Weighted Average Spread**
```python
# GPT-Executor: "requer o cálculo do spread médio ponderado pelo tempo
# dentro de um intervalo"

def time_weighted_avg_spread(snapshots):
    """
    snapshots: lista de (timestamp, book) snapshots
    """
    total_time = 0
    weighted_spread = 0

    for i in range(len(snapshots) - 1):
        t1, book1 = snapshots[i]
        t2, book2 = snapshots[i + 1]

        spread = float(book1['asks'][0]['price']) - float(book1['bids'][0]['price'])
        time_delta = (t2 - t1).total_seconds()

        weighted_spread += spread * time_delta
        total_time += time_delta

    return weighted_spread / total_time if total_time > 0 else 0
```

---

### 3. **OUTROS DADOS CRÍTICOS**

#### **Funding Rate**
```python
# Usar absoluto E delta
# GPT: "tanto o valor absoluto quanto a mudança são importantes"

funding_rate_abs = current_funding_rate
funding_rate_delta = current_funding_rate - previous_funding_rate

# Threshold de alerta: ±0.01% (0.0001) é considerado neutro
# Acima de ±0.05% (0.0005) indica forte pressão
if abs(funding_rate_abs) > 0.0005:
    alert = "HIGH_FUNDING_PRESSURE"
```

#### **Mark Price vs Spot Price**
```python
# Premium/Discount em %

premium_pct = ((mark_price - spot_price) / spot_price) * 100

# Positivo = futures mais caro (bullish)
# Negativo = futures mais barato (bearish)
```

#### **Open Interest**
```python
# Usar % change, não absoluto
# GPT: mudanças no OI são mais informativas que valor absoluto

oi_change_pct = ((current_oi - previous_oi) / previous_oi) * 100

# OI subindo + preço subindo = bullish (novos longs)
# OI subindo + preço caindo = bearish (novos shorts)
# OI caindo = fechamento de posições
```

#### **Liquidations**
```python
# Volume de liquidações: longs vs shorts

liq_long_volume = sum([liq['quantity'] for liq in liquidations if liq['side'] == 'SELL'])
liq_short_volume = sum([liq['quantity'] for liq in liquidations if liq['side'] == 'BUY'])

liq_ratio = liq_long_volume / liq_short_volume if liq_short_volume > 0 else float('inf')

# Ratio > 1: mais longs liquidados (bearish)
# Ratio < 1: mais shorts liquidados (bullish)
```

---

## ⏱️ FREQUÊNCIA DE COLETA

**Consenso das IAs:**

| Tipo de Dado | Frequência | Justificativa |
|--------------|------------|---------------|
| **Klines** | 1m resample | Base temporal padrão |
| **AggTrades** | **1s aggregate** | "intervalos de 1s são razoáveis para manter precisão sem sobrecarregar" |
| **Book Depth** | **500ms snapshots** | "500ms para book depth são intervalos razoáveis" |
| **Funding Rate** | A cada 8h | Atualizado pela exchange a cada 8h |
| **Mark Price** | 1s | Segue aggtrades |
| **Open Interest** | 1m | Não muda com muita frequência |
| **Liquidations** | Real-time (tick) | Eventos importantes, capturar todos |

**GPT-Executor:**
> *"equilibrar a carga de dados e a capacidade de processamento. 1 segundo para AggTrades e 500ms para book depth são intervalos razoáveis para manter a precisão sem sobrecarregar o sistema."*

---

## 💾 STORAGE SCHEMA

**Consenso: Parquet particionado por HORA com compressão Snappy**

**GPT-Executor:**
> *"usar Parquet particionado por hora com compressão Snappy é recomendado para otimizar a leitura/escrita e armazenamento. Isso permite consultas rápidas e armazenamento eficiente, facilitando a análise histórica."*

### Schema Exato:

```python
# 1. AGGTRADES
aggtrades_schema = {
    'timestamp': 'int64',           # Unix timestamp em ms
    'trade_id': 'int64',
    'price': 'float64',
    'quantity': 'float64',
    'is_buyer_maker': 'bool',       # True = sell, False = buy
    'first_trade_id': 'int64',
    'last_trade_id': 'int64',
}
# Particionamento: /aggtrades/BTCUSDT/2024/11/08/hour=14/data.parquet
# Compressão: snappy

# 2. BOOK DEPTH
book_depth_schema = {
    'timestamp': 'int64',
    'last_update_id': 'int64',
    'bids': 'object',               # JSON array: [[price, qty], ...]
    'asks': 'object',               # JSON array: [[price, qty], ...]
    # Features calculadas:
    'bid_vol_5': 'float64',
    'ask_vol_5': 'float64',
    'bid_vol_10': 'float64',
    'ask_vol_10': 'float64',
    'bid_vol_20': 'float64',
    'ask_vol_20': 'float64',
    'imbalance_5': 'float64',
    'imbalance_10': 'float64',
    'imbalance_20': 'float64',
    'spread': 'float64',
    'mid_price': 'float64',
    'weighted_mid': 'float64',
}
# Particionamento: /book_depth/BTCUSDT/2024/11/08/hour=14/data.parquet

# 3. FEATURES AGREGADAS (por 1s, 1m, 5m, 15m)
features_schema = {
    'timestamp': 'int64',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    # AggTrades features
    'cvd': 'float64',
    'vwap': 'float64',
    'buy_volume': 'float64',
    'sell_volume': 'float64',
    'buy_pressure': 'float64',       # buy_vol / total_vol
    'trade_count': 'int64',
    'trade_intensity_1s': 'float64',
    'trade_intensity_5s': 'float64',
    'trade_intensity_10s': 'float64',
    'large_trade_count': 'int64',
    'large_trade_volume': 'float64',
    # Book depth features
    'imbalance_5': 'float64',
    'imbalance_10': 'float64',
    'imbalance_20': 'float64',
    'spread_avg': 'float64',
    'spread_std': 'float64',
    'bid_slope': 'float64',
    'ask_slope': 'float64',
    # Outros
    'funding_rate': 'float64',
    'mark_price': 'float64',
    'spot_premium_pct': 'float64',
    'open_interest': 'float64',
    'oi_change_pct': 'float64',
    'liq_long_vol': 'float64',
    'liq_short_vol': 'float64',
}
# Particionamento: /features/BTCUSDT/1s/2024/11/08/hour=14/data.parquet
#                  /features/BTCUSDT/1m/2024/11/08/hour=14/data.parquet
```

**Por que Snappy e não Gzip?**
- Snappy: mais rápido para ler/escrever (importante para backtests frequentes)
- Gzip: maior compressão mas mais lento
- Para dados de trading, velocidade > tamanho

---

## 🏗️ ARQUITETURA DE INTEGRAÇÃO ML/DL

**GPT-Strategist:**
> *"Implementar uma arquitetura baseada em microservices na nuvem pode permitir que diferentes componentes do sistema sejam escalados independentemente, dependendo da carga de dados e da complexidade dos cálculos. O uso de tecnologias de stream processing, como Apache Kafka e Apache Flink, pode ajudar a processar dados em tempo real de forma eficiente."*

### Pipeline Proposto:

```
┌─────────────────┐
│  Binance API    │
│  (WebSocket)    │
└────────┬────────┘
         │
         ├─→ AggTrades (tick-by-tick)
         ├─→ Book Depth (500ms)
         ├─→ Funding (8h)
         ├─→ Mark Price (1s)
         ├─→ Open Interest (1m)
         └─→ Liquidations (tick)
         │
         ▼
┌─────────────────────────┐
│  Stream Processor       │
│  (Apache Kafka/Flink)   │
│  - Aggregate to 1s      │
│  - Calculate features   │
│  - Join streams         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Store          │
│  (Parquet/hour/snappy)  │
│  - Raw data             │
│  - Computed features    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  ML/DL Models           │
│  - XGBoost              │
│  - RandomForest         │
│  - GRU/TCN (sequences)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Trading Executor       │
│  (HFT System)           │
└─────────────────────────┘
```

---

## 🎯 FEATURES FINAIS PARA MODELOS ML/DL

Com base no debate, estas são as **features finais** a incluir:

### **1. OHLCV Base** (7 features)
- open, high, low, close, volume
- returns, log_returns

### **2. AggTrades Features** (12 features)
- cvd (Cumulative Volume Delta)
- vwap (Volume-Weighted Average Price)
- buy_volume, sell_volume
- buy_pressure (ratio)
- trade_count
- trade_intensity_1s, trade_intensity_5s, trade_intensity_10s
- large_trade_count, large_trade_volume, large_trade_pct

### **3. Book Depth Features** (10 features)
- imbalance_5, imbalance_10, imbalance_20
- spread_avg, spread_std
- bid_slope, ask_slope
- weighted_mid_price_diff_bps
- bid_vol_ratio_5_20, ask_vol_ratio_5_20

### **4. Market Features** (7 features)
- funding_rate, funding_rate_delta
- spot_premium_pct
- open_interest, oi_change_pct
- liq_long_vol, liq_short_vol, liq_ratio

### **5. Technical Indicators** (existentes em selector21.py)
- RSI, MACD, ATR, Bollinger Bands, etc.

**TOTAL: ~40-50 features** por timeframe (1m, 5m, 15m)

---

## ✅ PRÓXIMOS PASSOS (Definidos pelas IAs)

1. ✅ **API Selection:** Binance Futures WebSocket API
2. ✅ **Implementar coletores:**
   - `collect_aggtrades.py` (WebSocket → 1s aggregate → Parquet)
   - `collect_book_depth.py` (WebSocket → 500ms snapshots → Parquet)
   - `collect_market_data.py` (REST API → funding, OI, mark → Parquet)
3. ✅ **Feature engineering:**
   - `compute_aggtrade_features.py`
   - `compute_book_features.py`
   - `merge_all_features.py`
4. ✅ **Integrar com selector21.py:**
   - Modificar `_make_ml_features_v2()` para incluir novas features
   - Atualizar Walk-Forward para usar dados completos
5. ✅ **Testar pipeline end-to-end**
6. ✅ **Executar Walk-Forward com dados completos**
7. ✅ **Deploy HFT system**

---

## 🏆 CONCLUSÃO DO DEBATE

**Consenso Final:**

> *"Para construir um sistema de trading robusto e eficiente, especialmente em um ambiente de alta frequência como o Binance Futures BTCUSDT, é **essencial integrar uma variedade de dados além dos klines (OHLCV)**. Enquanto os klines fornecem uma visão macro dos movimentos de preços, eles são **insuficientes para capturar a dinâmica do mercado** necessária para estratégias de alta frequência."*

**O usuário estava 100% correto:**
> "só os klines não servem pra nada"

**As IAs confirmam:**
- ✅ AggTrades são CRÍTICOS para fluxo de volume e momentum intrabar
- ✅ Book Depth é ESSENCIAL para liquidez e pressure
- ✅ Funding, OI, Liquidations são NECESSÁRIOS para contexto de mercado
- ✅ Frequência: 1s para trades, 500ms para book
- ✅ Storage: Parquet/hora/snappy

---

**Arquivos:**
- ✅ `DEBATE_MICROSTRUCTURE_DATA.md` (este arquivo)
- 🔄 `collect_aggtrades.py` (próximo)
- 🔄 `collect_book_depth.py` (próximo)
- 🔄 `compute_features.py` (próximo)

**Próximo passo: IMPLEMENTAR os coletores!** 🚀
