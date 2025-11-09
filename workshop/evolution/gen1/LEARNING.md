# APRENDIZADOS - GERAÇÃO 1

## Análise Claude

Análise dos Resultados:

1. O que funcionou?
   - Os métodos MACD Trend e EMA Crossover tiveram os melhores resultados, com Sharpe Ratio positivo, lucro e número razoável de trades.
   - Estes métodos parecem funcionar bem em diferentes timeframes, como visto nos testes 4, 2 e 8.
   - Os períodos de teste também variam, indicando que estes métodos podem ser robustos em diferentes condições de mercado.
   - Esses métodos de tendência e cruzamento de médias móveis provavelmente funcionaram bem por capturarem os movimentos de mercado de maneira eficaz.

2. O que não funcionou?
   - Os métodos RSI Reversion e OB Imbalance Break tiveram resultados negativos, com Sharpe Ratio negativo e prejuízo.
   - Sharpe Ratio negativo indica que o risco ajustado do sistema não é compensado pelos retornos, ou seja, o sistema perde dinheiro de forma inconsistente.

3. Hipóteses:
   - Possível overfitting: os métodos que tiveram ótimos resultados em alguns testes (como EMA Crossover) apresentaram péssimo desempenho em outros, indicando falta de robustez.
   - Alguns períodos de teste podem ter sido inadequados para certos métodos, resultando em desempenho fraco.
   - Alguns métodos podem não ser adequados para as características deste mercado específico.
   - Possível falta de stops adequados para limitar as perdas.

4. Aprendizados Chave:
   - É necessário analisar cuidadosamente o desempenho dos métodos em diferentes períodos para avaliar a robustez.
   - Métodos baseados em tendência e cruzamento de médias móveis parecem funcionar melhor neste contexto.
   - É importante investigar as razões pelas quais alguns métodos falharam, como overfitting, inadequação aos períodos de teste ou às características do mercado.
   - A implementação de stops adequados pode ser crucial para melhorar a performance dos sistemas.
   - É preciso um equilíbrio entre a complexidade dos métodos e sua capacidade de generalização.

## Propostas GPT

Para a Geração 2, proporei melhorias concretas baseadas na análise dos resultados obtidos na Geração 1, com enfoque em ajustes de parâmetros, novos métodos e combinações, períodos alternativos, e aprimoramento do gerenciamento de risco. Vamos detalhar cada um dos pontos:

### 1. **AJUSTES DE PARÂMETROS**

**Quais parâmetros devemos testar?**
- *MACD*: Ajustar rapidamente os parâmetros padrão (12, 26, 9). Testar combinações como (8, 24, 9) e (5, 35, 10) para avaliar a resposta a diferentes sensibilidades.
- *EMA Crossover*: Alterar os períodos das EMAs utilizadas. Testar pares como (9, 21) e (5, 50) para melhor captura de tendências a curto e longo prazo.
- *RSI*: Ajustar o período padrão de 14 para valores como 5 e 20 para verificar a eficácia em diferentes condições de mercado.

**Por que esses valores?**
- Testar diferentes sensibilidades permite captar reações a movimentos de mercado rápidos e lentos, adequando-se a diferentes volatidades e tendências.

### 2. **NOVOS MÉTODOS/COMBOS**

**Quais métodos testar?**
- *Combinar MACD com RSI*: Utilizar o MACD para detectar tendência e o RSI para identificar condições de sobrecompra/sobrevenda.
- *ATR como filtro em cruzamento de médias*: Utilizar o Average True Range para determinar volatilidade e filtrar entradas falsas.

**Combos promissores**
- *EMA + MACD*: Usar crossover EMA para entrada e MACD histograma para confirmação.
- *RSI + Bollinger Bands*: Entrar quando o RSI está em sobrecompra/sobrevenda e toca as bandas.

### 3. **PERÍODOS ALTERNATIVOS**

**Testar outros períodos?**
- Testar períodos menores como 5 minutos para scalping e maiores como 1 dia para swing trading.

**Por que esses períodos?**
- Períodos menores podem capturar ganhos em mercados mais voláteis e liquidar rapidamente, enquanto períodos maiores podem se manter em tendências baseadas em movimentos macroeconômicos.

### 4. **FILTROS/STOPS**

**Como melhorar risk management?**
- Implementar stops dinâmicos com base na volatilidade (ATR) e ajustar de acordo com a média histórica do ativo.

**Stops dinâmicos?**
- Usar ATR para definir stops que se ajustam à volatilidade atual do mercado, oferecendo proteção eficiente sem limitar desnecessariamente o potencial de lucro.

### 5. **TOP 5 EXPERIMENTOS para Geração 2**

1. **MACD + RSI Estratégia Mista**
   - Objetivo: Capturar tendência e reversão.
   - Config: MACD (8, 24, 9), RSI (14).
   - Hipótese: Melhorar o timing de entradas com duplo critério.

2. **Cruzamento de EMA com Filtro ATR**
   - Objetivo: Filtrar trades falsos em movimentos laterais.
   - Config: EMA (9, 21), ATR Stop.
   - Hipótese: Reduzir drawdowns em mercados laterais.

3. **ESCALPAGEM com EMAs Rápidas**
   - Objetivo: Capturar movimentos rápidos.
   - Config: Cross EMA (5, 13), Período: 5min.
   - Hipótese: Maior quantidade de trades lucrativos em escalas menores.

4. **Donchian Breakout em Período Diário**
   - Objetivo: Capturar grandes tendências.
   - Config: Donchian Channel (20), Período: 1 dia.
   - Hipótese: Maior lucratividade em tendências prolongadas.

5. **RSI e Bandas de Bollinger**
   - Objetivo: Aproveitar movimentos de reversão dentro de bandas.
   - Config: RSI (14), Bollinger Bands (20, 2).
   - Hipótese: Melhorar as entradas com confirmação de reversão.

Esses experimentos visam refinar técnicas existentes e explorar novas combinações, equilibrando risco e retornos em diferentes condições de mercado.

## Próxima Geração

Para a Geração 2, vamos criar uma série de testes baseados nas hipóteses e aprendizado que você apresentou. Vamos focar em ajustar os parâmetros dos métodos existentes, introduzir combinações de métodos e melhorar o gerenciamento de risco. Aqui estão os novos testes em formato Python:

```python
TESTS_GEN2 = [
    {
        "name": "gen2_test1_macd_adjust",
        "desc": "Testar combinação de parâmetros no MACD para maior sensibilidade",
        "hypothesis": "Parâmetros ajustados do MACD capturam movimentos de tendência mais rapidamente",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2022-01-01",
            "--end", "2023-01-01",
            "--strategy", "MACD",
            "--macd_fast", "8",
            "--macd_slow", "24",
            "--macd_signal", "9",
            "--out_root", "./resultados/gen2/test1"
        ]
    },
    {
        "name": "gen2_test2_ema_crossover_9_21",
        "desc": "Verificar eficácia do EMA crossover usando 9 e 21 períodos",
        "hypothesis": "Combinação 9/21 EMAs captura tendências de curto e médio prazo de forma eficaz",
        "args": [
            "--umcsv_root", "./data_weekly",
            "--symbol", "BTCUSDT",
            "--start", "2022-01-01",
            "--end", "2023-01-01",
            "--strategy", "EMA_CROSSOVER",
            "--ema_fast", "9",
            "--ema_slow", "21",
            "--out_root", "./resultados/gen2/test2"
        ]
    },
    {
        "name": "gen2_test3_rsi_macd",
        "desc": "Combinar o MACD com o RSI para sinais de entrada e saída",
        "hypothesis": "MACD confirma a tendência enquanto RSI identifica condições extremas para entradas e saídas",
        "args": [
            "--umcsv_root", "./data_daily",
            "--symbol", "BTCUSDT",
            "--start", "2022-01-01",
            "--end", "2023-01-01",
            "--strategy", "MACD_RSI_COMBO",
            "--macd_fast", "12",
            "--macd_slow", "26",
            "--macd_signal", "9",
            "--rsi_period", "14",
            "--rsi_overbought", "70",
            "--rsi_oversold", "30",
            "--out_root", "./resultados/gen2/test3"
        ]
    },
    {
        "name": "gen2_test4_atr_filter_on_ema",
        "desc": "Aplicar ATR como filtro em cruzamento de médias móveis",
        "hypothesis": "ATR ajuda a evitar sinais falsos ao considerar a volatilidade de mercado nas entradas",
        "args": [
            "--umcsv_root", "./data_daily",
            "--symbol", "BTCUSDT",
            "--start", "2022-01-01",
            "--end", "2023-01-01",
            "--strategy", "ATR_FILTER_EMA",
            "--ema_fast", "9",
            "--ema_slow", "50",
            "--atr_period", "14",
            "--out_root", "./resultados/gen2/test4"
        ]
    },
    {
        "name": "gen2_test5_rsi_bollinger",
        "desc": "Entrada baseada em RSI e Bollinger Bands",
        "hypothesis": "Combinação de RSI com Bollinger Bands melhora precisão ao capturar movimentos extremos",
        "args": [
            "--umcsv_root", "./data_daily",
            "--symbol", "BTCUSDT",
            "--start", "2022-01-01",
            "--end", "2023-01-01",
            "--strategy", "RSI_BBANDS",
            "--rsi_period", "14",
            "--bollinger_period", "20",
            "--bollinger_dev", "2",
            "--out_root", "./resultados/gen2/test5"
        ]
    },
    {
        "name": "gen2_test6_ema_macd_confirmation",
        "desc": "Usar cruzamento de EMAs com confirmação de MACD histograma",
        "hypothesis": "MACD histograma oferece confirmação adicional ao sinal de cruzamento EMA",
        "args": [
            "--umcsv_root", "./data_weekly",
            "--symbol", "BTCUSDT",
            "--start", "2022-06-01",
            "--end", "2023-06-01",
            "--strategy", "EMA_MACD_CONFIRM",
            "--ema_fast", "5",
            "--ema_slow", "35",
            "--macd_histogram_threshold", "0.5",
            "--out_root", "./resultados/gen2/test6"
        ]
    }
]
```

Esses testes variam os parâmetros e combinam diferentes características dos métodos para buscar melhorar a robustez e desempenho em diversas condições de mercado. Cada teste é designado para capturar aspectos específicos aprendidos durante a Geração 1 e poderá ajudar a refinar a estratégia geral.