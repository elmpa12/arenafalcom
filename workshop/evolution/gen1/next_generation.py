# GERAÇÃO 2 - Auto-gerada pelas IAs

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