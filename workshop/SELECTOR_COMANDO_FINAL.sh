#!/bin/bash
# Comando gerado pelas IAs após análise COMPLETA do selector21.py

Com base nas análises e recomendações fornecidas por Claude e GPT, aqui está um comando completo para o `selector21.py`, incorporando todos os parâmetros essenciais das 10 categorias discutidas:

```bash
python3 selector21.py \
  --symbol BTCUSDT \
  --start 2019-01-01 \
  --end 2023-09-30 \
  --data_dir /caminho/para/dados \
  --data_glob 'BTCUSDT_*_1m.parquet;BTCUSDT_*_5m.parquet;BTCUSDT_*_15m.parquet' \
  --interval auto \
  --smoke_months 0 \
  --max_rows_loader 0 \
  --exec_rules 'rsi,macd,atr,ema,bb,stoch' \
  --methods 'all' \
  --run_base \
  --run_combos \
  --co 'atr_stop' \
  --entry_rules 'rsi_cross,macd_cross' \
  --exit_rules 'rsi_exit,macd_exit' \
  --risk_management 'fixed_risk' \
  --position_sizing 'percent_of_equity' \
  --reporting 'detailed' \
  --optimization 'grid,random' \
  --validation 'walk_forward' \
  --seed 42
```

### Explicações Adicionais:

1. **Data Loading**:
   - Início em 2019 e fim em 2023-09-30 para capturar dados antes, durante e após condições de mercado importantes.
   - `auto` permite que o script determine os melhores intervalos com base nos dados disponíveis.

2. **Execution Rules**:
   - Uso de uma combinação diversa de indicadores para uma análise abrangente e adaptabilidade a diferentes condições do mercado.

3. **Combos e Entry/Exit Rules**:
   - Parâmetros como `atr_stop`, `rsi_cross`, e `macd_cross` são exemplos para sinais de entrada e saída, maximizando oportunidades.

4. **Risk Management e Position Sizing**:
   - Parâmetros importantes para simular estratégias realistas, como `fixed_risk` e `percent_of_equity`.

5. **Reporting**:
   - Relatórios detalhados são essenciais para avaliar a performance de maneira compreensiva.

6. **Optimization & Validation**:
   - Uso de `grid` e `random` para otimização, e `walk_forward` para validação, que são práticas comuns para evitar overfitting e garantir robustez dos resultados.

7. **Seed**:
   - Definição de uma seed (`42`) para reprodutibilidade de resultados.

Esses parâmetros foram escolhidos com o objetivo de fornecer uma configuração de backtest sólida e abrangente, considerando a adaptação a uma variedade de condições de mercado e maximizando a eficiência da estratégia de trading avaliada.