# GERAÇÃO 2 → GERAÇÃO 3: APRENDIZADOS E HIPÓTESES

## Análise de 30 Testes Ultra-Rápidos

**Melhor Método (menor perda)**: orr_reversal
**Melhor Período**: Semana 4 (Jan)

## Hipóteses para Geração 3

### HIPÓTESE 1: Método orr_reversal tem melhor potencial
   → Gen 2 mostrou que orr_reversal teve menor perda média
   → Testar orr_reversal com diferentes parâmetros de risco

### HIPÓTESE 2: Período Semana 4 (Jan) é mais favorável
   → Gen 2 mostrou melhor desempenho relativo neste período
   → Focar testes em períodos similares (volatilidade/condições de mercado)

### HIPÓTESE 3: Hit Rate vs Payoff tradeoff
   → Alguns métodos têm hit >40% mas perdem dinheiro
   → Problema: payoff ratio insuficiente ou stops mal calibrados
   → Testar diferentes configurações de stop/target

### HIPÓTESE 4: Timeframe 1m pode ser ruidoso demais
   → Todos os testes usaram 1m
   → Testar 5m e 15m para reduzir ruído

### HIPÓTESE 5: Período de teste muito curto
   → 1 semana = 50-300 trades
   → Aumentar para 2-4 semanas para melhor amostra estatística

## Próximos Passos

```bash
# Rodar Geração 3 (30 testes, ~5-15s cada, 20 paralelos)
python3 run_from_config.py gen3_tests_config.json --parallel 20
```
