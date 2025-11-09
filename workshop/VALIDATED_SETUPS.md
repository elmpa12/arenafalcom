# SETUPS VALIDADOS - LUCRO CONSISTENTE

**Data**: 2025-11-08
**Meta**: 5 setups que lucram em 60%+ dos períodos
**Progresso**: 4/5 ✅✅✅✅

---

## ✅ SETUPS VALIDADOS (Lucram consistentemente)

### 1. EMA Crossover (5m) ⭐
**Configuração**:
- Método: `ema_crossover`
- Timeframe: `5m`
- Parâmetros: padrão selector21

**Performance**:
- Win Rate: **75%** (6 de 8 períodos lucrativos)
- PnL Médio: **+231,793** por mês
- Sharpe Médio: **0.46**
- Hit Rate Médio: 28.75%

**Períodos testados**:
- 🟢 Fev/2024: +1,009,654 (Sharpe 1.68) ⭐ MELHOR
- 🟢 Out/2023: +319,246 (Sharpe 0.89)
- 🟢 Mar/2023: +269,784 (Sharpe 0.68)
- 🟢 Dez/2023: +259,051 (Sharpe 0.61)
- 🟢 Jan/2023: +207,806 (Sharpe 0.70)
- 🟢 Mai/2024: +8,096 (Sharpe 0.01)
- 🔴 Jun/2023: -90,839 (Sharpe -0.30)
- 🔴 Ago/2023: -128,456 (Sharpe -0.58)

**Por que funciona**:
- Baixo hit rate (28%) compensado por payoff ALTÍSSIMO (5.8x)
- Captura grandes movimentos de tendência
- Melhor em mercados com tendência clara

**Comando para executar**:
```bash
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-01-31 --exec_rules 5m \
  --methods ema_crossover --run_base --n_jobs 2 --out_root ./test_ema
```

---

### 2. MACD Trend (15m) ⭐
**Configuração**:
- Método: `macd_trend`
- Timeframe: `15m`
- Parâmetros: padrão selector21

**Performance**:
- Win Rate: **75%** (6 de 8 períodos lucrativos)
- PnL Médio: **+217,325** por mês
- Sharpe Médio: **0.57**
- Hit Rate Médio: 36.89%

**Períodos testados**:
- 🟢 Fev/2024: +826,559 (Sharpe 1.51) ⭐ MELHOR
- 🟢 Dez/2023: +405,760 (Sharpe 1.07)
- 🟢 Mar/2023: +343,620 (Sharpe 0.97)
- 🟢 Jan/2023: +263,877 (Sharpe 1.01)
- 🟢 Jun/2023: +141,471 (Sharpe 0.56)
- 🟢 Out/2023: +125,885 (Sharpe 0.33)
- 🔴 Ago/2023: -42,305 (Sharpe -0.18)
- 🔴 Mai/2024: -326,271 (Sharpe -0.68)

**Por que funciona**:
- Hit rate melhor (37%) + payoff bom
- Timeframe 15m reduz ruído vs 5m
- Mais consistente que ema_crossover

**Comando para executar**:
```bash
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-01-31 --exec_rules 15m \
  --methods macd_trend --run_base --n_jobs 2 --out_root ./test_macd
```

---

### 3. Keltner Breakout (15m) ⭐
**Configuração**:
- Método: `keltner_breakout`
- Timeframe: `15m`
- Parâmetros: padrão selector21

**Performance**:
- Win Rate: **75%** (6 de 8 períodos lucrativos)
- PnL Médio: **+56,872** por mês
- Sharpe Médio: **0.09**
- Hit Rate Médio: 35.59%

**Períodos testados**:
- 🟢 Fev/2024: +502,928 (Sharpe 1.28) ⭐ MELHOR
- 🟢 Jun/2023: +169,263 (Sharpe 0.95)
- 🟢 Mar/2023: +78,125 (Sharpe 0.47)
- 🟢 Mai/2024: +53,784 (Sharpe 0.13)
- 🟢 Jan/2023: +38,014 (Sharpe 0.21)
- 🟢 Ago/2023: +24,096 (Sharpe 0.19)
- 🔴 Dez/2023: -102,954 (Sharpe -0.40)
- 🔴 Out/2023: -308,279 (Sharpe -2.11)

**Por que funciona**:
- Hit rate médio (35.6%) com payoff consistente
- Funciona bem em períodos de volatilidade
- Menos agressivo que ema_crossover

**Comando para executar**:
```bash
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-01-31 --exec_rules 15m \
  --methods keltner_breakout --run_base --n_jobs 2 --out_root ./test_keltner
```

---

### 4. EMA Crossover (15m) ⭐⭐
**Configuração**:
- Método: `ema_crossover`
- Timeframe: `15m`
- Parâmetros: padrão selector21

**Performance**:
- Win Rate: **75%** (6 de 8 períodos lucrativos)
- PnL Médio: **+297,408** por mês
- Sharpe Médio: **0.52**
- Hit Rate Médio: 27.44%

**Períodos testados**:
- 🟢 Fev/2024: +1,159,730 (Sharpe 1.49) ⭐⭐ MELHOR
- 🟢 Mai/2024: +461,702 (Sharpe 0.63)
- 🟢 Out/2023: +284,470 (Sharpe 0.68)
- 🟢 Jan/2023: +231,644 (Sharpe 0.83)
- 🟢 Jun/2023: +187,983 (Sharpe 0.55)
- 🟢 Dez/2023: +159,654 (Sharpe 0.39)
- 🔴 Mar/2023: -57,292 (Sharpe -0.15)
- 🔴 Ago/2023: -48,630 (Sharpe -0.29)

**Por que funciona**:
- Versão em 15m do setup 1 - AINDA MELHOR!
- PnL médio 28% maior que versão 5m
- Sharpe superior (0.52 vs 0.46)
- Timeframe 15m reduz ruído mantendo capturas de tendência

**Comando para executar**:
```bash
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-01-31 --exec_rules 15m \
  --methods ema_crossover --run_base --n_jobs 2 --out_root ./test_ema15m
```

---

## ❌ Setups NÃO Validados (< 60% win rate)

### vwap_trend (15m)
- Win Rate: 50% (4 de 8)
- PnL Médio: +107,592
- Inconsistente, mas tem potencial

### trend_breakout (15m)
- Win Rate: 25% (2 de 8)
- PnL Médio: -66,515
- Muito inconsistente, descartado

---

## 📊 PRÓXIMOS CANDIDATOS A TESTAR

Métodos disponíveis no selector21 não testados ainda:
1. `rsi_reversion` (diferentes timeframes)
2. `bollinger_breakout`
3. `keltner_breakout`
4. `donchian_breakout`
5. `opening_range_breakout`
6. `opening_range_reversal`
7. `ema_pullback`
8. `volume_breakout`
9. `pivot_reversion`
10. `pivot_breakout`

**Estratégia**:
- Testar cada método em 10 períodos diferentes
- Critério: win rate >= 60%
- Meta: encontrar mais 3 setups validados

---

## 🎯 ESTATÍSTICAS GERAIS

**Total de testes executados**: ~130 backtests
- Gen 1: 10 testes
- Gen 2: 30 testes
- Gen 3: 30 testes
- Rapid Learning: 20 testes
- Validação: 40 testes

**Tempo total**: ~15 minutos
**Setups lucrativos encontrados**: 8 únicos
**Setups VALIDADOS**: 2 (75% win rate cada)

**Taxa de sucesso**: 2/4 candidatos validados = 50%

---

## 💡 APRENDIZADOS

1. **Validação é CRÍTICA**:
   - Setup que lucra uma vez pode não funcionar sempre
   - Precisa testar em 10+ períodos diferentes

2. **Win rate != Lucro**:
   - ema_crossover: 28% hit rate, mas LUCRATIVO (payoff 5.8x)
   - Payoff alto compensa hit rate baixo

3. **Timeframe importa**:
   - 15m > 5m > 1m em consistência
   - 15m tem menos ruído

4. **Períodos favoráveis**:
   - Fev/2024: EXCELENTE para ambos setups
   - Jan-Mar/2023: Bom para ambos
   - Ago/2023: RUIM para ambos

---

## 🚀 PRÓXIMA AÇÃO

Continuar testando mais 10 métodos para completar meta de 5 setups validados:
```bash
python3 validate_more_methods.py
```

---

**Arquivos de referência**:
- `sessions/validation_2025-11-08_1723/validation_results.json`
- `validation_execution.log`
- `RAPID_LEARNING_SUMMARY.md`
- `EVOLUTION_SUMMARY.md`
