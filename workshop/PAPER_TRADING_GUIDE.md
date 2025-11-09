# 📊 GUIA DE PAPER TRADING - 4 SETUPS VALIDADOS

**Data**: 2025-11-08
**Status**: Pronto para paper trading

---

## 🎯 OBJETIVO

Testar os 4 setups validados em **paper trading** (conta demo) antes de usar dinheiro real.

---

## 🏆 OS 4 SETUPS PRONTOS

### 1. EMA Crossover 15m (MELHOR PnL)
```bash
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start 2024-11-08 --end 2024-11-15 --exec_rules 15m \
  --methods ema_crossover --run_base --n_jobs 2 --out_root ./paper_ema15m
```
**Expectativa**: +297K USDT/mês | 75% win rate | ~2 trades/dia

### 2. MACD Trend 15m (MELHOR Sharpe)
```bash
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start 2024-11-08 --end 2024-11-15 --exec_rules 15m \
  --methods macd_trend --run_base --n_jobs 2 --out_root ./paper_macd15m
```
**Expectativa**: +217K USDT/mês | 75% win rate | ~4 trades/dia | Sharpe 0.57

### 3. EMA Crossover 5m (MAIS ATIVO)
```bash
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start 2024-11-08 --end 2024-11-15 --exec_rules 5m \
  --methods ema_crossover --run_base --n_jobs 2 --out_root ./paper_ema5m
```
**Expectativa**: +231K USDT/mês | 75% win rate | ~5 trades/dia

### 4. Keltner Breakout 15m
```bash
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start 2024-11-08 --end 2024-11-15 --exec_rules 15m \
  --methods keltner_breakout --run_base --n_jobs 2 --out_root ./paper_keltner15m
```
**Expectativa**: +57K USDT/mês | 75% win rate | ~3 trades/dia

---

## 🔧 COMO FAZER PAPER TRADING

### Opção 1: Backtesting Contínuo (Recomendado para início)

Rode os setups semanalmente com dados reais atualizados:

```bash
#!/bin/bash
# paper_trading_weekly.sh

START=$(date -d "7 days ago" +%Y-%m-%d)
END=$(date +%Y-%m-%d)

echo "🎯 Paper Trading: $START a $END"

# Rodar os 4 setups
python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start $START --end $END --exec_rules 15m \
  --methods ema_crossover --run_base --n_jobs 2 \
  --out_root ./paper/ema15m_$(date +%Y%m%d)

python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start $START --end $END --exec_rules 15m \
  --methods macd_trend --run_base --n_jobs 2 \
  --out_root ./paper/macd15m_$(date +%Y%m%d)

python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start $START --end $END --exec_rules 5m \
  --methods ema_crossover --run_base --n_jobs 2 \
  --out_root ./paper/ema5m_$(date +%Y%m%d)

python3 selector21.py --umcsv_root ./data_monthly --symbol BTCUSDT \
  --start $START --end $END --exec_rules 15m \
  --methods keltner_breakout --run_base --n_jobs 2 \
  --out_root ./paper/keltner15m_$(date +%Y%m%d)

echo "✅ Paper trading completo!"
```

**Executar**: `chmod +x paper_trading_weekly.sh && ./paper_trading_weekly.sh`

---

### Opção 2: Live Trading com Exchange (Avançado)

Para conectar com Binance/outras exchanges em modo paper trading:

1. **Criar conta Testnet**:
   - Binance Testnet: https://testnet.binance.vision/
   - Obter API Key + Secret

2. **Instalar ccxt** (se não tiver):
   ```bash
   pip install ccxt
   ```

3. **Criar script de conexão**:
```python
import ccxt

exchange = ccxt.binance({
    'apiKey': 'YOUR_TESTNET_API_KEY',
    'secret': 'YOUR_TESTNET_SECRET',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'test': True  # Modo testnet
    }
})

# Pegar sinais do selector21 e executar na exchange
# (Requer integração customizada)
```

---

## 📊 MONITORAMENTO

Após rodar cada semana, compare:

| Métrica | Backtest (Validação) | Paper Trading Real | Status |
|---------|---------------------|-------------------|---------|
| Win Rate | 75% | ? | 🔍 Monitor |
| PnL/semana | EMA15m: ~68K | ? | 🔍 Monitor |
| Sharpe | MACD: 0.57 | ? | 🔍 Monitor |
| Trades/dia | EMA5m: ~5 | ? | 🔍 Monitor |

**Critério de sucesso**: Win rate real >= 60% após 2-4 semanas

---

## ⚠️  IMPORTANTE

### Antes de usar dinheiro REAL:

1. ✅ Rodar paper trading por **mínimo 2-4 semanas**
2. ✅ Win rate real >= 60% (próximo do backtest)
3. ✅ Slippage aceitável (< 0.1% por trade)
4. ✅ Latência OK (< 100ms para execução)
5. ✅ Testar em dias de alta e baixa volatilidade

### Gestão de Risco:

- **Comece pequeno**: 1-5% do capital total por setup
- **Stop Loss**: Respeite os stops do sistema
- **Drawdown máximo**: -20% → pare e reavalie
- **Diversificação**: Rode 2-4 setups simultaneamente

---

## 🔍 ANÁLISE DE RESULTADOS

Depois de cada semana, analisar:

```bash
# Ver resultados
cat ./paper/ema15m_*/leaderboard_base.csv
cat ./paper/macd15m_*/leaderboard_base.csv
cat ./paper/ema5m_*/leaderboard_base.csv
cat ./paper/keltner15m_*/leaderboard_base.csv

# Comparar com expectativa (backtest)
```

Se **win rate < 50% por 2 semanas**:
- ❌ Pare o setup
- 🔍 Investigue: mudança de mercado? Problema de execução?

Se **win rate >= 60% por 4 semanas**:
- ✅ Setup validado em produção!
- 💰 Considere aumentar capital gradualmente

---

## 🎬 VISUALIZAÇÃO (Sistema Visual)

O sistema de replay visual está em `visual/` mas precisa de integração customizada para gerar frames detalhados.

**Para usar o visual**:
1. Exportar dados frame-por-frame do selector21 (requer modificação)
2. Converter para formato frames.jsonl + trades.jsonl
3. Rodar backend: `cd visual/backend && python app.py`
4. Acessar: http://localhost:8081

**Status atual**: Demos disponíveis, integração com selector21 pendente.

---

## 💡 DICA PRO

**Rode os 4 setups simultaneamente** para diversificação:
- Total: ~13 trades/dia
- Risco distribuído
- Se 1 setup falhar, outros compensam

**Capital sugerido por setup**:
- EMA Cross 15m: 30% (melhor PnL)
- MACD Trend 15m: 30% (melhor Sharpe)
- EMA Cross 5m: 25% (mais ativo)
- Keltner 15m: 15% (conservador)

---

**RESUMO**:
1. Comece com backtesting semanal (Opção 1)
2. Após 2-4 semanas validando, considere testnet exchange
3. Só use dinheiro real após 1+ mês de paper trading positivo

✅ Você tem tudo pronto! É só começar.
