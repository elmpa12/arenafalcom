#!/bin/bash
# Paper Trading Semanal - 4 Setups Validados
# Rode este script toda semana para monitorar performance real

START=$(date -d "7 days ago" +%Y-%m-%d)
END=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "=============================================================================="
echo "🎯 PAPER TRADING SEMANAL - 4 SETUPS VALIDADOS"
echo "=============================================================================="
echo "📅 Período: $START a $END"
echo "⏰ Timestamp: $TIMESTAMP"
echo ""

# Criar diretório para resultados da semana
PAPER_DIR="./paper_trading/$TIMESTAMP"
mkdir -p "$PAPER_DIR"

echo "📁 Salvando em: $PAPER_DIR"
echo ""
echo "=============================================================================="
echo ""

# Setup 1: EMA Crossover 15m (MELHOR PnL)
echo "[1/4] 🎯 EMA Crossover 15m (Expectativa: +68K/semana, 75% win rate)..."
python3 selector21.py \
  --umcsv_root ./data_monthly \
  --symbol BTCUSDT \
  --start "$START" \
  --end "$END" \
  --exec_rules 15m \
  --methods ema_crossover \
  --run_base \
  --n_jobs 2 \
  --out_root "$PAPER_DIR/ema_cross_15m" \
  > "$PAPER_DIR/ema_cross_15m.log" 2>&1

if [ -f "$PAPER_DIR/ema_cross_15m/leaderboard_base.csv" ]; then
    echo "  ✅ Completo! Resultados em $PAPER_DIR/ema_cross_15m/"
else
    echo "  ❌ Falhou - veja $PAPER_DIR/ema_cross_15m.log"
fi
echo ""

# Setup 2: MACD Trend 15m (MELHOR Sharpe)
echo "[2/4] 🎯 MACD Trend 15m (Expectativa: +50K/semana, Sharpe 0.57)..."
python3 selector21.py \
  --umcsv_root ./data_monthly \
  --symbol BTCUSDT \
  --start "$START" \
  --end "$END" \
  --exec_rules 15m \
  --methods macd_trend \
  --run_base \
  --n_jobs 2 \
  --out_root "$PAPER_DIR/macd_trend_15m" \
  > "$PAPER_DIR/macd_trend_15m.log" 2>&1

if [ -f "$PAPER_DIR/macd_trend_15m/leaderboard_base.csv" ]; then
    echo "  ✅ Completo! Resultados em $PAPER_DIR/macd_trend_15m/"
else
    echo "  ❌ Falhou - veja $PAPER_DIR/macd_trend_15m.log"
fi
echo ""

# Setup 3: EMA Crossover 5m (MAIS ATIVO)
echo "[3/4] 🔥 EMA Crossover 5m (Expectativa: +53K/semana, ~35 trades)..."
python3 selector21.py \
  --umcsv_root ./data_monthly \
  --symbol BTCUSDT \
  --start "$START" \
  --end "$END" \
  --exec_rules 5m \
  --methods ema_crossover \
  --run_base \
  --n_jobs 2 \
  --out_root "$PAPER_DIR/ema_cross_5m" \
  > "$PAPER_DIR/ema_cross_5m.log" 2>&1

if [ -f "$PAPER_DIR/ema_cross_5m/leaderboard_base.csv" ]; then
    echo "  ✅ Completo! Resultados em $PAPER_DIR/ema_cross_5m/"
else
    echo "  ❌ Falhou - veja $PAPER_DIR/ema_cross_5m.log"
fi
echo ""

# Setup 4: Keltner Breakout 15m
echo "[4/4] ⚖️  Keltner Breakout 15m (Expectativa: +13K/semana, conservador)..."
python3 selector21.py \
  --umcsv_root ./data_monthly \
  --symbol BTCUSDT \
  --start "$START" \
  --end "$END" \
  --exec_rules 15m \
  --methods keltner_breakout \
  --run_base \
  --n_jobs 2 \
  --out_root "$PAPER_DIR/keltner_break_15m" \
  > "$PAPER_DIR/keltner_break_15m.log" 2>&1

if [ -f "$PAPER_DIR/keltner_break_15m/leaderboard_base.csv" ]; then
    echo "  ✅ Completo! Resultados em $PAPER_DIR/keltner_break_15m/"
else
    echo "  ❌ Falhou - veja $PAPER_DIR/keltner_break_15m.log"
fi
echo ""

echo "=============================================================================="
echo "📊 ANÁLISE RÁPIDA"
echo "=============================================================================="
echo ""

# Função para extrair métricas do CSV
extract_metrics() {
    CSV_FILE="$1"
    SETUP_NAME="$2"

    if [ -f "$CSV_FILE" ]; then
        # Ler primeira linha (melhor resultado)
        METRICS=$(tail -n 1 "$CSV_FILE")

        # Extrair campos (assumindo ordem: hit, payoff, total_pnl, sharpe, maxdd, n_trades)
        HIT=$(echo "$METRICS" | cut -d',' -f1 | tail -c +1)
        PNL=$(echo "$METRICS" | cut -d',' -f3 | tail -c +1)
        SHARPE=$(echo "$METRICS" | cut -d',' -f4 | tail -c +1)
        TRADES=$(echo "$METRICS" | cut -d',' -f6 | tail -c +1)

        printf "%-25s PnL: %10s | Sharpe: %6s | Trades: %3s\n" \
            "$SETUP_NAME" "$PNL" "$SHARPE" "$TRADES"
    else
        printf "%-25s ❌ Sem resultados\n" "$SETUP_NAME"
    fi
}

extract_metrics "$PAPER_DIR/ema_cross_15m/leaderboard_base.csv" "EMA Cross 15m"
extract_metrics "$PAPER_DIR/macd_trend_15m/leaderboard_base.csv" "MACD Trend 15m"
extract_metrics "$PAPER_DIR/ema_cross_5m/leaderboard_base.csv" "EMA Cross 5m"
extract_metrics "$PAPER_DIR/keltner_break_15m/leaderboard_base.csv" "Keltner 15m"

echo ""
echo "=============================================================================="
echo "✅ Paper Trading Semanal Completo!"
echo "=============================================================================="
echo ""
echo "📁 Resultados salvos em: $PAPER_DIR"
echo ""
echo "💡 PRÓXIMOS PASSOS:"
echo "   1. Compare PnL/Sharpe com expectativa (backtest)"
echo "   2. Se win rate >= 60% → Bom sinal!"
echo "   3. Se win rate < 50% por 2 semanas → Investigar"
echo "   4. Rode este script toda semana para monitorar"
echo ""
echo "📊 Para ver CSVs detalhados:"
echo "   cat $PAPER_DIR/*/leaderboard_base.csv"
echo ""
