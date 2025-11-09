#!/bin/bash
#
# SELECTOR21 - COMANDO FINAL LIMPO
# Gerado pelas IAs após estudo completo de ~209 parâmetros
#
# Versão: Corrigida (removidos parâmetros inventados, mantidos apenas os reais)
#

set -e

echo "=============================================================================="
echo "🤖 SELECTOR21 - Comando Consensuado pelas IAs"
echo "=============================================================================="
echo ""
echo "Baseado em análise COMPLETA de ~209 parâmetros reais do selector21.py"
echo ""
echo "Categorias cobertas:"
echo "  ✓ Data Loading"
echo "  ✓ Execution Rules (14 métodos)"
echo "  ✓ Combos (AND, MAJ, SEQ)"
echo "  ✓ Risk Management"
echo "  ✓ Stops & Take-Profits (ATR + Hard)"
echo "  ✓ Walk-Forward Optimization"
echo "  ✓ Machine Learning (XGBoost, RF, LogReg)"
echo "  ✓ Filters (ATR-Z, VHF, CVD)"
echo "  ✓ Performance (parallelization)"
echo "  ✓ Output & Metrics"
echo ""
echo "=============================================================================="
echo ""

# Ativar venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# COMANDO COMPLETO (parâmetros REAIS do selector21.py)
python3 selector21.py \
  --umcsv_root ./data_monthly \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --interval auto \
  --exec_rules '1m,5m,15m' \
  --methods 'trend_breakout,keltner_breakout,rsi_reversion,ema_crossover,macd_trend,vwap_trend,boll_breakout,orb_breakout,orr_reversal,ema_pullback,donchian_breakout,vwap_poc_reject,ob_imbalance_break,cvd_divergence_reversal' \
  --run_base \
  --run_combos \
  --combo_ops 'AND,MAJ,SEQ' \
  --combo_cap 400 \
  --combo_window 2 \
  --combo_min_votes 2 \
  --contracts 1.0 \
  --contract_value 100.0 \
  --fee_perc 0.0004 \
  --slippage 2 \
  --tick_size 0.01 \
  --max_hold '480,240,120' \
  --use_atr_stop \
  --atr_stop_len 14 \
  --atr_stop_mult 2.0 \
  --trailing \
  --timeout_mode both \
  --atr_timeout_len '14,14,14' \
  --atr_timeout_mult '8,10,12' \
  --use_atr_tp \
  --atr_tp_len '14,14,14' \
  --atr_tp_mult '3,3.5,4' \
  --hard_stop_usd '300,600,1000' \
  --hard_tp_usd '2000,2500,3000' \
  --use_candle_stop \
  --candle_stop_lookback 2 \
  --round_to_tick \
  --walkforward \
  --wf_train_months 6 \
  --wf_val_months 1 \
  --wf_step_months 1 \
  --wf_grid_mode medium \
  --wf_top_n 5 \
  --wf_expand \
  --run_ml \
  --ml_model_kind auto \
  --ml_horizon 5 \
  --ml_ret_thr 0.005 \
  --ml_lags 20 \
  --ml_opt_thr \
  --ml_thr_grid '0.50,0.70,0.02' \
  --ml_neutral_band 0.05 \
  --ml_add_base_feats \
  --ml_add_combo_feats \
  --ml_combo_top_n 10 \
  --ml_combo_ops 'AND,MAJ' \
  --ml_seed 42 \
  --ml_calibrate platt \
  --ml_recency_mode exp \
  --ml_recency_half_life 1000 \
  --atr_z_min 0.5 \
  --vhf_min 0.3 \
  --n_jobs -1 \
  --par_backend thread \
  --min_trades 10 \
  --min_hit 0.55 \
  --min_pnl 100 \
  --min_sharpe 1.0 \
  --max_dd 0.25 \
  --out_root ./resultados \
  --out_wf_report ./resultados/wf_report.csv \
  --out_wf_ml ./resultados/wf_ml.csv \
  --ml_save_dir ./resultados/models \
  --print_top10

EXIT_CODE=$?

echo ""
echo "=============================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SELECTOR21 COMPLETADO!"
    echo ""
    echo "Resultados em: ./resultados/"
    echo ""
    echo "Próximos passos:"
    echo "  1. Ver top 10: cat resultados/wf_report.csv | head -11"
    echo "  2. Ver ML results: cat resultados/wf_ml.csv | head -11"
    echo "  3. Exportar para visual:"
    echo "     python3 export_to_visual.py --output visual/data/selector-run"
    echo ""
else
    echo "❌ SELECTOR21 FALHOU (código: $EXIT_CODE)"
    echo ""
fi

echo "=============================================================================="
echo ""
echo "📝 APRENDIZADO DAS IAs SALVO EM:"
echo "  • selector_analysis_claude.txt"
echo "  • selector_analysis_gpt.txt"
echo "  • Este comando (SELECTOR_FINAL_LIMPO.sh)"
echo ""
echo "Na próxima iteração, as IAs vão analisar esses resultados e propor melhorias!"
echo "=============================================================================="
