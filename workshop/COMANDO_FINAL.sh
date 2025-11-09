#!/bin/bash
#
# COMANDO FINAL - Consenso Claude + GPT (12 Rodadas)
#
# As IAs debateram e decidiram os parâmetros ideais.
# Este script executa o backtest com os critérios definidos.
#

set -e

echo "=============================================================================="
echo "🤖 BACKTEST - Consenso das IAs (Claude + GPT)"
echo "=============================================================================="
echo ""
echo "Parâmetros decididos pelas IAs:"
echo "  • Símbolo: BTCUSDT"
echo "  • Período: 2024-01-01 → 2024-06-01 (6 meses)"
echo "  • Dados: ./data_monthly (consolidados)"
echo "  • Validação: Walk-Forward (6m train / 1m test)"
echo ""
echo "Critérios de Aprovação:"
echo "  • Sharpe Ratio: ≥ 1.5 (ideal ≥ 2.0)"
echo "  • Win Rate: ≥ 55% (ideal ≥ 65%)"
echo "  • Max Drawdown: ≤ 20% (ideal ≤ 10%)"
echo "  • Profit Factor: ≥ 1.5 (ideal ≥ 2.0)"
echo ""
echo "=============================================================================="
echo ""

# Ativar venv se existir
if [ -d ".venv" ]; then
    echo "📦 Ativando venv..."
    source .venv/bin/activate
fi

# Rodar backtest
echo "🚀 Executando backtest..."
echo ""

python3 run_backtest_with_ias.py \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --data_dir ./data_monthly

BACKTEST_EXIT_CODE=$?

echo ""
echo "=============================================================================="

if [ $BACKTEST_EXIT_CODE -eq 0 ]; then
    echo "✅ BACKTEST COMPLETADO COM SUCESSO!"
    echo ""
    echo "Próximos passos:"
    echo ""
    echo "1. Ver resultado:"
    echo "   cat backtest_result_*.json | jq '.evaluation'"
    echo ""
    echo "2. Analisar aprendizados das IAs:"
    echo "   tail -20 claudex/LEARNING_LOG.jsonl | python3 -m json.tool"
    echo ""
    echo "3. Exportar para visual:"
    echo "   python3 export_to_visual.py \\"
    echo "     --backtest backtest_result_*.json \\"
    echo "     --output visual/data/run-$(date +%Y%m%d) \\"
    echo "     --run-id run-$(date +%Y%m%d)"
    echo ""
    echo "4. Ver replay visual:"
    echo "   cd visual/backend && python app.py"
    echo "   # http://localhost:8888/app"
    echo ""
else
    echo "❌ BACKTEST FALHOU (código: $BACKTEST_EXIT_CODE)"
    echo ""
    echo "Verifique os logs para mais detalhes."
fi

echo "=============================================================================="
