#!/bin/bash
#
# Monitor selector21 e dispara análise das IAs quando terminar
#

SELECTOR_PID=$(pgrep -f "python3 selector21.py")

if [ -z "$SELECTOR_PID" ]; then
    echo "❌ Selector21 não está rodando"
    exit 1
fi

echo "📊 Monitorando selector21 (PID: $SELECTOR_PID)..."
echo ""

# Aguarda término
while kill -0 $SELECTOR_PID 2>/dev/null; do
    # Mostra uso de recursos
    MEM=$(ps -p $SELECTOR_PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
    CPU=$(ps -p $SELECTOR_PID -o %cpu= | awk '{printf "%.1f%%", $1}')
    ELAPSED=$(ps -p $SELECTOR_PID -o etime= | xargs)

    echo -ne "\r⏳ Rodando... | CPU: $CPU | RAM: $MEM | Tempo: $ELAPSED    "
    sleep 10
done

echo ""
echo ""
echo "✅ SELECTOR21 COMPLETOU!"
echo ""

# Verifica se gerou resultados
if [ -d "./resultados" ]; then
    echo "📁 Resultados encontrados em ./resultados/"
    echo ""
    ls -lh ./resultados/*.csv ./resultados/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""

    # Conta trades/estratégias nos CSVs
    if [ -f "./resultados/wf_leaderboard_all.csv" ]; then
        STRATEGIES=$(tail -n +2 ./resultados/wf_leaderboard_all.csv 2>/dev/null | wc -l)
        echo "📊 Estratégias no leaderboard: $STRATEGIES"
    fi

    echo ""
    echo "🤖 Próximo passo: Análise detalhada pelas IAs"
    echo "   Execute: python3 analyze_results.py"
else
    echo "⚠️  Diretório ./resultados não encontrado"
fi

echo ""
