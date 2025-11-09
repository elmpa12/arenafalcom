#!/bin/bash
################################################################################
# DOWNLOAD 2 ANOS - TODOS OS DADOS - Para rodar no root@lab
################################################################################
#
# Execute este script no seu servidor root@lab:
#   chmod +x DOWNLOAD_2_ANOS_COMPLETO.sh
#   ./DOWNLOAD_2_ANOS_COMPLETO.sh
#
# Ou rode os comandos individuais abaixo
################################################################################

echo "🚀 DOWNLOAD 2 ANOS - BINANCE PUBLIC DATA (COMPLETO)"
echo "=========================================="
echo ""
echo "📊 Dados a baixar (3 símbolos: BTC, ETH, SOL):"
echo ""
echo "  Via Binance Vision (rápido):"
echo "    • AggTrades (2022-11-08 → 2024-11-08)"
echo "    • Klines: 1m, 5m, 15m, 1h, 4h, 1d"
echo ""
echo "  Via API REST (rate limited):"
echo "    • Funding Rate (a cada 8h)"
echo "    • Open Interest (a cada 5min)"
echo "    • Long/Short Ratio (a cada 5min)"
echo ""
echo "⏱️  Tempo estimado:"
echo "    - Binance Vision: 45-60 minutos"
echo "    - API REST: 15-30 minutos"
echo "    - Total: ~60-90 minutos"
echo ""
echo "💾 Tamanho final: ~35-50GB"
echo ""

# Verificar se está no diretório correto
if [ ! -f "download_binance_public_data.py" ]; then
    echo "❌ ERRO: download_binance_public_data.py não encontrado!"
    echo "Execute este script dentro de /opt/botscalpv3/"
    exit 1
fi

# Verificar dependências
echo "📦 Verificando dependências..."
python3 -c "import pandas, pyarrow, requests, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Instalando dependências..."
    pip3 install pandas pyarrow requests tqdm --quiet
fi

# Testar se API da Binance está acessível
echo ""
echo "🔍 Testando acesso à API da Binance..."
API_TEST=$(python3 -c "
import requests
try:
    resp = requests.get('https://fapi.binance.com/fapi/v1/ping', timeout=5)
    print('OK' if resp.status_code == 200 else 'BLOCKED')
except:
    print('ERROR')
" 2>/dev/null)

if [ "$API_TEST" = "BLOCKED" ]; then
    echo "⚠️  API da Binance está BLOQUEADA na sua região!"
    echo "   Dados via API (funding, OI, L/S ratio) NÃO serão baixados."
    echo "   Apenas Binance Vision (aggTrades, klines) será usado."
    echo ""
    read -p "Continuar apenas com Binance Vision? (s/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Download cancelado."
        exit 1
    fi
    SKIP_API=true
elif [ "$API_TEST" = "OK" ]; then
    echo "✅ API acessível! Todos os dados serão baixados."
    SKIP_API=false
else
    echo "⚠️  Erro ao testar API. Prosseguindo apenas com Binance Vision."
    SKIP_API=true
fi

echo ""
echo "✅ Iniciando downloads em paralelo..."
echo ""

# Símbolos para baixar
SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT")
PIDS=()

# ============================================================================
# 1. AggTrades (todos os símbolos)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 [1/5] AggTrades para ${SYMBOLS[@]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for SYMBOL in "${SYMBOLS[@]}"; do
    echo "   🚀 Iniciando $SYMBOL aggTrades..."
    nohup python3 download_binance_public_data.py \
        --data-type aggTrades \
        --symbol $SYMBOL \
        --market futures \
        --start-date 2022-11-08 \
        --end-date 2024-11-08 \
        --output-dir ./data \
        > /tmp/download_aggtrades_${SYMBOL}.log 2>&1 &

    PIDS+=($!)
    echo "      ✅ PID: $!"
done

# ============================================================================
# 2. Klines (todos os timeframes + todos os símbolos)
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 [2/5] Klines (1m, 5m, 15m, 1h, 4h, 1d) para ${SYMBOLS[@]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for SYMBOL in "${SYMBOLS[@]}"; do
    echo "   🚀 Iniciando $SYMBOL klines..."
    nohup python3 download_binance_public_data.py \
        --data-type klines \
        --symbol $SYMBOL \
        --market futures \
        --intervals 1m,5m,15m,1h,4h,1d \
        --start-date 2022-11-08 \
        --end-date 2024-11-08 \
        --output-dir ./data \
        > /tmp/download_klines_${SYMBOL}.log 2>&1 &

    PIDS+=($!)
    echo "      ✅ PID: $!"
done

# ============================================================================
# 3. Funding Rate via API (todos os símbolos) - OPCIONAL
# ============================================================================
if [ "$SKIP_API" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💰 [3/5] Funding Rate via API para ${SYMBOLS[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  Via API REST (rate limited) - mais lento"
    echo ""

    for SYMBOL in "${SYMBOLS[@]}"; do
        echo "   🚀 Iniciando $SYMBOL funding rate..."
        nohup python3 download_futures_data_api.py \
            --data-type fundingRate \
            --symbol $SYMBOL \
            --start-date 2022-11-08 \
            --end-date 2024-11-08 \
            --output-dir ./data \
            > /tmp/download_funding_${SYMBOL}.log 2>&1 &

        PIDS+=($!)
        echo "      ✅ PID: $!"
    done
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏭️  [3/5] Funding Rate - PULADO (API bloqueada)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# ============================================================================
# 4. Open Interest via API (todos os símbolos) - OPCIONAL
# ============================================================================
if [ "$SKIP_API" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 [4/5] Open Interest via API para ${SYMBOLS[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for SYMBOL in "${SYMBOLS[@]}"; do
        echo "   🚀 Iniciando $SYMBOL open interest..."
        nohup python3 download_futures_data_api.py \
            --data-type openInterest \
            --symbol $SYMBOL \
            --start-date 2022-11-08 \
            --end-date 2024-11-08 \
            --output-dir ./data \
            > /tmp/download_oi_${SYMBOL}.log 2>&1 &

        PIDS+=($!)
        echo "      ✅ PID: $!"
    done
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏭️  [4/5] Open Interest - PULADO (API bloqueada)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# ============================================================================
# 5. Long/Short Ratio via API (todos os símbolos) - OPCIONAL
# ============================================================================
if [ "$SKIP_API" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 [5/5] Long/Short Ratio via API para ${SYMBOLS[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for SYMBOL in "${SYMBOLS[@]}"; do
        echo "   🚀 Iniciando $SYMBOL long/short ratio..."
        nohup python3 download_futures_data_api.py \
            --data-type longShortRatio \
            --symbol $SYMBOL \
            --start-date 2022-11-08 \
            --end-date 2024-11-08 \
            --output-dir ./data \
            > /tmp/download_ls_${SYMBOL}.log 2>&1 &

        PIDS+=($!)
        echo "      ✅ PID: $!"
    done
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏭️  [5/5] Long/Short Ratio - PULADO (API bloqueada)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# ============================================================================
# Status final
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Todos os downloads iniciados com sucesso!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Total de processos: ${#PIDS[@]}"
echo "🔢 PIDs: ${PIDS[@]}"
echo ""
echo "📝 Logs salvos em /tmp/:"
echo "   - AggTrades:     /tmp/download_aggtrades_*.log"
echo "   - Klines:        /tmp/download_klines_*.log"
echo "   - Funding Rate:  /tmp/download_funding_*.log"
echo "   - Open Interest: /tmp/download_oi_*.log"
echo "   - L/S Ratio:     /tmp/download_ls_*.log"
echo ""
echo "👀 Acompanhar progresso:"
echo "   # Binance Vision (rápido)"
echo "   tail -f /tmp/download_aggtrades_BTCUSDT.log"
echo "   tail -f /tmp/download_klines_BTCUSDT.log"
echo ""
echo "   # API REST (mais lento)"
echo "   tail -f /tmp/download_funding_BTCUSDT.log"
echo "   tail -f /tmp/download_oi_BTCUSDT.log"
echo ""
echo "📊 Ver progresso de todos:"
echo "   watch -n 5 'tail -3 /tmp/download_*.log'"
echo ""
echo "📈 Contar processos ativos:"
echo "   ps aux | grep -E 'download_binance|download_futures' | grep -v grep | wc -l"
echo ""
echo "⏸️  Parar TODOS os downloads:"
echo "   kill ${PIDS[@]}"
echo "   # ou: pkill -f 'download_binance|download_futures'"
echo ""
echo "⏱️  Tempo estimado: 60-90 minutos... ☕☕"
echo ""
echo "📁 Dados serão salvos em: ./data/"
echo "   Binance Vision:"
echo "     • ./data/aggTrades/{BTCUSDT,ETHUSDT,SOLUSDT}/"
echo "     • ./data/klines/{1m,5m,15m,1h,4h,1d}/{BTCUSDT,ETHUSDT,SOLUSDT}/"
echo ""
echo "   API REST:"
echo "     • ./data/fundingRate/{BTCUSDT,ETHUSDT,SOLUSDT}/"
echo "     • ./data/openInterest/{BTCUSDT,ETHUSDT,SOLUSDT}/"
echo "     • ./data/longShortRatio/{BTCUSDT,ETHUSDT,SOLUSDT}/"
echo ""
