#!/bin/bash
################################################################################
# DOWNLOAD TURBO - Binance Data (10-20x mais rápido!)
################################################################################

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    🚀 BINANCE TURBO DOWNLOADER - Downloads Paralelos 🚀     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Verificar se está no diretório correto
if [ ! -f "download_binance_turbo.py" ]; then
    echo "❌ ERRO: Execute dentro de /opt/botscalpv3/"
    exit 1
fi

# Verificar dependências
echo "📦 Verificando dependências..."
python3 -c "import pandas, pyarrow, requests, tqdm, concurrent.futures" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Instalando dependências..."
    pip3 install pandas pyarrow requests tqdm --quiet
fi

echo "✅ Dependências OK!"
echo ""

# Configuração
START_DATE="2022-11-08"
END_DATE="2024-11-08"
SYMBOL="BTCUSDT"
MARKET="futures"
WORKERS=15  # 15 downloads simultâneos!

echo "📊 Configuração:"
echo "   Símbolo: $SYMBOL"
echo "   Período: $START_DATE → $END_DATE"
echo "   Workers: $WORKERS (downloads simultâneos)"
echo "   Output: ./data/"
echo ""

# Calcular dias
DAYS=$(( ($(date -d "$END_DATE" +%s) - $(date -d "$START_DATE" +%s)) / 86400 ))
echo "📅 Total: $DAYS dias de dados"
echo ""

# Escolher o que baixar
echo "O que deseja baixar?"
echo ""
echo "1️⃣  AggTrades (rápido: ~5-10 min)"
echo "2️⃣  Klines 1m (médio: ~10-15 min)"
echo "3️⃣  Klines 5m (rápido: ~3-5 min)"
echo "4️⃣  Klines 15m (rápido: ~2-3 min)"
echo "5️⃣  TUDO (todos acima: ~20-30 min)"
echo ""

read -p "Escolha (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Baixando AggTrades..."
        python3 download_binance_turbo.py \
            --market $MARKET \
            --data-type aggTrades \
            --symbol $SYMBOL \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --workers $WORKERS \
            --output-dir ./data
        ;;

    2)
        echo ""
        echo "🚀 Baixando Klines 1m..."
        python3 download_binance_turbo.py \
            --market $MARKET \
            --data-type klines \
            --symbol $SYMBOL \
            --intervals 1m \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --workers $WORKERS \
            --output-dir ./data
        ;;

    3)
        echo ""
        echo "🚀 Baixando Klines 5m..."
        python3 download_binance_turbo.py \
            --market $MARKET \
            --data-type klines \
            --symbol $SYMBOL \
            --intervals 5m \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --workers $WORKERS \
            --output-dir ./data
        ;;

    4)
        echo ""
        echo "🚀 Baixando Klines 15m..."
        python3 download_binance_turbo.py \
            --market $MARKET \
            --data-type klines \
            --symbol $SYMBOL \
            --intervals 15m \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --workers $WORKERS \
            --output-dir ./data
        ;;

    5)
        echo ""
        echo "🚀 Baixando TUDO (AggTrades + Klines 1m, 5m, 15m)..."
        echo ""

        # AggTrades
        echo "📈 [1/4] AggTrades..."
        python3 download_binance_turbo.py \
            --market $MARKET \
            --data-type aggTrades \
            --symbol $SYMBOL \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --workers $WORKERS \
            --output-dir ./data

        echo ""

        # Klines 1m
        echo "📊 [2/4] Klines 1m..."
        python3 download_binance_turbo.py \
            --market $MARKET \
            --data-type klines \
            --symbol $SYMBOL \
            --intervals 1m \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --workers $WORKERS \
            --output-dir ./data

        echo ""

        # Klines 5m
        echo "📊 [3/4] Klines 5m..."
        python3 download_binance_turbo.py \
            --market $MARKET \
            --data-type klines \
            --symbol $SYMBOL \
            --intervals 5m \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --workers $WORKERS \
            --output-dir ./data

        echo ""

        # Klines 15m
        echo "📊 [4/4] Klines 15m..."
        python3 download_binance_turbo.py \
            --market $MARKET \
            --data-type klines \
            --symbol $SYMBOL \
            --intervals 15m \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --workers $WORKERS \
            --output-dir ./data
        ;;

    *)
        echo "❌ Opção inválida"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 ✅ DOWNLOAD COMPLETO! ✅                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Dados salvos em: ./data/"
echo ""
echo "📊 Verificar tamanho:"
echo "   du -sh ./data/*"
echo ""
echo "🔍 Listar arquivos:"
echo "   ls -lh ./data/*/BTCUSDT/*.parquet | head -20"
echo ""
