#!/bin/bash
################################################################################
# TESTE_CLAUDEX.sh - Teste rápido das capacidades das IAs
################################################################################

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      🤖 TESTE CLAUDEX - IAs Colaborando em Código         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd /home/user/botscalpv3

# Verificar .env
if ! grep -q "OPENAI_API_KEY" .env; then
    echo "❌ OPENAI_API_KEY não configurada no .env"
    exit 1
fi

echo "✅ API Keys configuradas"
echo ""

# Menu de testes
echo "Escolha um teste:"
echo ""
echo "1️⃣  Pipeline Completo (PLAN + IMPLEMENT + REVIEW)"
echo "   Tarefa: Criar função para calcular RSI"
echo ""
echo "2️⃣  Debate Técnico"
echo "   Tema: Melhor timeframe para scalping BTC"
echo ""
echo "3️⃣  Pipeline Avançado"
echo "   Tarefa: Sistema de detecção de regime de volatilidade"
echo ""

read -p "Escolha (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Executando Pipeline: RSI Calculator"
        echo ""
        python3 claudex_dual_gpt.py --pipeline "Criar uma função Python para calcular RSI (Relative Strength Index) com janela configurável, retornando um pandas Series"
        ;;
    2)
        echo ""
        echo "💬 Iniciando Debate: Timeframes para Scalping"
        echo ""
        python3 claudex_dual_gpt.py --debate "Qual o melhor timeframe para scalping em BTC/USDT? 1m, 5m ou 15m? Considerar edge, noise e execution speed"
        ;;
    3)
        echo ""
        echo "🚀 Executando Pipeline Avançado: Detector de Volatilidade"
        echo ""
        python3 claudex_dual_gpt.py --pipeline "Criar detector de regime de volatilidade usando ATR e Bollinger Bands. Deve classificar em: low, normal, high, extreme. Retornar DataFrame com regime e confidence score"
        ;;
    *)
        echo "❌ Opção inválida"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ TESTE COMPLETO!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📁 Resultados salvos em:"
echo "   /home/user/botscalpv3/claudex/work/<session_id>/"
echo ""
echo "📂 Arquivos gerados:"
if [ "$choice" == "1" ] || [ "$choice" == "3" ]; then
    echo "   ✅ spec.json - Planejamento"
    echo "   ✅ implementation.json - CÓDIGO PYTHON!"
    echo "   ✅ REVIEW.md - Review cruzado"
else
    echo "   ✅ debate.json - Debate completo"
fi
echo ""
