#!/usr/bin/env bash
#
# RUN_BOTSCALP.sh - Wrapper simples para master_orchestrator.py
# Uso: ./run_botscalp.sh [--dry-run] [--resume]
#

set -e

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🚀 BotScalp v3 - Master Orchestrator 🚀           ║"
echo "║          Arquitetura by Claudex 2.0 (Dual AI)             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Carrega configuração do .env se existir
if [ -f .env ]; then
    echo -e "${YELLOW}Carregando configuração do .env...${NC}"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Argumentos padrão
ARGS=(
    --key-name "${AWS_KEY_NAME:-botscalp-key}"
    --symbol "${SYMBOL:-BTCUSDT}"
    --data-dir "${DATA_DIR:-./datafull}"
    --work-dir "${WORK_DIR:-./work}"
    --ssh-key "${SSH_KEY:-~/.ssh/id_rsa}"
)

# Adiciona argumentos passados ao script
ARGS+=("$@")

# Executa
echo ""
echo -e "${GREEN}Executando master orchestrator...${NC}"
echo -e "${YELLOW}Comando: python3 master_orchestrator.py ${ARGS[@]}${NC}"
echo ""

python3 master_orchestrator.py "${ARGS[@]}"

# Captura status
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Pipeline concluído com sucesso!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  Pipeline falhou com código: $STATUS${NC}"
fi

exit $STATUS
