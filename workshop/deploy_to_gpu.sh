#!/bin/bash
# Deploy BotScalp para instância GPU

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
KEY_PATH="${HOME}/.ssh/falcom.pem"
REMOTE_USER="ubuntu"

# Ler IP da última instância
IP=$(cat /opt/botscalpv3/tools/last_gpu.json 2>/dev/null | grep -o '"public_ip": "[^"]*' | cut -d'"' -f4)

if [ -z "$IP" ]; then
    echo -e "${RED}❌ Erro: Não consegui ler o IP da instância!${NC}"
    echo "Execute primeiro: python aws_gpu_launcher.py ..."
    exit 1
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   DEPLOY BOTSCALP PARA GPU${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}IP da Instância:${NC} $IP"
echo -e "${YELLOW}Chave SSH:${NC} $KEY_PATH"
echo ""

# Step 1: Aguardar SSH estar disponível
echo -e "${YELLOW}[1/6]${NC} Aguardando SSH ficar disponível..."
for i in {1..30}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_PATH" "$REMOTE_USER@$IP" "echo 'SSH OK'" 2>/dev/null; then
        echo -e "${GREEN}✅ SSH disponível!${NC}"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

# Step 2: Criar estrutura de diretórios
echo -e "${YELLOW}[2/6]${NC} Criando estrutura de diretórios remota..."
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "$REMOTE_USER@$IP" << 'EOSSH'
mkdir -p ~/botscalpv3/{backend,tools,visual/backend,frontend,datafull}
mkdir -p ~/botscalpv3/.env_backup
echo "✅ Diretórios criados"
EOSSH

# Step 3: Upload dos arquivos principais
echo -e "${YELLOW}[3/6]${NC} Fazendo upload dos arquivos Python..."
scp -o StrictHostKeyChecking=no -i "$KEY_PATH" \
    /opt/botscalpv3/requirements.txt \
    /opt/botscalpv3/__init__.py \
    /opt/botscalpv3/selector21_core.py \
    /opt/botscalpv3/selector21.py \
    "$REMOTE_USER@$IP:~/botscalpv3/"

scp -o StrictHostKeyChecking=no -i "$KEY_PATH" \
    /opt/botscalpv3/backend/*.py \
    "$REMOTE_USER@$IP:~/botscalpv3/backend/"

scp -o StrictHostKeyChecking=no -i "$KEY_PATH" \
    /opt/botscalpv3/tools/*.py \
    "$REMOTE_USER@$IP:~/botscalpv3/tools/"

echo -e "${GREEN}✅ Arquivos Python enviados${NC}"

# Step 4: Configurar variáveis de ambiente
echo -e "${YELLOW}[4/6]${NC} Configurando variáveis de ambiente..."
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "$REMOTE_USER@$IP" << 'EOSSH'
cat > ~/.env << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-placeholder

# AWS Configuration
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=us-east-1

# Gateway
GATEWAY_TOKEN=botscalp-gpu-secret
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000

# Data paths
DATA_PATH=/home/ubuntu/botscalpv3/datafull
MODELS_PATH=/home/ubuntu/botscalpv3/models
LOGS_PATH=/home/ubuntu/botscalpv3/logs
EOF
echo "✅ .env configurado"
EOSSH

# Step 5: Instalar dependências Python
echo -e "${YELLOW}[5/6]${NC} Instalando dependências Python..."
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "$REMOTE_USER@$IP" << 'EOSSH'
cd ~/botscalpv3
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "✅ Dependências instaladas"
EOSSH

# Step 6: Validar ambiente
echo -e "${YELLOW}[6/6]${NC} Validando ambiente remoto..."
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "$REMOTE_USER@$IP" << 'EOSSH'
cd ~/botscalpv3
source venv/bin/activate

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VALIDAÇÃO DO AMBIENTE REMOTO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Python:"
python --version
echo ""

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️  GPU não disponível ainda (cloud-init em andamento)"
echo ""

echo "Pacotes principais:"
python -c "import boto3, torch, pandas, numpy; print('✅ Principais imports OK')" 2>&1 || echo "⚠️  Alguns pacotes ainda não prontos"
echo ""

echo "Diretórios:"
ls -la ~/ | grep botscalpv3
echo ""

echo "✅ Ambiente remoto validado!"
EOSSH

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ DEPLOY COMPLETO!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Próximos passos:${NC}"
echo ""
echo "1️⃣  Conectar via SSH:"
echo "   ssh -i ~/.ssh/falcom.pem ubuntu@$IP"
echo ""
echo "2️⃣  Verificar GPU:"
echo "   ssh -i ~/.ssh/falcom.pem ubuntu@$IP nvidia-smi"
echo ""
echo "3️⃣  Rodar o gateway:"
echo "   ssh -i ~/.ssh/falcom.pem ubuntu@$IP 'cd ~/botscalpv3 && source venv/bin/activate && python -m backend.openai_gateway'"
echo ""
echo "4️⃣  Executar seletor:"
echo "   ssh -i ~/.ssh/falcom.pem ubuntu@$IP 'cd ~/botscalpv3 && source venv/bin/activate && python selector21.py'"
echo ""

echo -e "${YELLOW}💡 Dica: Salve este IP para referência futura:${NC}"
echo "   IP=$IP"
echo ""
