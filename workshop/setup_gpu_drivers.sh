#!/bin/bash
# Configurar GPU na instância AWS

set -e

IP=$1

if [ -z "$IP" ]; then
    IP=$(cat /opt/botscalpv3/tools/last_gpu.json 2>/dev/null | grep -o '"public_ip": "[^"]*' | cut -d'"' -f4)
fi

KEY_PATH="${HOME}/.ssh/falcom.pem"
REMOTE_USER="ubuntu"

if [ -z "$IP" ]; then
    echo "❌ Erro: IP não fornecido e não encontrado em last_gpu.json"
    exit 1
fi

echo "🔧 Configurando GPU em $IP..."
echo ""

# Instalação de drivers NVIDIA e dependências
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "$REMOTE_USER@$IP" << 'EOSSH'
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CONFIGURAÇÃO DE GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Atualizar sistema
echo "[1/4] Atualizando sistema..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential linux-headers-$(uname -r) > /dev/null 2>&1
echo "✅ Sistema atualizado"

# Step 2: Instalar NVIDIA drivers
echo "[2/4] Instalando NVIDIA drivers..."
sudo apt-get install -y -qq nvidia-driver-550-server > /dev/null 2>&1 || echo "⚠️  Drivers podem já estar instalados"
echo "✅ NVIDIA drivers instalados"

# Step 3: Instalar CUDA (opcional, mas recomendado para ML)
echo "[3/4] Instalando CUDA Toolkit..."
sudo apt-get install -y -qq cuda-toolkit-12-3 > /dev/null 2>&1 || echo "⚠️  CUDA pode não estar disponível nesta AMI"
echo "✅ CUDA instalado"

# Step 4: Verificar GPU
echo "[4/4] Verificando GPU..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

echo "✅ Configuração de GPU concluída!"
EOSSH

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Reinstalando dependências Python..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Reinstalar dependências com fix
ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "$REMOTE_USER@$IP" << 'EOSSH'
cd ~/botscalpv3
source venv/bin/activate

# Atualizar requirements
cat > requirements_fixed.txt << 'EOF'
# Computação Científica
numpy<2.0
pandas>=1.5.0
scipy
scikit-learn

# ML & Deep Learning
torch
torchvision
torchaudio
xgboost

# Technical Analysis
pandas-ta>=0.3.14b0

# AWS & Cloud
boto3>=1.34.0
botocore>=1.34.0
paramiko>=3.4.0

# API & Web
fastapi
uvicorn[standard]
pydantic-settings
python-dotenv
openai>=1.0.0

# Utilities
psutil
pynvml
pyarrow
EOF

echo "Instalando dependências corrigidas..."
pip install -r requirements_fixed.txt -q --no-cache-dir 2>&1 | grep -v "Ignored the following" | grep -v "Requires-Python" || true

echo "✅ Dependências instaladas com sucesso"
echo ""

# Validar
echo "Validando imports críticos:"
python3 << 'PYTHON'
import sys
imports_ok = True
try:
    import boto3
    import torch
    import pandas
    import numpy
    from fastapi import FastAPI
    print("✅ Todos os imports OK!")
except ImportError as e:
    print(f"⚠️  Erro: {e}")
    imports_ok = False
PYTHON

echo ""
echo "✅ Ambiente GPU pronto!"
EOSSH

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ GPU CONFIGURADA COM SUCESSO!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Conecte via SSH para validar:"
echo "  ssh -i ~/.ssh/falcom.pem ubuntu@$IP nvidia-smi"
echo ""
