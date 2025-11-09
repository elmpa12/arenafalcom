#!/bin/bash
set -euo pipefail

echo "================================================"
echo "   BotScalp AWS GPU Setup Automático"
echo "================================================"
echo ""

# 1. Verificar Python
echo "[1/5] Verificando Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.9+ e tente novamente."
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION encontrado"
echo ""

# 2. Criar/ativar venv
echo "[2/5] Configurando ambiente virtual..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Ambiente virtual criado"
else
    echo "✅ Ambiente virtual já existe"
fi
source .venv/bin/activate
pip install --upgrade pip -q
echo "✅ pip atualizado"
echo ""

# 3. Instalar dependências
echo "[3/5] Instalando dependências (boto3, botocore, paramiko, python-dotenv)..."
pip install -r requirements.txt -q
echo "✅ Todas as dependências instaladas"
echo ""

# 4. Validar .env
echo "[4/5] Validando credenciais AWS no .env..."
if grep -q "AWS_ACCESS_KEY_ID=" .env && grep -q "AWS_SECRET_ACCESS_KEY=" .env; then
    echo "✅ Credenciais AWS encontradas no .env"
else
    echo "⚠️  Credenciais AWS não encontradas no .env"
    echo "   Configure manualmente ou use: aws configure --profile botscalp"
fi
echo ""

# 5. Teste rápido
echo "[5/5] Testando importação de módulos..."
python3 << 'PYEOF'
try:
    import boto3
    import botocore
    import paramiko
    from dotenv import load_dotenv
    print("✅ Todos os módulos importados com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar: {e}")
    exit(1)
PYEOF
echo ""

echo "================================================"
echo "   Setup Concluído! ✅"
echo "================================================"
echo ""
echo "Próximos passos:"
echo ""
echo "1. Confirme suas credenciais AWS:"
echo "   grep AWS .env"
echo ""
echo "2. Crie um EC2 key pair (se não tiver):"
echo "   aws ec2 create-key-pair --key-name falcom --region us-east-1 \\"
echo "     --query 'KeyMaterial' --output text > ~/.ssh/falcom.pem"
echo "   chmod 600 ~/.ssh/falcom.pem"
echo ""
echo "3. Lance uma instância:"
echo "   python aws_gpu_launcher.py --region us-east-1 --instance-type g4dn.xlarge \\"
echo "     --key-name falcom --name v3 --spot --max-price 1.50 --volume-size 50"
echo ""
echo "4. Conecte via SSH:"
echo "   ssh -i ~/.ssh/falcom.pem ubuntu@<IP_PÚBLICO>"
echo ""
echo "Para mais detalhes, consulte: SETUP_AWS_GPU.md"
echo ""
