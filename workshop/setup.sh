#!/usr/bin/env bash
# ============================================================================
# BOTSCALP V3 - SCRIPT DE INSTALAÇÃO COMPLETO
# ============================================================================
# Data: 2025-11-08
# Uso: bash setup.sh [--gpu] [--cpu]
#
# Opções:
#   --gpu     Instala PyTorch com suporte CUDA (para servidores GPU)
#   --cpu     Instala PyTorch CPU apenas (padrão)
#   --help    Mostra ajuda
#
# Exemplo:
#   bash setup.sh --gpu    # Para servidor GPU
#   bash setup.sh          # Para servidor CPU/local
# ============================================================================

set -e  # Para em caso de erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funções de output
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Banner
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         BOTSCALP V3 - INSTALAÇÃO COMPLETA                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Parse argumentos
USE_GPU=false
if [[ "$*" == *"--gpu"* ]]; then
    USE_GPU=true
    info "Modo GPU ativado (instalará PyTorch com CUDA)"
elif [[ "$*" == *"--help"* ]]; then
    echo "Uso: bash setup.sh [--gpu] [--cpu]"
    echo ""
    echo "Opções:"
    echo "  --gpu     Instala PyTorch com suporte CUDA (para servidores GPU)"
    echo "  --cpu     Instala PyTorch CPU apenas (padrão)"
    echo "  --help    Mostra esta ajuda"
    echo ""
    echo "Exemplo:"
    echo "  bash setup.sh --gpu    # Para servidor GPU"
    echo "  bash setup.sh          # Para servidor CPU/local"
    exit 0
else
    info "Modo CPU (padrão). Use --gpu para instalar com suporte CUDA"
fi

# ============================================================================
# 1. VERIFICAR PYTHON
# ============================================================================

info "Verificando Python..."

if ! command -v python3 &> /dev/null; then
    error "Python 3 não encontrado!"
    echo ""
    echo "Instale Python 3.9+ com:"
    echo "  Ubuntu/Debian: sudo apt-get update && sudo apt-get install python3 python3-pip python3-venv"
    echo "  CentOS/RHEL:   sudo yum install python3 python3-pip"
    echo "  macOS:         brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
success "Python $PYTHON_VERSION encontrado"

# Verificar versão mínima (3.9+)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    error "Python 3.9+ necessário (encontrado $PYTHON_VERSION)"
    exit 1
fi

# ============================================================================
# 2. INSTALAR PYTHON3-VENV (se necessário)
# ============================================================================

info "Verificando python3-venv..."

if ! python3 -m venv --help &> /dev/null; then
    warning "python3-venv não encontrado. Tentando instalar..."

    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3-venv
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-venv
    else
        error "Não foi possível instalar python3-venv automaticamente"
        echo "Instale manualmente com:"
        echo "  Ubuntu/Debian: sudo apt-get install python3-venv"
        echo "  CentOS/RHEL:   sudo yum install python3-venv"
        exit 1
    fi

    success "python3-venv instalado"
else
    success "python3-venv disponível"
fi

# ============================================================================
# 3. CRIAR AMBIENTE VIRTUAL
# ============================================================================

info "Criando ambiente virtual..."

if [ -d "venv" ]; then
    warning "Ambiente virtual já existe em ./venv"
    read -p "Deseja remover e recriar? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        rm -rf venv
        info "Removido ./venv antigo"
    else
        info "Usando venv existente"
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    success "Ambiente virtual criado em ./venv"
else
    success "Usando ./venv existente"
fi

# ============================================================================
# 4. ATIVAR AMBIENTE VIRTUAL
# ============================================================================

info "Ativando ambiente virtual..."

source venv/bin/activate

if [ "$VIRTUAL_ENV" != "" ]; then
    success "Ambiente virtual ativado: $VIRTUAL_ENV"
else
    error "Falha ao ativar ambiente virtual"
    exit 1
fi

# ============================================================================
# 5. ATUALIZAR PIP, SETUPTOOLS, WHEEL
# ============================================================================

info "Atualizando pip, setuptools e wheel..."

python -m pip install --upgrade pip setuptools wheel --quiet

PIP_VERSION=$(pip --version | cut -d' ' -f2)
success "pip $PIP_VERSION atualizado"

# ============================================================================
# 6. INSTALAR PYTORCH (GPU ou CPU)
# ============================================================================

if [ "$USE_GPU" = true ]; then
    info "Instalando PyTorch com suporte CUDA..."
    echo ""
    warning "Verificando se há GPU disponível..."

    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        echo ""
        info "GPU detectada! Instalando PyTorch para CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        success "PyTorch (CUDA) instalado"
    else
        warning "nvidia-smi não encontrado. Instalando PyTorch CPU como fallback..."
        pip install torch torchvision torchaudio
        success "PyTorch (CPU) instalado"
    fi
else
    info "Instalando PyTorch (CPU apenas)..."
    pip install torch torchvision torchaudio
    success "PyTorch (CPU) instalado"
fi

# ============================================================================
# 7. INSTALAR TA-LIB (se disponível)
# ============================================================================

info "Verificando TA-Lib..."

if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    if ! dpkg -l | grep -q libta-lib0; then
        warning "TA-Lib não encontrado. Tentando instalar..."
        if sudo apt-get install -y ta-lib libta-lib-dev &> /dev/null; then
            success "TA-Lib system library instalada"
        else
            warning "Não foi possível instalar TA-Lib. Continuando sem ela..."
        fi
    else
        success "TA-Lib system library disponível"
    fi
elif command -v brew &> /dev/null; then
    # macOS
    if ! brew list ta-lib &> /dev/null; then
        warning "TA-Lib não encontrada. Tentando instalar..."
        if brew install ta-lib &> /dev/null; then
            success "TA-Lib instalada via Homebrew"
        else
            warning "Não foi possível instalar TA-Lib. Continuando sem ela..."
        fi
    else
        success "TA-Lib disponível via Homebrew"
    fi
else
    warning "TA-Lib deve ser instalada manualmente neste sistema"
fi

# ============================================================================
# 8. INSTALAR REQUIREMENTS.TXT
# ============================================================================

info "Instalando dependências do requirements.txt..."
echo ""

if [ ! -f "requirements.txt" ]; then
    error "requirements.txt não encontrado!"
    exit 1
fi

# Instala requirements (exceto torch, já instalado)
pip install -r requirements.txt

success "Todas as dependências instaladas!"

# ============================================================================
# 9. VERIFICAR INSTALAÇÃO
# ============================================================================

info "Verificando instalações principais..."
echo ""

# Função para verificar pacote
check_package() {
    local package=$1
    local display_name=$2

    if python -c "import $package" 2>/dev/null; then
        local version=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "N/A")
        success "$display_name: $version"
        return 0
    else
        warning "$display_name: NÃO INSTALADO"
        return 1
    fi
}

# Verificar pacotes principais
check_package "numpy" "NumPy"
check_package "pandas" "Pandas"
check_package "scipy" "SciPy"
check_package "sklearn" "scikit-learn"
check_package "xgboost" "XGBoost"
check_package "torch" "PyTorch"
check_package "fastapi" "FastAPI"
check_package "openai" "OpenAI"
check_package "anthropic" "Anthropic"
check_package "binance" "Binance Connector"

echo ""

# ============================================================================
# 10. TESTAR TORCH GPU (se instalado)
# ============================================================================

if [ "$USE_GPU" = true ]; then
    info "Testando PyTorch GPU..."

    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available (usando CPU)')
"
    echo ""
fi

# ============================================================================
# 11. CRIAR SCRIPT DE ATIVAÇÃO
# ============================================================================

info "Criando script de ativação rápida..."

cat > activate.sh << 'EOF'
#!/usr/bin/env bash
# Script para ativar o ambiente virtual
source venv/bin/activate
echo "✓ Ambiente virtual ativado!"
echo "Python: $(python --version)"
echo "Pip: $(pip --version | cut -d' ' -f1-2)"
echo ""
echo "Para desativar: deactivate"
EOF

chmod +x activate.sh

success "Script criado: ./activate.sh"

# ============================================================================
# 12. CRIAR .env TEMPLATE (se não existir)
# ============================================================================

if [ ! -f ".env" ]; then
    info "Criando template .env..."

    cat > .env << 'EOF'
# ============================================================================
# BOTSCALP V3 - VARIÁVEIS DE AMBIENTE
# ============================================================================

# OPENAI (GPT-4, GPT-4o)
OPENAI_API_KEY=sk-...

# ANTHROPIC (Claude Sonnet 4, Opus 4)
ANTHROPIC_API_KEY=sk-ant-...

# BINANCE
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# AWS (opcional - para servidor GPU)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# DATABASE (opcional)
# DATABASE_URL=postgresql://user:pass@localhost/botscalp

# ============================================================================
EOF

    success "Template .env criado (CONFIGURE AS API KEYS!)"
else
    success ".env já existe"
fi

# ============================================================================
# 13. RESUMO FINAL
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ INSTALAÇÃO COMPLETA!                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

success "Python $PYTHON_VERSION"
success "Ambiente virtual: ./venv"
success "Pip: $PIP_VERSION"

if [ "$USE_GPU" = true ]; then
    success "PyTorch: GPU (CUDA)"
else
    success "PyTorch: CPU"
fi

echo ""
info "PRÓXIMOS PASSOS:"
echo ""
echo "  1. Ative o ambiente virtual:"
echo "     source venv/bin/activate"
echo "     # OU"
echo "     bash activate.sh"
echo ""
echo "  2. Configure as API keys no .env:"
echo "     nano .env"
echo ""
echo "  3. Teste a instalação:"
echo "     python3 -c 'import torch; print(torch.__version__)'"
echo ""
echo "  4. Execute o sistema:"
echo "     python3 selector21.py --help"
echo "     python3 claudex_dual_gpt.py --help"
echo ""

if [ "$USE_GPU" = true ]; then
    echo "  5. Para treinar na GPU:"
    echo "     python3 selector21.py --run_dl --device cuda"
    echo ""
fi

echo "  Para desativar o ambiente virtual:"
echo "     deactivate"
echo ""

success "Setup completo! 🚀"
echo ""
