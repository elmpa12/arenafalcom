# 📦 GUIA DE INSTALAÇÃO - BotScalp v3

**Data:** 2025-11-08
**Servidor zerado?** Siga este guia! ✅

---

## ⚡ INSTALAÇÃO RÁPIDA (1 comando)

```bash
# No diretório do projeto
bash setup.sh
```

**Isso fará TUDO automaticamente:**
- ✅ Verifica Python 3.9+
- ✅ Instala python3-venv
- ✅ Cria ambiente virtual
- ✅ Atualiza pip/setuptools/wheel
- ✅ Instala PyTorch
- ✅ Instala TODAS as dependências
- ✅ Verifica instalação
- ✅ Cria scripts auxiliares

---

## 🖥️ INSTALAÇÃO NO SERVIDOR GPU

```bash
# Para servidor GPU (instala PyTorch com CUDA)
bash setup.sh --gpu
```

**Detecta automaticamente:**
- Se tem NVIDIA GPU disponível
- Instala PyTorch com CUDA 11.8
- Testa GPU após instalação

---

## 🛠️ INSTALAÇÃO MANUAL (Passo a Passo)

### 1. Instalar Python 3.9+ (se necessário)

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv
```

#### CentOS/RHEL:
```bash
sudo yum install -y python3 python3-pip python3-venv
```

#### macOS:
```bash
brew install python@3.11
```

---

### 2. Instalar python3-venv

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-venv

# CentOS/RHEL
sudo yum install -y python3-venv
```

---

### 3. Criar Ambiente Virtual

```bash
cd /home/user/botscalpv3
python3 -m venv venv
```

---

### 4. Ativar Ambiente Virtual

```bash
source venv/bin/activate

# OU use o script criado
bash activate.sh
```

---

### 5. Atualizar pip, setuptools, wheel

```bash
pip install --upgrade pip setuptools wheel
```

---

### 6. Instalar PyTorch

#### Para GPU (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Para CPU:
```bash
pip install torch torchvision torchaudio
```

---

### 7. Instalar TA-Lib (Opcional)

#### Ubuntu/Debian:
```bash
sudo apt-get install -y ta-lib libta-lib-dev
pip install ta-lib
```

#### macOS:
```bash
brew install ta-lib
pip install ta-lib
```

**Nota:** TA-Lib é opcional. O sistema funciona sem ela, mas algumas funções técnicas estarão desabilitadas.

---

### 8. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

### 9. Configurar Variáveis de Ambiente

```bash
# Copiar template
cp .env.example .env  # ou editar .env criado pelo setup.sh

# Editar .env
nano .env
```

**Configure:**
```bash
# OPENAI
OPENAI_API_KEY=sk-...

# ANTHROPIC (Claude)
ANTHROPIC_API_KEY=sk-ant-...

# BINANCE
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

---

## ✅ VERIFICAR INSTALAÇÃO

### 1. Testar Python

```bash
python --version
# Deve mostrar: Python 3.9+ ou superior
```

---

### 2. Testar Pacotes Principais

```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import openai; print('OpenAI:', openai.__version__)"
python -c "import anthropic; print('Anthropic:', anthropic.__version__)"
```

---

### 3. Testar PyTorch GPU (se instalado)

```bash
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
"
```

**Output esperado (com GPU):**
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA A100-SXM4-80GB
```

---

### 4. Testar Claudex

```bash
python3 claudex_dual_gpt.py --help
```

**Output esperado:**
```
╔════════════════════════════════════════════════════════════════════════════╗
║                     🔥 CLAUDEX DUAL - Sistema de Debate                   ║
║                Claude vs GPT ou GPT vs GPT (auto-detect)                  ║
╚════════════════════════════════════════════════════════════════════════════╝
...
```

---

### 5. Testar Selector21

```bash
python3 selector21.py --help
```

---

## 🚀 USO RÁPIDO

### Ativar Ambiente Virtual

```bash
# Opção 1
source venv/bin/activate

# Opção 2 (script criado pelo setup)
bash activate.sh
```

---

### Desativar Ambiente Virtual

```bash
deactivate
```

---

### Executar Scripts

```bash
# Ativar venv primeiro
source venv/bin/activate

# Depois executar
python3 selector21.py --symbol BTCUSDT --run_base
python3 claudex_dual_gpt.py --debate "tema"
python3 download_binance_public_data.py --help
```

---

## 📊 ESTRUTURA DE DEPENDÊNCIAS

### Essenciais (obrigatórios)
- Python 3.9+
- python-dotenv
- requests
- pyyaml

### Data Science
- numpy
- scipy
- pandas
- pyarrow

### Machine Learning
- scikit-learn
- xgboost
- imbalanced-learn

### Deep Learning
- torch
- torchvision
- torchaudio

### Trading
- binance-connector
- pandas-ta

### AI APIs
- openai (GPT-4)
- anthropic (Claude)

### Web Services
- fastapi
- uvicorn
- pydantic

### AWS
- boto3
- botocore
- paramiko
- awscli

### Utils
- tqdm
- psutil
- pynvml (GPU monitoring)

### Dev/Test
- pytest
- ruff
- black
- mypy

---

## 🐛 TROUBLESHOOTING

### Erro: "python3-venv not found"

```bash
# Ubuntu/Debian
sudo apt-get install python3-venv

# CentOS/RHEL
sudo yum install python3-venv
```

---

### Erro: "ta-lib not found"

**Opção 1:** Instalar ta-lib system library

```bash
# Ubuntu/Debian
sudo apt-get install ta-lib libta-lib-dev
pip install ta-lib

# macOS
brew install ta-lib
pip install ta-lib
```

**Opção 2:** Continuar sem ta-lib (opcional)

---

### Erro: "CUDA not available" (mas tem GPU)

**Verificar:**
1. Driver NVIDIA instalado: `nvidia-smi`
2. CUDA version: `nvidia-smi` (top right)
3. PyTorch instalado corretamente:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Erro: "ModuleNotFoundError: No module named 'X'"

```bash
# Ativar venv
source venv/bin/activate

# Reinstalar requirements
pip install -r requirements.txt

# Ou instalar pacote específico
pip install <nome-do-pacote>
```

---

### Erro: "Permission denied" ao executar setup.sh

```bash
chmod +x setup.sh
bash setup.sh
```

---

## 📝 NOTAS IMPORTANTES

### 1. PyTorch GPU

Para treinar na GPU, **SEMPRE** instale PyTorch com CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**NÃO** instale apenas `pip install torch` (vai instalar CPU por padrão).

---

### 2. Polars (Alternativa ao Pandas)

Se quiser usar Polars (10-15x mais rápido):

```bash
pip install polars
```

---

### 3. DuckDB (Queries SQL em Parquet)

Para queries rápidas em arquivos Parquet:

```bash
pip install duckdb
```

---

### 4. Ambiente Virtual

**SEMPRE** ative o venv antes de executar scripts:

```bash
source venv/bin/activate
```

Sem ativar, os pacotes não estarão disponíveis!

---

### 5. Visual Replay Backend

Para o visual replay backend:

```bash
cd visual/backend
pip install -r requirements.txt
```

---

## 🎯 PRÓXIMOS PASSOS

Após instalação completa:

1. ✅ Configure API keys no `.env`
2. ✅ Baixe dados: `bash DOWNLOAD_2_ANOS_COMPLETO.sh`
3. ✅ Execute backtests: `bash COMANDO_WF_OTIMIZADO.sh`
4. ✅ Teste debates: `python3 claudex_dual_gpt.py --debate "teste"`
5. ✅ Treine DL na GPU: `python3 selector21.py --run_dl --device cuda`

---

## 📁 ARQUIVOS CRIADOS PELO SETUP

- `venv/` - Ambiente virtual
- `activate.sh` - Script de ativação rápida
- `.env` - Template de variáveis de ambiente (se não existir)

---

## ✨ DICAS

### Alias Úteis (adicione no ~/.bashrc)

```bash
# Ativar venv do BotScalp
alias botscalp='cd /home/user/botscalpv3 && source venv/bin/activate'

# Claudex
alias claudex='cd /home/user/botscalpv3 && source venv/bin/activate && python3 claudex_dual_gpt.py'
```

Depois: `source ~/.bashrc`

---

### Script de Atualização

Para atualizar dependências:

```bash
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install --upgrade -r requirements.txt
```

---

**Happy Trading! 🚀📈**
