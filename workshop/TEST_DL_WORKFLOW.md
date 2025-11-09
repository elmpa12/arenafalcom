# Teste do Workflow Deep Learning - AWS GPU

## Fluxo Completo

```
LOCAL (Tampa)                    AWS GPU (g4dn.xlarge)
═════════════                    ═════════════════════

selector21.py
    ↓ features
orchestrator.py ─────SSH────→    dl_heads_v8.py
    ↓ upload data                    ↓ treina modelos
    ↓ execute DL                     ↓ GRU, TCN, etc
    ↓ download results               ↓ salva weights
    ↓                                ↓
  ./work/<session>/            C:\gpu_root\out\
    results/                       └── models/
```

## Arquivos Envolvidos

### 1. Orchestrator (coordenador local)
- **orchestrator.py** - Coordena tudo
  - Conecta SSH no servidor GPU
  - Faz upload de dados
  - Executa dl_heads_v8.py remotamente
  - Baixa resultados

### 2. DL Script (executa na GPU)
- **dl_heads_v8.py** - Script de DL
  - Carrega dados
  - Treina modelos (GRU, TCN)
  - Usa selector21.py para features
  - Salva modelos treinados

### 3. Selector (feature engineering)
- **selector21.py** - Feature engineering
  - Usado por dl_heads_v8.py
  - Gera features para DL

### 4. Heads (arquiteturas DL)
- **heads.py** - Define arquiteturas
- **dl_head.py** - Classes base

## Teste Passo a Passo

### Pré-requisitos

```bash
# 1. Verificar arquivos principais
ls -lh orchestrator.py dl_heads_v8.py selector21.py heads.py

# 2. Verificar dados (se já baixou)
du -sh data/

# 3. Verificar .env
grep -E "AWS_|GPU_|DL_" .env
```

---

## TESTE 1: Validar Imports Localmente

```bash
python3 << 'EOF'
print("="*60)
print("TESTE 1: Validar imports localmente")
print("="*60)
print()

# Test orchestrator
print("1. Testando orchestrator.py...")
try:
    import orchestrator
    print("   ✅ orchestrator.py importado")
except Exception as e:
    print(f"   ❌ Erro: {e}")

# Test dl_heads_v8
print("\n2. Testando dl_heads_v8.py...")
try:
    import dl_heads_v8
    print("   ✅ dl_heads_v8.py importado")
except Exception as e:
    print(f"   ❌ Erro: {e}")

# Test selector21
print("\n3. Testando selector21.py...")
try:
    import selector21
    print("   ✅ selector21.py importado")
except Exception as e:
    print(f"   ❌ Erro: {e}")

# Test heads
print("\n4. Testando heads.py...")
try:
    from heads import available_head_names
    print(f"   ✅ heads.py importado")
    print(f"   Heads disponíveis: {available_head_names()}")
except Exception as e:
    print(f"   ❌ Erro: {e}")

print("\n" + "="*60)
print("TESTE 1 COMPLETO")
print("="*60)
EOF
```

---

## TESTE 2: Simular Conexão GPU (Dry-run)

```bash
python3 << 'EOF'
import os
from pathlib import Path

print("="*60)
print("TESTE 2: Verificar conexão GPU")
print("="*60)
print()

# Verificar metadados da última GPU
metadata_file = Path("tools/last_gpu.json")

if metadata_file.exists():
    import json
    with open(metadata_file) as f:
        meta = json.load(f)

    print("Última instância GPU:")
    print(f"  Instance ID: {meta.get('instance_id')}")
    print(f"  State: {meta.get('state')}")
    print(f"  IP: {meta.get('public_ip')}")
    print(f"  Region: {meta.get('region')}")
    print()

    if meta.get('state') == 'running':
        print("✅ GPU está rodando!")
        print("\nTestar conexão SSH:")
        ip = meta.get('public_ip')
        print(f"  ssh -i ~/.ssh/botscalp.pem ubuntu@{ip}")
    else:
        print("⚠️  GPU não está rodando")
else:
    print("⚠️  Nenhuma GPU provisionada ainda")
    print()
    print("Para provisionar:")
    print("  ./run_gpu_job.sh --dry-run")

print("\n" + "="*60)
print("TESTE 2 COMPLETO")
print("="*60)
EOF
```

---

## TESTE 3: Validar Dados para DL

```bash
python3 << 'EOF'
import pandas as pd
from pathlib import Path

print("="*60)
print("TESTE 3: Validar dados para DL")
print("="*60)
print()

# Verificar se tem dados baixados
data_dir = Path("data")

if not data_dir.exists():
    print("❌ Diretório data/ não encontrado")
    print("   Execute: ./DOWNLOAD_2_ANOS_COMPLETO.sh")
else:
    # Verificar aggTrades
    print("1. AggTrades:")
    bt_dir = data_dir / "aggTrades" / "BTCUSDT"
    if bt_dir.exists():
        files = list(bt_dir.glob("*.parquet"))
        print(f"   ✅ {len(files)} arquivos encontrados")

        if files:
            df = pd.read_parquet(files[0])
            print(f"   Colunas: {df.columns.tolist()}")
            print(f"   Shape: {df.shape}")
    else:
        print("   ❌ BTCUSDT aggTrades não encontrado")

    print()

    # Verificar klines
    print("2. Klines:")
    for tf in ['1m', '5m', '15m', '1h']:
        klines_dir = data_dir / "klines" / tf / "BTCUSDT"
        if klines_dir.exists():
            files = list(klines_dir.glob("*.parquet"))
            print(f"   ✅ {tf}: {len(files)} arquivos")
        else:
            print(f"   ⚠️  {tf}: não encontrado")

print("\n" + "="*60)
print("TESTE 3 COMPLETO")
print("="*60)
EOF
```

---

## TESTE 4: Executar Selector21 Localmente (CPU)

```bash
python3 << 'EOF'
print("="*60)
print("TESTE 4: Testar selector21.py (CPU)")
print("="*60)
print()

print("⏭️  PULANDO - Você está trabalhando no selector21")
print()
print("Quando quiser testar:")
print("  python3 selector21.py --help")
print()
print("Para rodar um teste rápido:")
print("  python3 selector21.py --symbol BTCUSDT --start 2024-11-01 --end 2024-11-07")

print("\n" + "="*60)
print("TESTE 4 PULADO")
print("="*60)
EOF
```

---

## TESTE 5: Dry-run Completo (SEM provisionar GPU)

Este é o teste **seguro** - não gasta nada!

```bash
python3 << 'EOF'
print("="*60)
print("TESTE 5: Dry-run completo (SEM GPU)")
print("="*60)
print()

print("Este teste simula o fluxo completo SEM:")
print("  ❌ Provisionar instância GPU")
print("  ❌ Fazer SSH")
print("  ❌ Gastar dinheiro")
print()
print("Ele apenas valida:")
print("  ✅ Configurações no .env")
print("  ✅ Arquivos necessários")
print("  ✅ Estrutura de diretórios")
print("  ✅ Comandos que seriam executados")
print()
print("="*60)
print()

import os
from pathlib import Path

# Verificar .env
print("1. Verificando .env...")
env_vars = {
    'GPU_HOST': os.getenv('GPU_HOST'),
    'GPU_USER': os.getenv('GPU_USER'),
    'GPU_PASSWORD': os.getenv('GPU_PASSWORD'),
    'GPU_ROOT': os.getenv('GPU_ROOT'),
    'GPU_PYTHON': os.getenv('GPU_PYTHON'),
    'DL_SCRIPT': os.getenv('DL_SCRIPT', 'dl_heads_v8.py'),
}

for k, v in env_vars.items():
    if v:
        masked = v if 'PASSWORD' not in k else '*' * len(v)
        print(f"   ✅ {k}={masked}")
    else:
        print(f"   ⚠️  {k} não configurado")

print()

# Verificar arquivos
print("2. Verificando arquivos necessários...")
required_files = [
    'orchestrator.py',
    'dl_heads_v8.py',
    'selector21.py',
    'heads.py',
    'dl_head.py',
]

for f in required_files:
    if Path(f).exists():
        print(f"   ✅ {f}")
    else:
        print(f"   ❌ {f} NÃO ENCONTRADO")

print()

# Simular comando que seria executado
print("3. Comando que seria executado:")
print()
print("   python3 orchestrator.py \\")
print("     --gpu-host <IP_GPU> \\")
print("     --gpu-user gpuadmin \\")
print("     --gpu-password *** \\")
print("     --symbol BTCUSDT \\")
print("     --start 2024-01-01 \\")
print("     --end 2024-11-08 \\")
print("     --dl-models gru,tcn \\")
print("     --dl-epochs 12")

print("\n" + "="*60)
print("TESTE 5 COMPLETO - Nenhum $ gasto! ✅")
print("="*60)
EOF
```

---

## TESTE 6: Executar Orchestrator com GPU Real (GASTA $$$)

⚠️ **ATENÇÃO:** Este teste **PROVISIONA GPU REAL** e **GASTA DINHEIRO**!

```bash
# NÃO RODAR AINDA! Apenas documentação:

# Opção 1: Provisionar nova GPU
python3 orchestrator.py \
    --gpu-host NEW \
    --symbol BTCUSDT \
    --start 2024-10-01 \
    --end 2024-11-08 \
    --dl-models gru \
    --dl-epochs 5

# Opção 2: Usar GPU já existente
python3 orchestrator.py \
    --gpu-host 100.88.219.118 \
    --gpu-user gpuadmin \
    --gpu-password coco123 \
    --symbol BTCUSDT \
    --start 2024-10-01 \
    --end 2024-11-08 \
    --dl-models gru \
    --dl-epochs 5
```

---

## Checklist Completo

Antes de rodar DL na GPU:

```bash
# ✅ 1. Dados baixados
[ ] du -sh data/aggTrades data/klines

# ✅ 2. Dependências instaladas
[ ] pip list | grep -E "torch|pandas|paramiko"

# ✅ 3. Selector21 funcionando
[ ] python3 selector21.py --help

# ✅ 4. .env configurado
[ ] grep GPU_ .env

# ✅ 5. Chaves SSH (se usar AWS)
[ ] ls ~/.ssh/botscalp.pem

# ✅ 6. Créditos AWS disponíveis
[ ] aws ec2 describe-instances --region us-east-1

# ✅ 7. Espaço em disco local (para resultados)
[ ] df -h .

# ✅ 8. Budget AWS configurado (recomendado!)
[ ] https://console.aws.amazon.com/billing/home#/budgets
```

---

## Problemas Comuns

### 1. "ModuleNotFoundError: No module named 'dl_head'"

```bash
# Verificar se arquivo existe
ls -l dl_head.py

# Instalar dependências
pip install -r requirements.txt
```

### 2. "SSH connection failed"

```bash
# Verificar GPU está rodando
cat tools/last_gpu.json

# Testar SSH manualmente
ssh -i ~/.ssh/botscalp.pem ubuntu@<IP>

# Ver security group
aws ec2 describe-security-groups --region us-east-1
```

### 3. "CUDA out of memory"

```bash
# Reduzir batch size no .env
DL_BATCH=2048  # era 8192

# Ou usar modelo menor
DL_MODELS=gru  # não usar gru,tcn,lstm juntos
```

### 4. "Data directory not found on remote"

```bash
# Verificar se dados foram copiados
# orchestrator.py deve fazer isso automaticamente

# Manualmente via SSH:
ssh -i ~/.ssh/botscalp.pem ubuntu@<IP>
ls -lh /home/ubuntu/datafull/
```

---

## Próximos Passos

Após validar todos os testes:

1. ✅ **Rodar teste 1-5** (sem gastar $)
2. ⚠️  **Decidir**: Usar servidor GPU existente ou provisionar novo?
3. 💰 **Rodar teste 6** (com GPU real)
4. 📊 **Analisar resultados** em `./work/<session>/results/`

---

## Scripts de Ajuda

### Ver status da GPU

```bash
cat tools/last_gpu.json | python3 -m json.tool
```

### Matar processos do orchestrator

```bash
pkill -f orchestrator.py
```

### Limpar resultados antigos

```bash
rm -rf work/old_*/
```

### Ver logs em tempo real

```bash
tail -f work/<session_id>/orchestrator.log
```

---

**Quer que eu rode algum desses testes agora?**
