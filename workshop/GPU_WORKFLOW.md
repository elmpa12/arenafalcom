# 🚀 GPU Workflow - BotScalp v3

**Última atualização:** 2025-11-08

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Quick Start](#quick-start)
3. [Workflow Completo](#workflow-completo)
4. [Configuração](#configuração)
5. [Uso Avançado](#uso-avançado)
6. [Troubleshooting](#troubleshooting)
7. [Custos](#custos)

---

## 🎯 Visão Geral

O BotScalp v3 usa **instâncias GPU temporárias** na AWS para treinar modelos de Deep Learning, economizando custos ao **criar → usar → destruir** automaticamente.

### Por que não GPU always-on?

- ❌ GPU always-on: **~$220/mês** (g4dn.xlarge)
- ✅ GPU on-demand: **$0.15-$0.30/job** (30-60min)
- 💰 **Economia: ~99%** se rodar 10-20 jobs/mês

---

## ⚡ Quick Start

### Primeira vez (setup único):

```bash
cd /opt/botscalpv3

# 1. Baixar dados (2 anos)
bash DOWNLOAD_2_ANOS_COMPLETO.sh

# 2. Instalar dependências
bash setup.sh

# 3. Configurar .env (já está pronto!)
nano .env  # Verificar se AWS_ACCESS_KEY_ID está correto
```

### Rodar job de DL na GPU:

```bash
# Modo normal (provisiona, roda, destrói)
./run_gpu_job.sh

# Ou testar sem gastar $ (dry-run)
./run_gpu_job.sh --dry-run
```

**Pronto!** Em 30-60 minutos você terá modelos DL treinados e instância destruída! 🎉

---

## 🔄 Workflow Completo

O script `run_gpu_job.sh` executa automaticamente:

### **Estágio 1: Provisionamento** ⚙️
- Cria key pair SSH (se não existir)
- Provisiona instância spot g4dn.xlarge (~$0.30/h)
- Aguarda instância ficar `running`

### **Estágio 2: SSH** 🔌
- Aguarda SSH ficar disponível (até 5 min)
- Testa conexão

### **Estágio 3: Deploy** 📦
- Cria estrutura de diretórios
- Upload de código Python
- Configura `.env` remoto
- Instala dependências (`requirements.txt`)
- Valida ambiente (GPU, PyTorch, etc)

### **Estágio 4: Deep Learning** 🧠
- Executa `orchestrator.py` remotamente
- Roda DL (GRU, TCN, etc)
- Treina modelos com GPU

### **Estágio 5: Download** ⬇️
- Baixa resultados via `rsync`
- Salva em `./work/<session_id>/results/`

### **Estágio 6: Cleanup** 🗑️
- **Termina instância** (economiza $$$)
- Gera relatório final

---

## ⚙️ Configuração

### Variáveis no `.env`:

```bash
# AWS Credentials (OBRIGATÓRIO)
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
AWS_REGION=us-east-1

# GPU Job Config
AWS_KEY_NAME=botscalp              # Nome do key pair
GPU_INSTANCE_TYPE=g4dn.xlarge      # Tipo de instância
GPU_INSTANCE_NAME=botscalp-temp-gpu
USE_SPOT=true                      # Usar spot (mais barato!)
SPOT_MAX_PRICE=1.50               # Preço máximo ($0.30 normal)
GPU_VOLUME_SIZE=50                # GB de disco

# DL Config
DL_SCRIPT=dl_heads_v8.py
DL_MODELS=gru,tcn
DL_EPOCHS=12
SYMBOL=BTCUSDT
```

### Arquivos Necessários:

```
botscalpv3/
├── run_gpu_job.sh             ← Script principal
├── aws_gpu_launcher.py         ← Provisiona instância
├── orchestrator.py             ← Executa DL remotamente
├── deploy_to_gpu.sh            ← Deploy de código
├── dl_heads_v8.py              ← Script de DL
├── requirements.txt            ← Dependências
├── .env                        ← Configuração
└── tools/
    ├── aws_provider.py         ← Provider AWS
    └── providers.py            ← Interface de providers
```

---

## 🎓 Uso Avançado

### Flags Disponíveis:

```bash
# Simular sem gastar $ (dry-run)
./run_gpu_job.sh --dry-run

# Manter instância após job (para debug)
./run_gpu_job.sh --no-cleanup

# Reutilizar instância existente
./run_gpu_job.sh --reuse
```

### Sobrescrever Configurações:

```bash
# Mudar tipo de instância
GPU_INSTANCE_TYPE=g4dn.2xlarge ./run_gpu_job.sh

# Usar on-demand ao invés de spot
USE_SPOT=false ./run_gpu_job.sh

# Rodar mais épocas
DL_EPOCHS=24 ./run_gpu_job.sh
```

### Monitorar Progresso:

```bash
# Acompanhar logs em tempo real
tail -f ./work/<session_id>/job.log

# Ver relatório final
cat ./work/<session_id>/REPORT.md

# Listar resultados
ls -lh ./work/<session_id>/results/
```

---

## 🔧 Troubleshooting

### Problema: "Chave SSH não encontrada"

**Solução:**
```bash
# O script cria automaticamente! Mas se falhar:
aws ec2 create-key-pair --key-name botscalp --region us-east-1 \
  --query 'KeyMaterial' --output text > ~/.ssh/botscalp.pem
chmod 600 ~/.ssh/botscalp.pem
```

### Problema: "InsufficientInstanceCapacity"

**Solução:**
```bash
# Tente outro tipo ou região:
GPU_INSTANCE_TYPE=g4dn.2xlarge ./run_gpu_job.sh
# OU
AWS_REGION=us-west-2 ./run_gpu_job.sh
```

### Problema: "AuthFailure: Unable to validate credentials"

**Solução:**
```bash
# Verifique credenciais no .env:
grep AWS_ACCESS_KEY_ID .env
grep AWS_SECRET_ACCESS_KEY .env

# Ou configure via AWS CLI:
aws configure --profile botscalp
export AWS_PROFILE=botscalp
```

### Problema: "Timeout aguardando SSH"

**Solução:**
```bash
# 1. Verificar se instância iniciou:
aws ec2 describe-instances --region us-east-1

# 2. Verificar security group (porta 22 liberada?)
# 3. Aguardar mais tempo (cloud-init pode demorar 5-10 min)
```

### Problema: "deploy_to_gpu.sh não encontrado"

**Solução:**
```bash
# O script faz deploy básico automaticamente se não encontrar
# Mas você pode baixar novamente:
git pull origin claude/review-botscalpv3-project-011CUun7aaeuet1uVKikS4vS
```

### Problema: Instância não foi terminada (cobrança contínua!)

**Solução:**
```bash
# Terminar manualmente:
INSTANCE_ID=$(cat tools/last_gpu.json | grep instance_id | cut -d'"' -f4)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1

# Verificar se foi terminada:
aws ec2 describe-instances --instance-ids $INSTANCE_ID --region us-east-1
```

---

## 💰 Custos

### Instâncias GPU Comuns:

| Tipo | GPU | VRAM | vCPU | Preço On-Demand | Preço Spot | Economia |
|------|-----|------|------|----------------|------------|----------|
| g4dn.xlarge | T4 | 16GB | 4 | $0.526/h | ~$0.16-0.30/h | ~70% |
| g4dn.2xlarge | T4 | 16GB | 8 | $0.752/h | ~$0.23-0.45/h | ~70% |
| g5.xlarge | A10G | 24GB | 4 | $1.006/h | ~$0.30-0.60/h | ~70% |
| g5.2xlarge | A10G | 24GB | 8 | $1.212/h | ~$0.36-0.72/h | ~70% |

### Estimativa de Custo por Job:

| Cenário | Tempo | Instância | Custo |
|---------|-------|-----------|-------|
| **Rápido** | 30 min | g4dn.xlarge spot | $0.15 |
| **Normal** | 60 min | g4dn.xlarge spot | $0.30 |
| **Pesado** | 120 min | g4dn.2xlarge spot | $0.90 |

### Economia Mensal:

- **10 jobs/mês:** ~$3.00 vs $220 always-on = **98.6% economia**
- **20 jobs/mês:** ~$6.00 vs $220 always-on = **97.3% economia**
- **50 jobs/mês:** ~$15.00 vs $220 always-on = **93.2% economia**

**Breakeven:** ~73 jobs/mês (neste ponto always-on é mais barato)

---

## 📊 Monitoramento de Custos

### Ver custos na AWS:

```bash
# Custos do dia
aws ce get-cost-and-usage \
  --time-period Start=$(date +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost

# Custos por serviço
aws ce get-cost-and-usage \
  --time-period Start=$(date +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

### Alertas de Budget (recomendado!):

1. Acesse: https://console.aws.amazon.com/billing/home#/budgets
2. Crie budget: **$50/mês** (ou valor desejado)
3. Configure alerta via email/SNS

---

## 🎯 Best Practices

### ✅ **DO:**
- Use **spot instances** sempre que possível
- Execute `--dry-run` antes de rodar de verdade
- Monitore custos semanalmente
- Verifique se instância foi terminada após job
- Mantenha credenciais AWS seguras (não commitar .env!)

### ❌ **DON'T:**
- Deixar instância rodando sem necessidade
- Usar on-demand sem motivo (spot é 70% mais barato)
- Ignorar alertas de budget
- Compartilhar chaves SSH publicamente
- Commitar credenciais no git

---

## 📚 Recursos Adicionais

### Documentação Relacionada:

- `SETUP_AWS_GPU.md` - Setup manual AWS
- `INSTALL.md` - Instalação completo
- `README_CLAUDEX.md` - Sistema de IAs
- `SISTEMA_APRENDIZADO.md` - Learning system

### AWS Docs:

- [EC2 Spot Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)
- [GPU Instance Types](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [Cost Management](https://aws.amazon.com/aws-cost-management/)

---

## 🆘 Suporte

### Em caso de problemas:

1. **Verifique logs:** `./work/<session_id>/job.log`
2. **Teste dry-run:** `./run_gpu_job.sh --dry-run`
3. **Consulte troubleshooting** acima
4. **Issues GitHub:** https://github.com/falcomlabs/botscalpv3/issues

---

## 🔄 Ciclo de Vida Completo

```
┌──────────────────────────────────────────────────────────┐
│  1. Desenvolvedor roda: ./run_gpu_job.sh                │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  2. Script provisiona GPU spot (~$0.30/h)               │
│     • Cria key pair (se necessário)                     │
│     • Lança instância g4dn.xlarge                       │
│     • Aguarda ficar running + SSH disponível            │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  3. Deploy de código                                     │
│     • Upload via scp/rsync                              │
│     • Instala dependências                              │
│     • Configura .env                                    │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  4. Executa DL na GPU (30-60 min)                       │
│     • orchestrator.py coordena                          │
│     • dl_heads_v8.py treina modelos                     │
│     • GRU, TCN, etc                                     │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  5. Download de resultados                               │
│     • rsync baixa modelos treinados                     │
│     • Salva em ./work/<session_id>/results/             │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  6. CLEANUP AUTOMÁTICO 💰                                │
│     • Termina instância GPU                             │
│     • Gera relatório final                              │
│     • Custo total: $0.15-$0.30                          │
└──────────────────────────────────────────────────────────┘
```

---

**Gerado por Claude Code - BotScalp v3**
**Data:** 2025-11-08
