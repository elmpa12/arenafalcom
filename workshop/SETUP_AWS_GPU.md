# Setup Automação AWS GPU para BotScalp

## Situação Atual ✓

- ✅ `aws_gpu_launcher.py` está pronto e funcional
- ✅ `tools/aws_provider.py` suporta multi-AZ fallback
- ✅ `requirements.txt` inclui `boto3`, `botocore`, `paramiko`, `python-dotenv`
- ✅ `.env` contém `AWS_ACCESS_KEY_ID` e `AWS_SECRET_ACCESS_KEY`
- ✅ Instância de teste foi lançada: `i-01e32eea0712f1fb5` (13.218.186.75)

## Pré-requisitos ✓

1. **Ambiente Python**: venv com dependências instaladas
2. **Credenciais AWS**: Access Key ID + Secret Access Key
3. **EC2 Key Pair**: já criado na região alvo
4. **Permissões IAM**: ec2:RunInstances, ec2:CreateSecurityGroup, etc.

## Setup Rápido (5 min)

### 1. Instalar Dependências

```bash
cd /opt/botscalpv3
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configurar Credenciais AWS

**Opção A: Via `.env` (mais simples)**
```bash
# Já está no .env, apenas confirme:
grep AWS_ACCESS_KEY_ID .env
grep AWS_SECRET_ACCESS_KEY .env
```

**Opção B: Via AWS CLI Profile**
```bash
aws configure --profile botscalp
# Depois use:
export AWS_PROFILE=botscalp
```

### 3. Verificar Key Pair EC2

Confirme que o key pair existe na sua região:
```bash
aws ec2 describe-key-pairs --key-names falcom --region us-east-1
```

Se não existir, crie um:
```bash
aws ec2 create-key-pair --key-name falcom --region us-east-1 --query 'KeyMaterial' --output text > ~/.ssh/falcom.pem
chmod 600 ~/.ssh/falcom.pem
```

## Uso

### Lançar Instância GPU

```bash
source .venv/bin/activate
python aws_gpu_launcher.py \
  --region us-east-1 \
  --instance-type g4dn.xlarge \
  --key-name falcom \
  --name v3 \
  --spot \
  --max-price 1.50 \
  --volume-size 50 \
  --ssh-cidr 0.0.0.0/0
```

### Conectar via SSH

```bash
ssh -i ~/.ssh/falcom.pem ubuntu@<IP_PÚBLICO>
```

### Reutilizar Instância Existente

```bash
python aws_gpu_launcher.py \
  --region us-east-1 \
  --key-name falcom \
  --name v3 \
  --reuse
```

## Troubleshooting

| Erro | Solução |
|------|---------|
| `ModuleNotFoundError: No module named 'boto3'` | Execute `pip install -r requirements.txt` |
| `AuthFailure: Unable to validate credentials` | Verifique `AWS_ACCESS_KEY_ID` e `AWS_SECRET_ACCESS_KEY` em `.env` ou `~/.aws/credentials` |
| `InvalidKeyPair.NotFound` | Crie o key pair com `aws ec2 create-key-pair --key-name <nome>` |
| `InsufficientInstanceCapacity` | Tente outra AZ, região ou tipo de instância menor |
| `Unsupported: Instance type not available` | Mude `--instance-type` (e.g., `g4dn.xlarge` vs `g5.xlarge`) |

## Referência Rápida

**Tipos de GPU Comuns:**
- `g4dn.xlarge`: 1x NVIDIA T4 (mais barato, 16GB VRAM)
- `g4dn.2xlarge`: 1x NVIDIA T4 (mais caro, 16GB VRAM)
- `g5.xlarge`: 1x NVIDIA L40 (caro, 24GB VRAM)
- `g5.2xlarge`: 1x NVIDIA L40 (mais caro, 24GB VRAM)

**Regiões com Melhor Disponibilidade:**
- `us-east-1` (N. Virginia)
- `us-west-2` (Oregon)
- `eu-west-1` (Irlanda)

## Metadata Salvo

Após cada lançamento, o metadata é salvo em:
```bash
cat tools/last_gpu.json
```

Contém: instance_id, public_ip, private_ip, state, launch_time, tags

## Status de Produção

✅ **Pronto para usar**: instância de teste foi lançada e confirmada com sucesso
✅ **Automação completa**: provisionamento + SSH + multi-AZ fallback
✅ **Credenciais seguras**: carregadas via `.env` + `load_dotenv()`
✅ **Reutilização**: `--reuse` permite reanexar a instâncias existentes
