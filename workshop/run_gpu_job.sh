#!/usr/bin/env bash
################################################################################
# RUN_GPU_JOB.sh - Provisionamento Temporário AWS GPU
################################################################################
#
# Este script:
#   1. Cria instância GPU spot na AWS (barato!)
#   2. Aguarda ficar pronta + SSH disponível
#   3. Faz deploy do código
#   4. Executa o job de DL
#   5. Baixa resultados
#   6. DESTRÓI a instância ($$$ economizado!)
#
# Uso:
#   ./run_gpu_job.sh
#   ./run_gpu_job.sh --dry-run        # Simula sem gastar $
#   ./run_gpu_job.sh --no-cleanup     # Mantém instância após job
#   ./run_gpu_job.sh --reuse          # Usa instância existente
#
################################################################################

set -e

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuração padrão (pode sobrescrever via .env)
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${GPU_INSTANCE_TYPE:-g4dn.xlarge}"
KEY_NAME="${AWS_KEY_NAME:-botscalp}"
INSTANCE_NAME="${GPU_INSTANCE_NAME:-botscalp-temp-gpu}"
SPOT="${USE_SPOT:-true}"
MAX_PRICE="${SPOT_MAX_PRICE:-1.50}"
VOLUME_SIZE="${GPU_VOLUME_SIZE:-50}"

# Paths
KEY_PATH="${HOME}/.ssh/${KEY_NAME}.pem"
METADATA_FILE="tools/last_gpu.json"
WORK_DIR="./work/$(date +%Y%m%d_%H%M%S)"

# Flags
DRY_RUN=false
NO_CLEANUP=false
REUSE=false

# Parse argumentos
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --reuse)
            REUSE=true
            shift
            ;;
        *)
            ;;
    esac
done

# Banner
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       🚀 AWS GPU Job Runner - BotScalp v3 🚀                ║${NC}"
echo -e "${CYAN}║       (Provisiona → Executa → Destrói)                      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}⚠️  MODO DRY-RUN: Simulando sem gastar dinheiro!${NC}"
    echo ""
fi

# Carregar .env se existir
if [ -f .env ]; then
    echo -e "${BLUE}[INFO]${NC} Carregando configuração do .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Criar diretório de trabalho
mkdir -p "$WORK_DIR"
LOG_FILE="$WORK_DIR/job.log"

# Função de log
log() {
    local level=$1
    shift
    local msg="$@"
    local timestamp=$(date '+%H:%M:%S')

    case $level in
        INFO)
            echo -e "${BLUE}[${timestamp}] [INFO]${NC} $msg" | tee -a "$LOG_FILE"
            ;;
        SUCCESS)
            echo -e "${GREEN}[${timestamp}] [OK]${NC} $msg" | tee -a "$LOG_FILE"
            ;;
        WARN)
            echo -e "${YELLOW}[${timestamp}] [WARN]${NC} $msg" | tee -a "$LOG_FILE"
            ;;
        ERROR)
            echo -e "${RED}[${timestamp}] [ERRO]${NC} $msg" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Cleanup handler (chamado ao sair)
cleanup_on_exit() {
    if [ "$NO_CLEANUP" = true ]; then
        log WARN "Cleanup desabilitado (--no-cleanup), instância mantida!"
        return 0
    fi

    if [ -f "$METADATA_FILE" ]; then
        INSTANCE_ID=$(cat "$METADATA_FILE" | grep -o '"instance_id": "[^"]*' | cut -d'"' -f4)

        if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "null" ]; then
            log WARN "Terminando instância GPU: $INSTANCE_ID"

            if [ "$DRY_RUN" = false ]; then
                python3 << PYEOF
import os
os.environ["AWS_ACCESS_KEY_ID"] = "${AWS_ACCESS_KEY_ID}"
os.environ["AWS_SECRET_ACCESS_KEY"] = "${AWS_SECRET_ACCESS_KEY}"
os.environ["AWS_DEFAULT_REGION"] = "${REGION}"

import boto3
ec2 = boto3.client('ec2', region_name='${REGION}')

try:
    response = ec2.terminate_instances(InstanceIds=['${INSTANCE_ID}'])
    print(f"✅ Instância {response['TerminatingInstances'][0]['InstanceId']} sendo terminada ({response['TerminatingInstances'][0]['CurrentState']['Name']})")
except Exception as e:
    print(f"❌ Erro ao terminar instância: {e}")
PYEOF
                log SUCCESS "Instância terminada! $$$ economizado!"
            else
                log INFO "DRY-RUN: Instância seria terminada aqui"
            fi
        fi
    fi
}

# Registra cleanup ao sair (Ctrl+C, erro, ou fim normal)
trap cleanup_on_exit EXIT INT TERM

################################################################################
# ESTÁGIO 1: Provisionar GPU
################################################################################

log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log INFO "ESTÁGIO 1/6: Provisionamento de Instância GPU"
log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verificar se chave SSH existe
if [ ! -f "$KEY_PATH" ] && [ "$DRY_RUN" = false ] && [ "$REUSE" = false ]; then
    log ERROR "Chave SSH não encontrada: $KEY_PATH"
    log WARN "Criando key pair automaticamente..."

    python3 << PYEOF
import os
os.environ["AWS_ACCESS_KEY_ID"] = "${AWS_ACCESS_KEY_ID}"
os.environ["AWS_SECRET_ACCESS_KEY"] = "${AWS_SECRET_ACCESS_KEY}"
os.environ["AWS_DEFAULT_REGION"] = "${REGION}"

import boto3
ec2 = boto3.client('ec2', region_name='${REGION}')

try:
    # Tenta criar o key pair
    response = ec2.create_key_pair(KeyName='${KEY_NAME}')

    # Salva a chave privada
    with open('${KEY_PATH}', 'w') as f:
        f.write(response['KeyMaterial'])

    # Define permissões corretas
    os.chmod('${KEY_PATH}', 0o600)

    print(f"✅ Key pair '{KEY_NAME}' criado e salvo em ${KEY_PATH}")
except ec2.exceptions.ClientError as e:
    if 'InvalidKeyPair.Duplicate' in str(e):
        print(f"⚠️  Key pair '${KEY_NAME}' já existe na AWS, mas chave privada não encontrada localmente!")
        print(f"   Solução 1: Use --reuse para reutilizar instância existente")
        print(f"   Solução 2: Delete o key pair na AWS e rode novamente")
        exit(1)
    else:
        print(f"❌ Erro: {e}")
        exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        log ERROR "Falha ao criar key pair!"
        exit 1
    fi
fi

# Provisionar instância
if [ "$DRY_RUN" = true ]; then
    log WARN "DRY-RUN: Simulando provisionamento..."
    echo '{"instance_id":"i-dry-run-fake","public_ip":"1.2.3.4","state":"running"}' > "$METADATA_FILE"
    INSTANCE_IP="1.2.3.4"
else
    CMD="python3 aws_gpu_launcher.py --region $REGION --instance-type $INSTANCE_TYPE --key-name $KEY_NAME --name $INSTANCE_NAME --volume-size $VOLUME_SIZE --metadata $METADATA_FILE"

    if [ "$SPOT" = true ]; then
        CMD="$CMD --spot --max-price $MAX_PRICE"
    fi

    if [ "$REUSE" = true ]; then
        CMD="$CMD --reuse"
    fi

    log INFO "Comando: $CMD"

    if ! eval "$CMD"; then
        log ERROR "Falha ao provisionar instância GPU!"
        exit 1
    fi

    INSTANCE_IP=$(cat "$METADATA_FILE" | grep -o '"public_ip": "[^"]*' | cut -d'"' -f4)
    INSTANCE_ID=$(cat "$METADATA_FILE" | grep -o '"instance_id": "[^"]*' | cut -d'"' -f4)

    log SUCCESS "Instância provisionada: $INSTANCE_ID ($INSTANCE_IP)"
fi

echo ""

################################################################################
# ESTÁGIO 2: Aguardar SSH
################################################################################

log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log INFO "ESTÁGIO 2/6: Aguardando SSH ficar disponível"
log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$DRY_RUN" = false ]; then
    log INFO "Testando conexão SSH em $INSTANCE_IP..."

    for i in {1..60}; do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes -i "$KEY_PATH" "ubuntu@$INSTANCE_IP" "echo 'SSH OK'" 2>/dev/null; then
            log SUCCESS "SSH disponível!"
            break
        fi
        echo -n "."
        sleep 5

        if [ $i -eq 60 ]; then
            log ERROR "Timeout aguardando SSH (5 minutos)"
            exit 1
        fi
    done
    echo ""
else
    log WARN "DRY-RUN: SSH não testado"
fi

echo ""

################################################################################
# ESTÁGIO 3: Deploy de Código
################################################################################

log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log INFO "ESTÁGIO 3/6: Deploy de código para GPU"
log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$DRY_RUN" = false ]; then
    if [ -f "deploy_to_gpu.sh" ]; then
        log INFO "Usando deploy_to_gpu.sh..."
        bash deploy_to_gpu.sh
    else
        log WARN "deploy_to_gpu.sh não encontrado, fazendo deploy básico..."

        # Deploy mínimo
        ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" "ubuntu@$INSTANCE_IP" "mkdir -p ~/botscalpv3"

        scp -o StrictHostKeyChecking=no -i "$KEY_PATH" requirements.txt "ubuntu@$INSTANCE_IP:~/botscalpv3/"
        scp -o StrictHostKeyChecking=no -i "$KEY_PATH" *.py "ubuntu@$INSTANCE_IP:~/botscalpv3/" 2>/dev/null || true

        log SUCCESS "Deploy básico concluído"
    fi
else
    log WARN "DRY-RUN: Deploy não executado"
fi

echo ""

################################################################################
# ESTÁGIO 4: Executar Job DL
################################################################################

log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log INFO "ESTÁGIO 4/6: Executando Deep Learning na GPU"
log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$DRY_RUN" = false ]; then
    log INFO "Executando orchestrator.py remotamente..."

    # Usa orchestrator.py para rodar DL remotamente
    python3 orchestrator.py \
        --gpu-host "$INSTANCE_IP" \
        --gpu-user ubuntu \
        --gpu-key "$KEY_PATH" \
        --dl-script "${DL_SCRIPT:-dl_heads_v8.py}" \
        --symbol "${SYMBOL:-BTCUSDT}" \
        --dl-models "${DL_MODELS:-gru,tcn}" \
        --dl-epochs "${DL_EPOCHS:-12}" || {
            log ERROR "Falha ao executar DL!"
            exit 1
        }

    log SUCCESS "Job DL concluído!"
else
    log WARN "DRY-RUN: Job DL não executado"
fi

echo ""

################################################################################
# ESTÁGIO 5: Download de Resultados
################################################################################

log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log INFO "ESTÁGIO 5/6: Download de resultados"
log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

RESULTS_DIR="$WORK_DIR/results"
mkdir -p "$RESULTS_DIR"

if [ "$DRY_RUN" = false ]; then
    log INFO "Baixando resultados via rsync..."

    rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -i $KEY_PATH" \
        "ubuntu@$INSTANCE_IP:~/botscalpv3/dl_out/" \
        "$RESULTS_DIR/" || {
            log WARN "Falha ao baixar resultados (pode não existir)"
        }

    log SUCCESS "Resultados salvos em: $RESULTS_DIR"
else
    log WARN "DRY-RUN: Download não executado"
fi

echo ""

################################################################################
# ESTÁGIO 6: Relatório Final
################################################################################

log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log INFO "ESTÁGIO 6/6: Gerando relatório final"
log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

REPORT_FILE="$WORK_DIR/REPORT.md"

cat > "$REPORT_FILE" << EOF
# BotScalp v3 - GPU Job Report

**Session ID:** $(basename $WORK_DIR)
**Date:** $(date '+%Y-%m-%d %H:%M:%S')

## AWS Instance

- **Instance ID:** ${INSTANCE_ID:-N/A}
- **Instance Type:** $INSTANCE_TYPE
- **Region:** $REGION
- **IP:** ${INSTANCE_IP:-N/A}
- **Spot:** $SPOT
- **Dry Run:** $DRY_RUN

## Job Configuration

- **DL Script:** ${DL_SCRIPT:-dl_heads_v8.py}
- **Symbol:** ${SYMBOL:-BTCUSDT}
- **Models:** ${DL_MODELS:-gru,tcn}
- **Epochs:** ${DL_EPOCHS:-12}

## Results

- **Work Dir:** $WORK_DIR
- **Results Dir:** $RESULTS_DIR
- **Log File:** $LOG_FILE

## Cost Estimate

- **Instance Type:** $INSTANCE_TYPE (~\$0.30/h spot)
- **Estimated Runtime:** 30-60 min
- **Estimated Cost:** \$0.15 - \$0.30 💰

## Cleanup

- **Auto-cleanup:** $([ "$NO_CLEANUP" = true ] && echo "DISABLED" || echo "ENABLED")
- **Instance Status:** $([ "$NO_CLEANUP" = true ] && echo "Running (manual cleanup required)" || echo "Terminated")

---

Generated by \`run_gpu_job.sh\`
EOF

log SUCCESS "Relatório salvo em: $REPORT_FILE"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              ✅ JOB COMPLETO COM SUCESSO! ✅                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}📊 Resumo:${NC}"
echo -e "   Work Dir: ${BOLD}$WORK_DIR${NC}"
echo -e "   Results: ${BOLD}$RESULTS_DIR${NC}"
echo -e "   Report: ${BOLD}$REPORT_FILE${NC}"
echo ""

if [ "$NO_CLEANUP" = true ]; then
    echo -e "${YELLOW}⚠️  ATENÇÃO: Instância GPU ainda está RODANDO!${NC}"
    echo -e "   Para evitar custos, termine manualmente:"
    echo -e "   ${BOLD}aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION${NC}"
    echo ""
fi

echo -e "${GREEN}💰 Custo estimado: \$0.15 - \$0.30${NC}"
echo ""

exit 0
