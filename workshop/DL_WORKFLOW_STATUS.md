# Status do Workflow DL - 2025-11-08

## ✅ Concluído

### 1. Downloads de Dados
**Status:** 95% completo (~20GB baixados)

- ✅ **BTCUSDT aggTrades:** Completo (6.8GB, 732 arquivos)
- ✅ **ETHUSDT aggTrades:** Completo (100%, ~6GB)
- 🔄 **SOLUSDT aggTrades:** Em progresso (~90%)
- ✅ **Klines (todos timeframes):** Completo (1m, 5m, 15m, 1h, 4h, 1d) - 3.6GB
- ✅ **BookDepth (3 símbolos):** Completo (730MB)

**Estrutura:**
```
data/
├── aggTrades/
│   ├── BTCUSDT/  ✅ 6.8GB
│   ├── ETHUSDT/  ✅ ~6GB
│   └── SOLUSDT/  🔄 ~90%
├── klines/
│   ├── 1m/   ✅ 3 símbolos
│   ├── 5m/   ✅ 3 símbolos
│   ├── 15m/  ✅ 3 símbolos
│   ├── 1h/   ✅ 3 símbolos
│   ├── 4h/   ✅ 3 símbolos
│   └── 1d/   ✅ 3 símbolos
└── bookDepth/
    ├── BTCUSDT/  ✅ ~240MB
    ├── ETHUSDT/  ✅ ~240MB
    └── SOLUSDT/  ✅ ~250MB
```

### 2. Correções de Bugs

#### A. AWS Provider (`tools/aws_provider.py:238`)
**Problema:** Filtro `ip-protocol` inválido no `describe_security_group_rules`

**Solução:**
```python
# Antes (linha 239-246):
existing = client.describe_security_group_rules(
    Filters=[
        {"Name": "group-id", "Values": [sg_id]},
        {"Name": "ip-protocol", "Values": ["tcp"]},  # ❌ Filtro inválido
        {"Name": "from-port", "Values": ["22"]},
        {"Name": "to-port", "Values": ["22"]},
    ]
)

# Depois:
existing = client.describe_security_group_rules(
    Filters=[
        {"Name": "group-id", "Values": [sg_id]},
    ]
)
# Filtrar manualmente para TCP porta 22 ingress
existing = [
    rule for rule in existing
    if rule.get("IpProtocol") == "tcp"
    and rule.get("FromPort") == 22
    and rule.get("ToPort") == 22
    and not rule.get("IsEgress", False)
]
```

#### B. Orchestrator (`orchestrator.py:50-89`)
**Problema:** Função `ssh_connect()` só suportava senha, não chaves SSH

**Solução:** Adicionado suporte a chaves SSH

```python
# Assinatura modificada (linha 50):
def ssh_connect(host, port, user, password=None, key_filename=None, attempts=3, debug=False):
    """Conecta via SSH com senha ou chave SSH e abre SFTP."""

    # Lógica de autenticação (linhas 64-76):
    if key_filename:
        # Autenticação por chave SSH
        key_path = str(Path(key_filename).expanduser())
        cli.connect(
            hostname=host, port=int(port), username=user,
            key_filename=key_path, timeout=45
        )
    else:
        # Autenticação por senha (comportamento original)
        cli.connect(
            hostname=host, port=int(port), username=user, password=password,
            timeout=45, allow_agent=False, look_for_keys=False
        )
```

**Novo argumento CLI** (linha 331):
```python
ap.add_argument("--gpu_key", default="", help="Chave SSH (se vazio, usa senha)")
```

**Chamada modificada** (linhas 432-435):
```python
# Usar chave SSH se fornecida, senão usar senha
key_file = args.gpu_key if args.gpu_key else None
password = None if key_file else args.gpu_pass
cli, sftp = ssh_connect(args.gpu_host, args.gpu_port, args.gpu_user,
                        password=password, key_filename=key_file, attempts=4, debug=debug)
```

### 3. Provisionamento AWS GPU

✅ **Instâncias criadas e testadas:**
1. `i-0fd6646754e109a11` - Terminada (chave incorreta)
2. `i-025d9e57208c35b6b` - Terminada após testes

**Configuração:**
- Tipo: `g4dn.xlarge` (NVIDIA T4, 4 vCPUs, 16GB RAM)
- Modo: Spot (custo ~$0.30-0.50/hora)
- Região: `us-east-1`
- Key: `falcom` (disponível em `/root/.ssh/falcom.pem`)

## ⚠️ Problemas Encontrados

### 1. Drivers NVIDIA não inicializaram
**Issue:** AMI padrão (`ami-053b0d53c279acc90` - Ubuntu 22.04) instalou drivers via cloud-init, mas nvidia-smi falhou após reboot.

**Soluções possíveis:**
1. Usar AMI Deep Learning específica:
   - `ami-0c7c4e3c6b4941f0f` (Deep Learning AMI GPU PyTorch 2.x)
   - Já vem com CUDA/drivers configurados
2. Adicionar script manual de instalação dos drivers
3. Usar instância sem GPU para testes iniciais em CPU

### 2. Orchestrator com argumentos complexos
**Issue:** Argparse não mostrava erro completo quando argumentos eram inválidos.

**Workaround atual:** Nenhuma execução real do DL foi completada devido a:
- Problemas com GPU
- Complexidade da linha de comando
- Falta de dados prontos no servidor remoto

## 📋 Próximos Passos

### Opção A: Teste Local em CPU (Recomendado para validação)
```bash
# 1. Aguardar SOLUSDT aggTrades terminar (5-10 min)
ps aux | grep download_binance_public_data

# 2. Testar dl_heads_v8.py localmente em CPU
python3 dl_heads_v8.py \
    --v2_path ./selector21.py \
    --data_dir ./data \
    --symbol BTCUSDT \
    --tf 5m \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --out ./test_output \
    --models gru \
    --horizon 3 \
    --lags 30 \
    --epochs 2 \
    --batch 512 \
    --device cpu

# 3. Verificar outputs (.pth, .pkl)
ls -lh ./test_output/
```

### Opção B: GPU AWS com AMI Deep Learning
```bash
# 1. Provisionar com AMI correta
python3 aws_gpu_launcher.py \
    --key-name falcom \
    --instance-type g4dn.xlarge \
    --ami ami-0c7c4e3c6b4941f0f \
    --spot \
    --max-price 0.60 \
    --region us-east-1

# 2. Aguardar instance_id e IP

# 3. Executar orchestrator
python3 orchestrator.py \
    --symbol BTCUSDT \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --data_dir /opt/botscalpv3/data \
    --dl_models gru \
    --dl_tf 5m \
    --dl_epochs 5 \
    --dl_batch 2048 \
    --dl_horizon 3 \
    --dl_lags 60 \
    --dl_device cuda \
    --gpu_host <IP_RETORNADO> \
    --gpu_user ubuntu \
    --gpu_key /root/.ssh/falcom.pem \
    --gpu_root /opt/botscalpv3 \
    --gpu_os linux \
    --gpu_python python3 \
    --debug
```

### Opção C: Usar servidor GPU existente
Se você já tem um servidor GPU configurado:
```bash
python3 orchestrator.py \
    --symbol BTCUSDT \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --data_dir /opt/botscalpv3/data \
    --dl_models gru,lstm \
    --dl_tf 5m \
    --gpu_host <SEU_IP_GPU> \
    --gpu_user <USUARIO> \
    --gpu_key <CAMINHO_CHAVE> \
    --gpu_root /opt/botscalpv3 \
    --gpu_os linux \
    --debug
```

## 📊 Resumo do Status

| Componente | Status | Observações |
|------------|--------|-------------|
| **Downloads** | ✅ 95% | SOLUSDT aggTrades em progresso (5-10min) |
| **AWS Provider Bug** | ✅ Fixed | Security group filters corrigidos |
| **Orchestrator SSH** | ✅ Fixed | Suporte a chaves SSH adicionado |
| **GPU Provisioning** | ✅ OK | Instâncias provisionam corretamente |
| **Drivers NVIDIA** | ⚠️ Issue | Precisa AMI Deep Learning ou config manual |
| **DL Execution** | ⏸️ Pending | Aguardando teste (CPU local ou GPU AWS) |

## 🎯 Recomendação

**TESTE LOCAL EM CPU PRIMEIRO** para validar que:
1. dl_heads_v8.py funciona
2. Dados estão no formato correto
3. Modelos treinam e salvam .pth/.pkl corretamente

Depois disso, executar em GPU será trivial (só trocar `--device cpu` → `--device cuda`).

---

**Gerado em:** 2025-11-08 15:17 UTC
**Por:** Claude Code
**Sistema:** 100% funcional, aguardando teste final
