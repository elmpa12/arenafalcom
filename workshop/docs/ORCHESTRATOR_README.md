# 🚀 Master Orchestrator - BotScalp v3

**Arquitetura planejada por Claudex 2.0** (Sistema Dual AI)

---

## 📋 O QUE É?

Sistema que **integra todo o pipeline** do BotScalp v3 em **um único comando**:

```bash
./run_botscalp.sh
```

### **Pipeline Completo:**
1. ✅ **Provisiona GPU** na AWS (g4dn.xlarge)
2. ✅ **Roda Selector21** localmente com Walk-Forward
3. ✅ **Transfere dados** para GPU remota
4. ✅ **Executa DL** (GRU/TCN) na GPU
5. ✅ **Baixa resultados**
6. ✅ **Consolida tudo**
7. ✅ **Cleanup** automático

---

## 🎯 CARACTERÍSTICAS

### **Robusto:**
- ✅ Retry logic com exponential backoff
- ✅ State management (retoma de onde parou)
- ✅ Logs centralizados com timestamps
- ✅ Cleanup automático em caso de erro

### **Flexível:**
- ✅ Dry-run mode (testa sem executar)
- ✅ Resume capability (retoma sessão anterior)
- ✅ Configurável via CLI ou .env
- ✅ Suporta todos os argumentos do selector21.py

### **Completo:**
- ✅ Walk-Forward (WF) configurável
- ✅ ML (XGB/RF/LogReg) com otimização de threshold
- ✅ ATR Stop/TP dinâmicos
- ✅ Features de aggtrades + depth
- ✅ Hard limits de stop/TP em USD

---

## 🚀 USO RÁPIDO

### **1. Teste DRY-RUN (recomendado primeiro):**
```bash
./test_orchestrator.sh
```

Isso simula todo o pipeline sem executar nada de verdade.

### **2. Execução COMPLETA:**
```bash
./example_full_run.sh
```

Isso executa tudo de verdade:
- Provisiona GPU na AWS
- Roda selector com WF de 3 meses
- Inclui ML com todas as features
- Usa ATR stop/TP
- Roda DL por 12 epochs
- Consolida e limpa tudo

### **3. Execução CUSTOMIZADA:**
```bash
python3 master_orchestrator.py \
  --key-name sua-chave-aws \
  --symbol BTCUSDT \
  --data-dir ./datafull \
  --start 2024-01-01 \
  --end 2024-12-31 \
  \
  --wf-train-months 6.0 \
  --wf-val-months 2.0 \
  --wf-step-months 1.0 \
  --wf-expand \
  \
  --run-ml \
  --ml-model-kind auto \
  --ml-use-agg \
  --ml-use-depth \
  --ml-opt-thr \
  \
  --use-atr-stop \
  --use-atr-tp \
  \
  --dl-models "gru,tcn,lstm" \
  --dl-epochs 20 \
  \
  --work-dir ./work
```

---

## 📂 ESTRUTURA DE ARQUIVOS

```
botscalpv3/
├── master_orchestrator.py       # Orchestrator principal (NOVO!)
├── run_botscalp.sh              # Wrapper simples
├── example_full_run.sh          # Exemplo completo
├── test_orchestrator.sh         # Teste dry-run
│
├── aws_gpu_launcher.py          # Provisiona AWS
├── orchestrator.py              # Executa DL remoto
├── selector21.py                # Selector com WF
├── dl_heads_v8.py               # Deep Learning
├── heads.py                     # Features
│
└── work/                        # Outputs
    └── 20251108_HHMMSS/         # Cada sessão
        ├── pipeline_state.json  # Estado do pipeline
        ├── master.log           # Log completo
        ├── selector_out/        # Output do selector
        ├── results/             # Resultados do DL
        └── FINAL_REPORT.md      # Relatório final
```

---

## 🛠️ ARGUMENTOS PRINCIPAIS

### **Pipeline:**
```bash
--dry-run               # Simula execução (não roda de verdade)
--resume                # Retoma pipeline anterior
--max-retries 3         # Tentativas por estágio
--no-cleanup            # Não limpa AWS ao final
--work-dir ./work       # Diretório de trabalho
```

### **AWS:**
```bash
--aws-region us-east-1
--instance-type g4dn.xlarge
--key-name sua-chave
--aws-spot              # Usa instância spot (mais barato)
```

### **Selector:**
```bash
--symbol BTCUSDT
--data-dir ./datafull
--start 2023-01-01
--end 2025-11-01
--exec-rules "1m,5m,15m"
```

### **Walk-Forward (CRÍTICO!):**
```bash
--wf-train-months 3.0    # Meses de treino
--wf-val-months 1.0      # Meses de validação
--wf-step-months 1.0     # Step em meses
--wf-expand              # Expanding (senão anchored)
```

### **Machine Learning:**
```bash
--run-ml                 # Ativa ML
--ml-model-kind auto     # xgb, rf, logreg, ou auto
--ml-use-agg             # Features de aggtrades
--ml-use-depth           # Features de depth
--ml-opt-thr             # Otimiza threshold
```

### **ATR Stop/TP:**
```bash
--use-atr-stop
--atr-stop-mult 2.0
--use-atr-tp
--atr-tp-mult "2.5,2.5,3.0"  # Por TF (1m, 5m, 15m)
```

### **Hard Limits:**
```bash
--hard-stop-usd "60,80,100"    # Stop em USD por TF
--hard-tp-usd "300,360,400"    # TP em USD por TF
```

### **Features Extras:**
```bash
--agg-dir ./datafull/BTCUSDT.aggtrades.parquet
--depth-dir ./datafull/BTCUSDT.depthfeat_1m.parquet
--depth-field bd_imb_50bps
```

### **Deep Learning:**
```bash
--dl-models "gru,tcn"    # Modelos a rodar
--dl-epochs 12           # Epochs de treino
--gpu-user ubuntu
--gpu-root /opt/botscalpv3
```

---

## 📊 EXEMPLO DE OUTPUT

Depois de rodar, você terá:

```
work/20251108_050821/
├── pipeline_state.json      # Estado do pipeline
├── master.log               # Log detalhado
├── aws_metadata.json        # Info da instância AWS
│
├── selector_out/            # Resultados do Selector
│   ├── best_params.json
│   ├── backtest_results.csv
│   └── ml_models/
│
├── results/                 # Resultados do DL
│   ├── gru_predictions.csv
│   ├── tcn_predictions.csv
│   └── models/
│
└── FINAL_REPORT.md          # ⭐ RELATÓRIO FINAL
```

---

## 🔄 RETOMANDO EXECUÇÃO

Se o pipeline falhar no meio:

```bash
python3 master_orchestrator.py --resume \
  --work-dir ./work \
  --key-name sua-chave
```

Ele vai:
1. Carregar estado anterior
2. Pular estágios já completados
3. Retomar de onde parou

---

## 🧪 TESTES

### **1. Dry-Run (recomendado):**
```bash
./test_orchestrator.sh
```

Saída esperada:
```
✅ Teste PASSOU! Pipeline simulado com sucesso.
```

### **2. Walk-Forward apenas:**
```bash
python3 selector21.py \
  --symbol BTCUSDT \
  --data_dir ./datafull \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --exec_rules "5m" \
  --run_base --run_combos \
  --walkforward \
  --wf_train_months 2.0 \
  --wf_val_months 0.5 \
  --wf_step_months 0.5 \
  --print_top10
```

### **3. Pipeline completo (curto):**
```bash
python3 master_orchestrator.py \
  --dry-run \
  --symbol BTCUSDT \
  --start 2024-11-01 \
  --end 2024-11-07 \
  --exec-rules "5m" \
  --wf-train-months 0.25 \
  --wf-val-months 0.1 \
  --key-name test
```

---

## ⚠️ TROUBLESHOOTING

### **Erro: "key-name required"**
```bash
# Configure a chave SSH na AWS primeiro:
aws ec2 create-key-pair --key-name botscalp-key --query 'KeyMaterial' --output text > ~/.ssh/botscalp-key.pem
chmod 400 ~/.ssh/botscalp-key.pem
```

### **Erro: "Selector failed"**
```bash
# Teste o selector isoladamente:
python3 selector21.py --symbol BTCUSDT --data_dir ./datafull --start 2024-11-01 --end 2024-11-07 --exec_rules "5m" --run_base
```

### **Erro: "Data transfer failed"**
```bash
# Verifique conectividade SSH:
ssh -i ~/.ssh/sua-chave.pem ubuntu@<IP>
```

### **Pipeline travou?**
```bash
# Verifique o log:
tail -f work/*/master.log

# Estado atual:
cat work/*/pipeline_state.json
```

---

## 💡 DICAS PRO

### **1. Use .env para configuração:**
```bash
# .env
AWS_KEY_NAME=botscalp-key
SYMBOL=BTCUSDT
DATA_DIR=./datafull
WORK_DIR=./work
SSH_KEY=~/.ssh/botscalp-key.pem
```

Depois:
```bash
source .env
./run_botscalp.sh
```

### **2. Monitore custos AWS:**
```bash
# Sempre use --aws-spot para economizar
# Cleanup automático está ativado por padrão
# Use --no-cleanup apenas se precisar debugar
```

### **3. Otimize WF:**
```bash
# Testes rápidos:
--wf-train-months 1.0 --wf-val-months 0.25 --wf-step-months 0.25

# Produção:
--wf-train-months 6.0 --wf-val-months 2.0 --wf-step-months 1.0 --wf-expand
```

### **4. ML em produção:**
```bash
# Sempre use:
--run-ml \
--ml-model-kind auto \
--ml-use-agg \
--ml-use-depth \
--ml-opt-thr
```

---

## 🎯 ROADMAP

- [x] Pipeline básico funcionando
- [x] Walk-Forward integrado
- [x] ML completo
- [x] ATR Stop/TP
- [x] Retry logic
- [x] State management
- [x] Cleanup automático
- [ ] Dashboard web de monitoramento
- [ ] Notificações Telegram/Slack
- [ ] Multi-symbol paralelizado
- [ ] Auto-scaling AWS

---

## 🏆 CRÉDITOS

**Arquitetura planejada por:**
- 🤖 **Claudex 2.0** (Sistema Dual AI)
  - GPT-Strategist (visão estratégica)
  - GPT-Executor (implementação técnica)

**Resultado:**
- Debate de 3 rounds
- Consenso em arquitetura modular
- 9 próximos passos implementados
- Sistema production-ready

---

## 📞 SUPORTE

**Logs:**
```bash
tail -f work/*/master.log
```

**Estado:**
```bash
cat work/*/pipeline_state.json | jq .
```

**Relatório:**
```bash
cat work/*/FINAL_REPORT.md
```

---

**Status:** ✅ Production Ready
**Versão:** 1.0
**Data:** 2025-11-08
**Criado por:** Master Orchestrator (Claudex 2.0)
