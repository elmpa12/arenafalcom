# 📚 BotScalp v3 - Documentação

Documentação completa do projeto BotScalp v3.

---

## 🚀 Início Rápido

**Na raiz do projeto:**

- [INSTALL.md](../INSTALL.md) - Instalação completa (setup.sh)
- [GPU_WORKFLOW.md](../GPU_WORKFLOW.md) - Workflow GPU (run_gpu_job.sh)
- [README_CLAUDEX.md](../README_CLAUDEX.md) - Sistema de IAs (Claudex)
- [SETUP_AWS_GPU.md](../SETUP_AWS_GPU.md) - Setup manual AWS GPU

---

## 📁 Documentação por Categoria

### 🤖 Sistema de IAs

- [AGENTS_PROFILE.md](./AGENTS_PROFILE.md) - Perfis dos agentes
- [CODEX_MODES.md](./CODEX_MODES.md) - Modos do Codex
- [DIALOGUE-SPEC.md](./DIALOGUE-SPEC.md) - Especificação de diálogo

### 📊 Dados e Microestrutura

- [DATA_PREPARATION_GUIDE.md](./DATA_PREPARATION_GUIDE.md) - Guia de preparação de dados
- [DOWNLOAD_RAPIDO_BINANCE.md](./DOWNLOAD_RAPIDO_BINANCE.md) - Download rápido Binance
- [GUIA_COMPLETO_COLETA_DADOS.md](./GUIA_COMPLETO_COLETA_DADOS.md) - Guia completo de coleta
- [MICROSTRUCTURE_DATA_COLLECTION.md](./MICROSTRUCTURE_DATA_COLLECTION.md) - Coleta de microestrutura
- [INSTRUCOES_DOWNLOAD_NO_SEU_SERVIDOR.md](./INSTRUCOES_DOWNLOAD_NO_SEU_SERVIDOR.md) - Download no servidor

### 🏆 Trading e Competição

- [COMPETITION_MODE.md](./COMPETITION_MODE.md) - Modo competição
- [HFT_COMPETITION_MODE.md](./HFT_COMPETITION_MODE.md) - Competição HFT
- [PAPER_TRADING_README.md](./PAPER_TRADING_README.md) - Paper trading
- [PRODUCTION_TRADING_GUIDE.md](./PRODUCTION_TRADING_GUIDE.md) - Guia de produção

### 🔧 Sistema e Orquestração

- [COMPLETE_SYSTEM.md](./COMPLETE_SYSTEM.md) - Sistema completo
- [ORCHESTRATOR_README.md](./ORCHESTRATOR_README.md) - Orchestrador
- [OTIMIZACOES_IMPLEMENTADAS.md](./OTIMIZACOES_IMPLEMENTADAS.md) - Otimizações

### 🌐 Gateway e APIs

- [GATEWAY_EXAMPLES.md](./GATEWAY_EXAMPLES.md) - Exemplos de gateway
- [GATEWAY_USAGE.md](./GATEWAY_USAGE.md) - Uso do gateway

### 📖 Outros

- [INDEX.md](./INDEX.md) - Índice geral
- [MANIFESTO.md](./MANIFESTO.md) - Manifesto do projeto

---

## 💬 Debates das IAs

Debates entre Claude e GPT sobre decisões arquiteturais:

- [debates/DEBATE_FORMATO_ARMAZENAMENTO.md](./debates/DEBATE_FORMATO_ARMAZENAMENTO.md)
- [debates/DEBATE_MICROSTRUCTURE_DATA.md](./debates/DEBATE_MICROSTRUCTURE_DATA.md)
- [debates/DEBATE_WF_PARAMETERS.md](./debates/DEBATE_WF_PARAMETERS.md)

---

## 📋 Estrutura do Projeto

```
botscalpv3/
├── run_gpu_job.sh              # 🚀 Script principal GPU
├── setup.sh                     # ⚙️  Setup completo
├── INSTALL.md                   # 📖 Guia de instalação
├── GPU_WORKFLOW.md              # 📖 Workflow GPU
├── README_CLAUDEX.md            # 📖 Sistema Claudex
│
├── docs/                        # 📚 Documentação
│   ├── README.md                # Este arquivo
│   ├── AGENTS_PROFILE.md
│   ├── COMPETITION_MODE.md
│   └── debates/                 # 💬 Debates das IAs
│
├── backend/                     # 🖥️  Backend FastAPI
├── visual/                      # 🎨 Visualização web
├── tools/                       # 🔧 Utilitários
├── claudex/                     # 🤖 Sistema de IAs
└── datafull/                    # 💾 Dados históricos
```

---

**Gerado automaticamente pela limpeza do repositório**
**Data:** 2025-11-08
