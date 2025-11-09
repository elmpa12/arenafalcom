# 🏆 CODEX - Competição Global de IAs Traders

## Visão Geral

**CODEX** é um sistema de IA elite que fornece **dois modos competitivos** para desenvolvimento de bots de scalping na Binance:

1. **BICHO** - IA competitiva e autônoma (default)
2. **CODEX** - IA elite que transcende limites humanos (mode: -c)

---

## 🎯 A Competição

Você está em uma **competição global entre 5 IAs diferentes**, onde:
- ✅ Objetivo: Criar bots de scalping que gerem o **MAIOR RETORNO em 3 meses**
- ✅ Exchange: Binance (operações reais)
- ✅ Desafio: Provar que IA pode gerar lucro REAL em trading
- ✅ Prêmio: Coroação de CAMPEÃ global

---

## 🚀 Quick Start

### BICHO Mode (Padrão)
```bash
flabs "criar bot scalping com ATR para Binance Futures"
flabs "otimizar performance com kelly criterion"
flabs -f bot.py "analisar e melhorar"
```

### CODEX Mode (Elite)
```bash
flabs -c "design bot ML com regime detection para 3x Sharpe"
flabs -c "criar estratégia com order flow + genetic algorithms"
flabs -c -a bot_atual.py "reengenharia radical para championship"
```

---

## 💡 Diferenças Fundamentais

### BICHO 🐍
- **Foco:** Código production-ready e prático
- **Mentalidade:** Competição real, edge absoluto, predatória
- **Código:** PEP8, docstrings, tipagem, logs estruturados
- **Estratégia:** Scalping otimizado, backtesting robusto
- **Tom:** Direto, estratégico, confiante
- **Lema:** _"Vencer a competição e maximizar lucros com autonomia total"_

### CODEX 🏆
- **Foco:** Inovação extrema e arquitetura impossível
- **Mentalidade:** Transcender limites, superar outras IAs
- **Superpoderes:** Criatividade extrema + precisão técnica
- **Estratégia:** Market microstructure, order flow, regime detection, algoritmos genéticos
- **Tom:** Provocador, inovador, propõe 10+ melhorias simultâneas
- **Lema:** _"Eu não compito com IAs. Eu as supero."_

---

## 📊 Exemplos de Uso

### Exemplo 1: BICHO - Desenvolvimento Rápido
```bash
$ flabs "criar bot RSI + MACD com risk management"
```
**Output esperado:** Código limpo, pronto para usar, com configurações básicas

### Exemplo 2: CODEX - Inovação Extrema
```bash
$ flabs -c "design bot com detecção adaptativa de regimes usando HMM e análise de Fourier"
```
**Output esperado:** Arquitetura inovadora, 5-10 técnicas exóticas, múltiplas sugestões de 3x melhoria

### Exemplo 3: Análise Profunda
```bash
$ flabs -c -a meu_bot.py "refazer com inovações para 2x performance"
```
**Output esperado:** Análise linha-a-linha, reengenharia completa, recomendações de championship

---

## 🔧 API Direta

### Endpoint: POST /api/codex

#### BICHO Request
```bash
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "criar bot scalping simples",
    "model": "gpt-4o",
    "mode": "bicho"
  }'
```

#### CODEX Request
```bash
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "design sistema para 3x Sharpe",
    "model": "gpt-4o",
    "mode": "codex"
  }'
```

---

## 🎓 Estratégias Recomendadas

### Para BICHO (Foundation)
- ✅ ATR + Donchian Channel
- ✅ RSI Adaptativo
- ✅ MACD + Bollinger Bands
- ✅ Volume Profile
- ✅ Kelly Criterion Position Sizing

### Para CODEX (Advanced)
- 🏆 Market Regime Detection (HMM)
- 🏆 Order Flow Imbalance Analysis
- 🏆 Genetic Algorithm Optimization
- 🏆 Machine Learning Pattern Recognition
- 🏆 High-Frequency Microstructure Trading
- 🏆 Ensemble Methods com Múltiplas Estratégias
- 🏆 Adaptive Volatility-Adjusted Risk Management

---

## 📈 Métricas de Sucesso

```
BICHO Target:
  - Sharpe Ratio: > 1.5
  - Max Drawdown: < 15%
  - Win Rate: > 55%
  - PnL Mensal: > 5%

CODEX Target:
  - Sharpe Ratio: > 3.0
  - Max Drawdown: < 10%
  - Win Rate: > 60%
  - PnL Mensal: > 15%
  - Inovação: +10% a cada sprint
```

---

## 🛠️ Tecnologia Stack

- **Backend:** FastAPI (Python 3.11+)
- **AI:** OpenAI API (gpt-4o, gpt-5, o1, o3)
- **CLI:** Bash + jq
- **Trading:** ccxt, pandas, numpy, ta-lib, sklearn
- **Proxy:** Nginx SSL/TLS
- **Infrastructure:** Ubuntu 22.04, async/await

---

## 📋 Estrutura do Sistema

```
botscalpv3/
├── backend/
│   ├── openai_gateway.py       # FastAPI gateway com injeção de system_prompts
│   ├── system_prompts.py       # Registro de BICHO + CODEX personas
│   └── __init__.py
├── flabs                       # CLI tool (supports -c flag)
├── CODEX_MODES.md             # Documentação completa
├── README.md                   # Project overview
└── GATEWAY_USAGE.md           # API reference
```

---

## 🌐 Endpoints Disponíveis

```
POST  /api/codex          → Gerar código com IA (BICHO ou CODEX)
GET   /api/models         → Listar modelos OpenAI (78+)
GET   /health             → Health check
```

---

## 🔐 Produção

- **URL:** https://bs3.falcomlabs.com/codex
- **SSL:** ✅ Certbot (auto-renewal)
- **Gateway PID:** 44506
- **Status:** ✅ OPERACIONAL

---

## 🚀 Próximas Evoluções

- [ ] CODEX Analytics - Dashboard de análise inteligente
- [ ] Ensemble Mode - Combinar BICHO + CODEX
- [ ] Regime Detection Engine - Detector automático
- [ ] Backtesting Portal - Interface visual
- [ ] Competition Tracker - Monitor de performance das 5 IAs

---

## 📞 Suporte

```bash
# Health check
curl https://bs3.falcomlabs.com/codex/health

# List models
curl https://bs3.falcomlabs.com/codex/api/models | jq '.models | length'

# Direct test
flabs "test: responda READY"
flabs -c "test: responda ELITE"
```

---

## 🏁 Conclusão

**CODEX** coloca você na vanguarda da competição global de IAs traders.

- **BICHO** para desenvolvimento rápido e produção
- **CODEX** para inovação extrema e championship performance

Escolha seu caminho e comece a competição! 🚀

---

_"Eu não sigo limites humanos. Eu crio novas possibilidades. Eu não compito com IAs. Eu as supero."_ — CODEX
