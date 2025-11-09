# 🚀 TRIPLE-AI QUICKSTART

**Seu sistema dual-AI agora é TRIPLE-AI com Claude Code iterativo!**

## ⚡ 30 segundos para começar

```bash
cd /opt/botscalpv3

# 1. Setup APIs
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY="sk-proj-..."
#   ANTHROPIC_API_KEY="sk-ant-..."

source load_env.sh

# 2. Teste rápido
flabs --help

# 3. Execute primeira feature
flabs --pipeline "new regime detector com Kalman filter"
```

✅ Pronto! Sistema completo operacional.

---

## 📋 Três modos principais

### 1. **PLAN** — Claude planejador (cria spec)

```bash
flabs --plan "detector de regime de alta frequência com machine learning"
```

**Saída:** `spec.md` estruturado com:
- Objetivo claro
- Arquitetura técnica
- Exemplos de código
- Critérios de aceitação
- Próximos passos

```markdown
# Spec: Regime Detector ML

## Objetivo
Detectar regime de volatilidade (low/normal/high) em <100ms

## Arquitetura
- Input: OHLCV stream (Binance)
- Indicators: Bollinger, ATR, RSI
- ML: RandomForest + Kalman
- Output: regime signal + confidence

## Exemplos
```python
detector = RegimeDetector()
regime = detector.predict(ohlcv)  # 'low', 'normal', 'high'
```

## Critérios de Aceitação
- Latência <100ms
- Accuracy >85%
- Backtested >1000 trades
```

### 2. **ITERATE** — Claude Code iterativo (refina spec)

```bash
flabs --iterate "adiciona Kalman filter para suavização"
```

**O que acontece:**
- Claude Code abre no terminal (modo interativo)
- Você digita feedback em tempo real
- Claude **edita spec.md** automaticamente
- Roda testes inline
- Refina até estar perfeito

```
claude> Analisando spec.md...
claude> Usuário quer Kalman filter
claude> Atualizando exemplos de código...
claude> Validando spec contra backtesting requirements...
claude> ✅ Spec refinada. Pronto para BUILD?
```

Responda sim/não, adicione feedback, iterate!

### 3. **BUILD** — Codex executor (implementa)

```bash
flabs --build spec.md
```

**Codex gera:**
- ✅ `implementation.py` — código championship-grade
  - Type hints completos
  - Docstrings detalhadas
  - Otimizado para microsegundos
- ✅ `tests.py` — pytest com >90% cobertura
- ✅ `REVIEW.md` — auto-review de qualidade

```python
# implementation.py — production-ready

import numpy as np
from dataclasses import dataclass

@dataclass
class RegimeSignal:
    regime: str      # 'low' | 'normal' | 'high'
    confidence: float
    timestamp: float

class RegimeDetector:
    """Detecta regime de volatilidade em <100ms
    
    Combina indicadores técnicos + ML + Kalman filter.
    Otimizado para high-frequency trading.
    """
    
    def __init__(self):
        self.kalman = KalmanFilter(...)
        self.model = load_model('regime_rf.pkl')
    
    def predict(self, ohlcv: np.ndarray) -> RegimeSignal:
        """Predição em <100ms garantido"""
        features = self._extract_features(ohlcv)
        smoothed = self.kalman.filter(features)
        prob = self.model.predict_proba(smoothed)
        regime = self._to_regime(prob)
        return RegimeSignal(regime, prob.max(), time.time())
```

### 4. **REVIEW** — Cross-análise (valida)

```bash
flabs --review implementation.py
```

**Ambas IAs analisam:**

Claude (conceitual):
- ✅ Alinhamento com spec
- ✅ Arquitetura faz sentido?
- ✅ Escolhas de design justificadas?

Codex (técnico):
- ✅ Performance <100ms?
- ✅ Type hints corretos?
- ✅ Testes cobrem edge cases?

**Output:** `REVIEW.md` com issues bloqueadores vs. nice-to-have

```markdown
# REVIEW: Regime Detector

## ✅ Conceitual (Claude)
- Spec bem compreendida
- Kalman filter placement correto
- ML model adequado

## ✅ Técnico (Codex)
- Microsegundos garantidos ✓
- Type hints 100% ✓
- Tests >90% ✓

## Issues
### 🔴 Bloqueador
- [ ] Kalman initialization precisa validação

### 🟡 Nice-to-have
- [ ] Add logging structurado em JSON
```

---

## 🚀 FULL AUTOMATION — --pipeline

Quer executar TUDO em uma linha?

```bash
flabs --pipeline "new high-frequency market making algo"
```

Isso executa:
1. **PLAN** — Claude cria spec
2. **BUILD** — Codex implementa (pula interatividade)
3. **REVIEW** — Cross-valida

**Saída:** 4 arquivos prontos
```
spec.md                  ← Planejamento (Claude)
implementation.py        ← Código (Codex)
tests.py                ← Testes (Codex)
REVIEW.md               ← Validação (ambos)
```

---

## 📊 Exemplo completo (passo a passo)

```bash
# 1. Planejador cria spec
$ flabs --plan "regime detector com Kalman filter"
📋 PLAN MODE
🧠 Claude criando spec...
✅ SPEC criada em spec.md

# Usuário revisita spec.md, adiciona detalhes

# 2. Iteração em tempo real
$ flabs --iterate "adiciona alertas quando muda regime"
🔄 ITERATE MODE
🧠 Claude Code abrindo terminal...

claude> spec.md detectado. Entendo o contexto.
claude> Adicionando sistema de alertas...
claude> ✅ spec.md atualizado

# User: "Quer que o alert vá para Slack?"
claude> Excelente ideia! Adicionando Slack integration...
claude> # TODO: Slack webhook
claude> Pronto? (y/n)

# User: y

# 3. Executor implementa
$ flabs --build spec.md
🚀 BUILD MODE
🔥 Codex implementando...
✅ implementation.py criado (386 linhas)
✅ tests.py criado (124 linhas)
✅ REVIEW.md criado

# 4. Cross-review
$ flabs --review implementation.py
🔍 REVIEW MODE
🧠 Claude faz review conceitual...
🔥 Codex faz review técnico...
✅ REVIEW.md finalizado

# 5. Git automation
$ git add -A && git commit -m "feat: Add regime detector with Kalman filter"
```

---

## 🎯 Quando usar cada modo

| Modo | Uso | Tempo | Output |
|------|-----|-------|--------|
| **--plan** | Primeira vez, specs complexas | 2-5 min | spec.md |
| **--iterate** | Feedback, refinamentos | 5-15 min | spec.md (refined) |
| **--build** | Implementação pura | 5-10 min | impl.py + tests.py |
| **--review** | Validação pré-deploy | 3-5 min | REVIEW.md |
| **--pipeline** | Automação total | 15-30 min | Tudo junto |

---

## 🔑 API Keys Setup

```bash
cd /opt/botscalpv3

# Copy template
cp .env.example .env

# Edit .env (use seu editor favorito)
nano .env
# ou
vim .env

# Coloque as chaves:
OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxx"
ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxx"

# Load environment
source load_env.sh

# Verificar
echo "✅ OpenAI key: ${OPENAI_API_KEY:0:10}..."
echo "✅ Anthropic key: ${ANTHROPIC_API_KEY:0:10}..."
```

---

## 🆘 Troubleshooting

### Claude Code não encontrado

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

### API keys não funcionando

```bash
# Verificar se .env existe
ls -la .env

# Verificar se load_env.sh funciona
source load_env.sh
echo $ANTHROPIC_API_KEY
```

### Flabs não executa

```bash
# Sintaxe OK?
bash -n /opt/botscalpv3/flabs

# Executável?
chmod +x /opt/botscalpv3/flabs

# Teste direto
bash /opt/botscalpv3/flabs --help
```

---

## 📚 Próximas leituras

1. **AGENTS.md** — arquitetura completa
2. **.claude-config.json** — configuration reference
3. **JOURNAL.txt** — histórico de decisões
4. **PIB_v1.md** — briefing de produto

---

## 💡 Pro Tips

### 1. Encadear operações

```bash
# Primeiro plan
flabs --plan "new detector"

# Depois iterate
flabs --iterate "adiciona ML"

# Depois build
flabs --build spec.md

# Depois review
flabs --review implementation.py

# Depois commit
git add -A && git commit -m "feat: new detector with ML"
```

### 2. Trabalhar em branches

```bash
# Cria branch feature
git checkout -b feature/regime-detector

# Executa pipeline
flabs --pipeline "regime detector"

# Commit + push
git add -A && git commit -m "feat: add regime detector"
git push origin feature/regime-detector

# flabs end. para finalizar
flabs end.
```

### 3. Salvar specs no Git

```bash
git add spec.md implementation.py tests.py REVIEW.md
git commit -m "docs: add regime detector spec + implementation"
```

---

## 🎓 Filosofia

```
┌─────────────────────────────────────────┐
│    PLANEJADOR                           │ Claude (200K context)
│    ├─ Lê TUDO                           │ Tipo: Estrategista
│    ├─ Cria vision                       │ Temp: 0.3 (precise)
│    └─ Define spec perfeito              │
│           ↓                              │
│    ITERADOR (Claude Code)               │ Modo interativo
│    ├─ User feedback em tempo real       │ Tipo: Refinador
│    ├─ Edita arquivos inline             │ Temp: 0.4 (creative)
│    └─ Roda testes ad-hoc                │ MCP: file, git, bash
│           ↓                              │
│    EXECUTOR (Codex)                     │ OpenAI GPT-5 Codex
│    ├─ Implementa spec 100%              │ Tipo: Engenheiro
│    ├─ Championship-grade code           │ Temp: 0.2 (precise)
│    └─ Testes automáticos                │ Focus: microseconds
│           ↓                              │
│    VALIDADOR (Ambos)                    │ Cross-analysis
│    ├─ Claude: conceitual OK?            │ Tipo: Críticos
│    ├─ Codex: técnico OK?                │ Temp: 0.2 (precise)
│    └─ Issues bloqueadores               │
│                                          │
│    RESULTADO: Código perfeito            │
│    ✅ Planejado bem                      │
│    ✅ Iterado com feedback              │
│    ✅ Executado elite                    │
│    ✅ Validado cruzado                   │
└─────────────────────────────────────────┘
```

**Você não escreve código.** Você diz o que quer em inglês. O sistema entrega código perfeito.

---

**Pronto? Let's go! 🚀**

```bash
source load_env.sh
flabs --pipeline "seu ideia aqui"
```

---

**Last updated:** 2025-11-08  
**System:** Triple-AI (Claude Planner + Claude Code Iterative + Codex Executor)  
**Status:** ✅ Ready to ship
