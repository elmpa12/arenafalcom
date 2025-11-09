# ⚡ QUICK START - Auto Evolution System

**Última atualização:** 2025-11-08T13:30:00Z
**Status:** 🟢 PRONTO PARA USO

---

## 🎯 O QUE ESTÁ PRONTO

✅ **Auto Evolution System** (Claude + GPT analisam tudo automaticamente)
✅ **3 Modos:** review / interactive / auto
✅ **Integração com Backtests** (wrapper transparente)
✅ **Integração com Selector21** (PRESERVA todos ML: XGBoost, RF, LogReg, Ensemble)
✅ **Roadmap 4 Fases:** Backtests → DL → Paper → Real

---

## 🚀 RODAR AGORA (1 comando)

```bash
python3 -c "
from selector21_auto_evolution import run_selector_with_evolution

run_selector_with_evolution(
    symbol='BTCUSDT',
    start='2024-01-01',
    end='2024-06-01',
    apply_mode='review'
)
"
```

**O que acontece:**
1. Selector21 executa walk-forward
2. XGBoost, RandomForest, LogReg treinam
3. Backtests rodam
4. **Claude + GPT analisam AUTOMATICAMENTE**
5. Logs salvos em `claudex/LEARNING_LOG.jsonl`

---

## 📊 3 MODOS DISPONÍVEIS

| Modo | O que faz | Quando usar |
|------|-----------|-------------|
| **review** | Apenas propõe mudanças | Produção (padrão) |
| **interactive** | Pergunta antes de aplicar | Validação |
| **auto** | Aplica tudo automaticamente | Testes controlados |

---

## 📁 ARQUIVOS IMPORTANTES

**Core:**
- `auto_evolution_system.py` - Sistema principal
- `backtest_integration.py` - Wrapper genérico
- `selector21_auto_evolution.py` - Integração selector21

**Docs:**
- `SESSION_RECOVERY.md` - Guia de recuperação
- `INTEGRATION_SUMMARY.md` - Resumo completo
- `evolution_roadmap.json` - Roadmap 4 fases

**Logs:**
- `claudex/LEARNING_LOG.jsonl` - Aprendizados
- `claudex/CODE_CHANGES_LOG.jsonl` - Mudanças propostas

**Storage:**
- `.session_storage.json` - Estado completo

---

## 🎯 ROADMAP DE EVOLUÇÃO

```
FASE 1: BACKTESTS EXIGENTES (EM PROGRESSO) ← VOCÊ ESTÁ AQUI
  ↓
  • Objetivo: 70%+ win rate, Sharpe > 1.5
  • Modelos: XGBoost, RF, LogReg, Ensemble (PRESERVADOS!)
  • Claude + GPT analisam cada backtest
  
FASE 2: DEEP LEARNING (PENDENTE)
  ↓
  • Modelos: GRU, TCN, Transformers
  • Integrar com dl_heads_v8.py
  • IAs aprendem com DL
  
FASE 3: PAPER TRADING (PENDENTE)
  ↓
  • Tempo real sem risco
  • Sistema aprende com mercado ao vivo
  
FASE 4: REAL TRADING (PENDENTE)
  ↓
  • Produção
  • Evolução contínua
```

---

## 💡 EXEMPLOS RÁPIDOS

### Exemplo 1: Backtest com auto-evolution (review mode)
```python
from selector21_auto_evolution import run_selector_with_evolution

run_selector_with_evolution(
    symbol="BTCUSDT",
    start="2024-01-01",
    end="2024-06-01",
    apply_mode="review",
)
```

### Exemplo 2: Modo Interactive (você decide o que aplicar)
```python
from selector21_auto_evolution import run_selector_with_evolution

run_selector_with_evolution(
    symbol="BTCUSDT",
    start="2024-01-01",
    end="2024-06-01",
    apply_mode="interactive",  # Pergunta antes de aplicar
)
```

### Exemplo 3: Ver logs
```bash
tail -10 claudex/LEARNING_LOG.jsonl | python3 -m json.tool
```

---

## 🔑 COMANDOS ESSENCIAIS

```bash
# Ativar venv
source .venv/bin/activate

# Ver estado atual
cat .session_storage.json | jq '.current_context'

# Ver roadmap
cat evolution_roadmap.json | jq '.'

# Ver últimos aprendizados
tail -5 claudex/LEARNING_LOG.jsonl
```

---

## ✅ CHECKLIST DE RECUPERAÇÃO

Se a sessão foi perdida (auto-compact):

1. ✅ Ler `.session_storage.json` (estado completo)
2. ✅ Ler `SESSION_RECOVERY.md` (guia detalhado)
3. ✅ Ler `QUICK_START.md` (este arquivo)
4. ✅ Rodar comando acima (selector21 com auto-evolution)
5. ✅ Ver logs gerados

---

**Tudo está salvo e pronto!** 🚀
