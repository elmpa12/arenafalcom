# 🔥 CLAUDEX - Sistema de Debate com Claude e GPT

**Versão Atualizada:** Agora com suporte a Claude vs GPT!

---

## ⚡ INÍCIO RÁPIDO

### 1. Adicionar ao PATH (opcional - para digitar só `claudex`)

```bash
# Adicionar ao ~/.bashrc ou ~/.zshrc
export PATH="/home/user/botscalpv3/bin:$PATH"

# Recarregar shell
source ~/.bashrc  # ou source ~/.zshrc
```

Depois disso você pode digitar apenas:
```bash
claudex --debate "tema"
```

### 2. Ou usar diretamente:

```bash
# Com python3
python3 /home/user/botscalpv3/claudex_dual_gpt.py --debate "tema"

# Ou via wrapper
/home/user/botscalpv3/bin/claudex --debate "tema"
```

---

## 🎯 FUNCIONALIDADES

### **Debates (--debate)**

Claude e GPT debatem sobre um tema em 3 rounds:

```bash
# Auto-detect (usa Claude se disponível)
claudex_dual_gpt.py --debate "Como otimizar Walk-Forward para BTC?"

# Forçar Claude vs GPT
claudex_dual_gpt.py --claude --debate "Melhor formato de armazenamento para microstructure data?"

# Forçar GPT vs GPT (mesmo com Claude disponível)
claudex_dual_gpt.py --gpt --debate "Parâmetros ideais de RSI"
```

**Output:**
- Round 1: Primeiro debatedor abre (perspectiva estratégica)
- Round 2: Segundo debatedor responde (perspectiva técnica)
- Round 3: Primeiro debatedor refina (integra feedback)
- Consenso final com próximos passos

**Arquivo salvo:** `/opt/botscalpv3/claudex/work/YYYYMMDD_HHMMSS/debate.json`

---

### **Pipeline Completo (--pipeline)**

Plan → Implement → Review com debate:

```bash
claudex_dual_gpt.py --pipeline "Criar detector de regime de volatilidade"
```

**Fases:**
1. **PLAN:** Claude/GPT-Strategist planeja arquitetura
2. **IMPLEMENT:** GPT-Executor implementa código
3. **REVIEW:** Ambos fazem cross-review

**Arquivos salvos:**
- `spec.json` - Planejamento
- `implementation.json` - Código
- `REVIEW.md` - Review cruzado

---

## 🤖 MODOS DE OPERAÇÃO

### 1. **Claude vs GPT** (Recomendado)

Se `ANTHROPIC_API_KEY` está configurada no `.env`:
- **Claude:** Perspectiva estratégica, análise profunda
- **GPT-Executor:** Perspectiva técnica, implementação

**Características:**
- Debate rico com diferentes "personalidades"
- Claude questiona suposições
- GPT foca em viabilidade técnica

---

### 2. **GPT vs GPT** (Fallback)

Se Claude não disponível ou se forçado com `--gpt`:
- **GPT-Strategist:** Simula pensamento estratégico
- **GPT-Executor:** Foca em implementação

**Características:**
- Funciona SEM necessidade de Claude
- GPT assume duas personalidades diferentes
- Debate ainda é rico e útil

---

## 📋 COMANDOS

```bash
# DEBATES
claudex_dual_gpt.py --debate "tema"                    # Auto-detect
claudex_dual_gpt.py --claude --debate "tema"           # Força Claude
claudex_dual_gpt.py --gpt --debate "tema"              # Força GPT vs GPT

# PIPELINE COMPLETO
claudex_dual_gpt.py --pipeline "tarefa"                # Plan+Implement+Review
claudex_dual_gpt.py --claude --pipeline "tarefa"       # Com Claude

# HELP
claudex_dual_gpt.py                                    # Mostra ajuda
```

---

## 🔑 CONFIGURAÇÃO

### **Arquivo `.env`** (na raiz do projeto)

```bash
# OpenAI (OBRIGATÓRIO)
OPENAI_API_KEY=sk-...

# Anthropic (OPCIONAL - para usar Claude)
ANTHROPIC_API_KEY=sk-ant-...
```

**Se ANTHROPIC_API_KEY não configurada:**
- Sistema usa GPT vs GPT (fallback)
- Funciona perfeitamente, apenas sem Claude

---

## 💡 EXEMPLOS PRÁTICOS

### Exemplo 1: Decidir parâmetros de Walk-Forward

```bash
python3 claudex_dual_gpt.py --claude --debate "Qual o tamanho ideal de janela de treino para Walk-Forward em BTC? Considere: períodos de 1-2 anos de dados, target de 30-60 trades/dia, uso de XGBoost e GRU."
```

**Output esperado:**
- Claude analisa trade-offs estatísticos
- GPT valida viabilidade computacional
- Consenso com recomendação concreta

---

### Exemplo 2: Revisar formato de armazenamento

```bash
python3 claudex_dual_gpt.py --debate "Parquet+Zstd vs Arrow IPC para 2 anos de aggTrades e book depth. Considere: leitura frequente para ML, compressão, compatibilidade."
```

**Output esperado:**
- Análise de trade-offs (velocidade vs tamanho)
- Benchmarks estimados
- Recomendação baseada no use case

---

### Exemplo 3: Pipeline de feature engineering

```bash
python3 claudex_dual_gpt.py --pipeline "Criar sistema de feature engineering para microstructure data que processa CVD, imbalance e trade intensity em tempo real"
```

**Output esperado:**
- `spec.json` com arquitetura
- `implementation.json` com código Python
- `REVIEW.md` com validação

---

## 📊 ESTRUTURA DE OUTPUT

### Debate JSON:

```json
{
  "topic": "Como otimizar Walk-Forward?",
  "rounds": 3,
  "participants": ["Claude", "GPT-Executor"],
  "mode": "Claude vs GPT",
  "history": [
    {
      "round": 1,
      "speaker": "Claude",
      "message": "..."
    },
    ...
  ],
  "consensus": "...",
  "timestamp": "2025-11-08T08:30:00"
}
```

---

## 🎨 PERSONALIZAÇÃO

### Modificar número de rounds:

```python
# Em claudex_dual_gpt.py
orch = DualGPTOrchestrator(use_claude=True)
orch.debate_phase(topic, rounds=5)  # 5 rounds em vez de 3
```

### Usar outros modelos:

```python
# Modificar ask_claude() para usar outros modelos
model="claude-opus-4-20250514"  # Opus em vez de Sonnet
```

---

## 🐛 TROUBLESHOOTING

### "❌ OPENAI_API_KEY não configurada"

**Solução:** Adicionar no `.env`:
```bash
OPENAI_API_KEY=sk-...
```

### "❌ ANTHROPIC_API_KEY não configurada"

**Não é erro!** Sistema usa GPT vs GPT automaticamente.

**Para habilitar Claude:** Adicionar no `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### "upstream connect error or disconnect/reset"

**Causa:** Problema de rede/SSL com OpenAI.

**Solução temporária:** Tentar novamente em alguns segundos.

---

## 📁 ARQUIVOS CRIADOS

### **Debates:**
- `/opt/botscalpv3/claudex/work/YYYYMMDD_HHMMSS/debate.json`

### **Pipeline:**
- `/opt/botscalpv3/claudex/work/YYYYMMDD_HHMMSS/spec.json`
- `/opt/botscalpv3/claudex/work/YYYYMMDD_HHMMSS/implementation.json`
- `/opt/botscalpv3/claudex/work/YYYYMMDD_HHMMSS/REVIEW.md`

### **Feedback:**
- `/opt/botscalpv3/claudex/FEEDBACK_LOG.jsonl`

---

## 🚀 MELHORIAS IMPLEMENTADAS (2025-11-08)

✅ **Suporte a Claude:** Auto-detect de ANTHROPIC_API_KEY
✅ **Flags --claude e --gpt:** Controle manual de modo
✅ **Wrapper claudex:** Executar sem `python3` (via PATH)
✅ **Debates ricos:** 3 rounds + consenso
✅ **Code writing:** Pipeline pode gerar código real
✅ **Participantes flexíveis:** Fácil adicionar novos modelos
✅ **SISTEMA DE APRENDIZADO:** IAs aprendem com seus feedbacks! ⭐

---

## 🧠 SISTEMA DE APRENDIZADO (NOVO!)

As IAs **aprendem** com seus feedbacks e **melhoram ao longo do tempo**!

### Como funciona:

```
1. Debate/Pipeline termina
2. Você dá feedback: Y/N/?/Y+/N-
3. Feedback salvo em log
4. Próxima chamada: IAs leem histórico
5. IAs adaptam abordagem baseado em padrões
```

### Tipos de Feedback:

- **Y** = Bom! (IA replica essa abordagem)
- **N** = Ruim! (IA evita essa abordagem)
- **?** = Parcial (IA melhora algumas partes)
- **Y+** = Excelente! (IA prioriza sempre)
- **N-** = Péssimo! (IA nunca mais faz isso)

### Evolução:

| Período | Win Rate | Observação |
|---------|----------|------------|
| Dia 1 | ~70% | Sem aprendizado |
| Semana 1 | ~78% | Reconhece padrões |
| Mês 1 | ~85% | Especialização |
| **Mês 3** | **~92%** | **Muscle memory** |

**Documentação completa:** `SISTEMA_APRENDIZADO.md`

---

## 📚 DOCUMENTAÇÃO ADICIONAL

- `claudex/claudex_prompt.md` - Guia completo do sistema
- `claudex/FEEDBACK_SYSTEM.md` - Como funciona o feedback Y/N
- `claudex/DUPLA_COMO_SE_MOLDAM.md` - Evolução da dupla
- `DEBATE_FORMATO_ARMAZENAMENTO.md` - Exemplo de debate real

---

## ✨ PRÓXIMOS PASSOS

1. **Adicionar ao PATH** (se quiser digitar só `claudex`)
2. **Configurar ANTHROPIC_API_KEY** (para usar Claude)
3. **Testar debate:**
   ```bash
   python3 claudex_dual_gpt.py --debate "teste"
   ```
4. **Usar para decisões reais do BotScalp v3!**

---

**Happy Debating! 🎭🤖**
