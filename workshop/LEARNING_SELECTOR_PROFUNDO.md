# APRENDIZADO PROFUNDO - SELECTOR21.PY

**Data**: 2025-11-08
**Iteração**: 1
**Status**: As IAs ERRARAM - não estudaram funções de output corretamente

---

## ❌ ERROS IDENTIFICADOS NESTA ITERAÇÃO

### 1. Argumentos de Output Faltando
- ❌ **--loader_verbose**: Não usamos, por isso não vimos o carregamento de dados
- ❌ **--out_best_trades**: Configurado mas pode ter path errado
- ❌ **--out_leaderboard_***: Não verificamos se os CSVs seriam gerados
- ⚠️ **--print_top10**: Usado, mas sem efeito porque outros args faltaram

### 2. Falta de Estudo de Todas as Funções
As IAs estudaram os **parâmetros** mas NÃO estudaram:
- Fluxo de execução do main()
- Funções de geração de leaderboards
- Funções de output e print
- Condições que bloqueiam outputs

---

## 🎯 PLANO DE ESTUDO PARA PRÓXIMA ITERAÇÃO

### FASE 1: Estrutura Completa do Código
```python
# As IAs DEVEM estudar:
1. def main() linha por linha (linha ~3112 até ~4835)
2. Todas as funções de output/save
3. Condições que controlam geração de CSVs
4. Flags de debug/verbose
```

### FASE 2: Argumentos de Output (CRÍTICOS)
```bash
# Argumentos que DEVEM ser incluídos:
--loader_verbose          # Ver carregamento de dados
--print_top10             # Resumo no final
--out_root ./resultados   # Raiz dos outputs ✓ (já temos)

# Argumentos que DEVEM ter paths corretos:
--out_wf_base             # Leaderboard base WF
--out_wf_combos           # Leaderboard combos WF
--out_wf_all              # Leaderboard tudo WF
--out_wf_trades           # Trades best WF
--out_leaderboard_base    # Leaderboard base full
--out_leaderboard_combos  # Leaderboard combos full
--out_leaderboard_all     # Leaderboard all full
--out_best_trades         # Best trades full
--out_report              # JSON report
--out_runtime             # Runtime config
--out_wf_report           # WF report
--out_wf_ml               # WF ML results
```

### FASE 3: Funções Críticas a Estudar
```python
# Funções que AS IAs DEVEM LER COMPLETAMENTE:

1. fast_read_klines_monthly()      # Como carrega dados
2. enrich_with_all_features()      # Como enriquece features
3. run_strategy_single()           # Como executa uma estratégia
4. build_combos()                  # Como gera combos
5. run_walkforward()               # Como faz walk-forward
6. run_ml_pipeline()               # Como treina ML
7. save_leaderboards()             # Como salva resultados
8. print_summary()                 # Como imprime resumo
9. compute_metrics()               # Quais métricas calcula
10. filter_strategies()            # Como filtra por min_trades, min_sharpe, etc.
```

### FASE 4: Debugging Next Run
```bash
# Para próxima iteração, TESTAR PRIMEIRO com:
--smoke_months 1              # Apenas 1 mês para teste rápido
--loader_verbose              # Ver carregamento
--print_top10                 # Ver resumo
--combo_cap 10                # Apenas 10 combos (teste)
--run_base                    # Só estratégias base primeiro
# (NÃO rodar combos/ML até base funcionar)
```

---

## 📝 PRÓXIMA ITERAÇÃO: COMANDOS PROGRESSIVOS

### COMANDO 1: TESTE MÍNIMO (5-10min)
```bash
python3 selector21.py \
  --umcsv_root ./data_monthly \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2024-02-01 \
  --smoke_months 1 \
  --interval auto \
  --exec_rules '1m' \
  --methods 'trend_breakout,rsi_reversion,ema_crossover' \
  --run_base \
  --loader_verbose \
  --print_top10 \
  --out_root ./resultados
```

### COMANDO 2: BASE COMPLETO (15-30min)
Só depois do COMANDO 1 funcionar:
```bash
# Adicionar:
--exec_rules '1m,5m,15m'
--methods 'all'
--walkforward
--wf_train_months 3
--wf_val_months 1
```

### COMANDO 3: COMBOS (30-60min)
Só depois do COMANDO 2 funcionar:
```bash
# Adicionar:
--run_combos
--combo_ops 'AND,MAJ'
--combo_cap 50
```

### COMANDO 4: FULL (60-120min)
Só depois do COMANDO 3 funcionar:
```bash
# Adicionar:
--combo_cap 400
--run_ml
--ml_model_kind auto
```

---

## 🤖 TAREFA PARA AS IAs

**ANTES de decidir QUALQUER parâmetro:**

1. ✅ Ler selector21.py COMPLETO linha por linha
2. ✅ Mapear TODAS as funções principais
3. ✅ Entender fluxo de execução do main()
4. ✅ Identificar TODAS as condições de output
5. ✅ Testar PROGRESSIVAMENTE (mínimo → completo)

**NÃO PULAR ETAPAS!**

---

## 📊 MÉTRICAS DE SUCESSO

Para considerar que as IAs APRENDERAM:

- [ ] Comando TESTE MÍNIMO gera CSVs com estratégias
- [ ] Comando BASE COMPLETO gera leaderboards WF
- [ ] Comando COMBOS gera combos e ranqueia
- [ ] Comando FULL gera ML e consensus
- [ ] Análise pós-run identifica melhorias específicas
- [ ] Próxima iteração melhora métricas (Sharpe, Win%, DD)

---

## 🔄 APRENDIZADO INCREMENTAL

Este documento será atualizado a cada iteração com:
- Novos erros descobertos
- Funções mapeadas
- Parâmetros otimizados
- Resultados comparativos

**Objetivo**: Em 3-5 iterações, as IAs dominam o selector21 completamente.

---

_Atualizado após Iteração 1 - Primeira execução com dados consolidados_
