# MAESTRO ARCHITECTURE - Sistema Multi-AI com Feedback em Tempo Real

**Data**: 2025-11-08
**Objetivo**: 500 micro-backtests com aprendizado exponencial através de feedback instantâneo

---

## ARQUITETURA MULTI-AI

### Agentes

| Agente | Modelo | Papel | Responsabilidades |
|--------|--------|-------|-------------------|
| **Maestro** | Claude 1 (eu) | Orquestrador | Coordena sessão, edita código, controla parâmetros, aplica patches |
| **Estrategista** | Claude 2 | Criativo | Propõe métodos, combinações, ajustes de parâmetros |
| **Crítico** | GPT-5 (B) | Analítico | Analisa resultados, identifica padrões, propõe otimizações |

### Fluxo de Comunicação

```
┌─────────────────────────────────────────────────────────────┐
│                    MAESTRO (Claude 1)                        │
│  Orquestra sessão | Edita código | Aplica patches           │
└─────────────────────────────────────────────────────────────┘
                    ▲              │
                    │              │
        ┌──────────────┐    ┌──────────────┐
        │   Feedback   │    │   Commands   │
        └──────────────┘    └──────────────┘
                    │              ▼
    ┌───────────────────────────────────────────┐
    │   STRATEGIST (Claude 2)                    │
    │   Propõe 3 variações: métodos + params    │
    └───────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────┐
    │   CRITIC (GPT-5 B)                         │
    │   Escolhe 1 variação + define métricas    │
    └───────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────┐
    │   EXECUTION ENGINE                         │
    │   Roda backtests em paralelo               │
    │   Emite TradeEvent/ScoreEvent              │
    └───────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────┐
    │   REAL-TIME ANALYZER                       │
    │   Lê logs por micro-episódio (120 barras)  │
    │   Detecta padrões, ajusta automaticamente  │
    └───────────────────────────────────────────┘
```

---

## SESSÃO: 500 MICRO-BACKTESTS

### Configuração

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| **Total Testes** | 500 | Divididos em 5 segmentos de 100 |
| **Timeframe** | 5m | Compromisso sensibilidade/ruído |
| **Janela** | 15 dias (~4300 barras) | Por micro-backtest |
| **Train** | 10 dias | Treinamento ML/estratégias |
| **Validation** | 3 dias | Verificação imediata |
| **Step** | 2 dias | Move janela (novos contextos) |
| **Micro-Episódio** | 120 barras | Mini-partida com feedback |
| **Paralelo** | 16-20 | Aproveita 64 cores |

### Metas

| Métrica | Alvo | Ação se Falhar |
|---------|------|----------------|
| **hit** | >= 0.52 | ↑ atr_stop_mult +0.3 |
| **payoff** | >= 1.25 | ↑ hard_tp_usd +200 |
| **maxdd** | >= -2000 | ↓ exposição (timeout/bars) |
| **blunders/100** | <= 2 | Ajustar gates/ML |
| **vol_bin alta %** | >= 30% | ↓ atr_z_min |

---

## FEEDBACK EM TEMPO REAL

### Eventos Emitidos

#### TradeEvent
```json
{
  "ts": "2025-11-08T14:22:03Z",
  "episode_id": 42,
  "tf": "5m",
  "side": "LONG",
  "reason": "signal",
  "entry": 68950.1,
  "exit": 69020.3,
  "pnl": 45.3,
  "gates": {"cvd": true, "imb": false, "atrz": true, "vhf": true},
  "bars_held": 15,
  "exit_reason": "tp_hit"
}
```

#### ScoreEvent
```json
{
  "episode_id": 42,
  "bars": 120,
  "n_trades": 8,
  "hit": 0.54,
  "payoff": 1.33,
  "total_pnl": 850.2,
  "maxdd": -1800,
  "blunders": 1,
  "score": 97.5,
  "vol_regime": "high"
}
```

#### PlanEvent
```json
{
  "episode_id": 42,
  "agent": "strategist",
  "action": "adjust_params",
  "change": {"atr_stop_mult": "1.5 -> 1.8"},
  "reason": "hit_below_target"
}
```

### Feedback Loop

```
Micro-Episódio (120 barras)
    ↓
TradeEvents + ScoreEvent emitidos
    ↓
GPT-5 (Critic) lê logs em tempo real
    ↓
Detecta padrão (ex: hit < 0.48)
    ↓
Emite PlanEvent com ajuste
    ↓
Maestro aplica patch
    ↓
Próximo episódio roda com ajustes
```

**Velocidade**: Feedback a cada ~10 minutos de dados (120 barras @ 5m)

---

## CICLO DE EXECUÇÃO (5 Segmentos x 100 Testes)

### Segmento (1 hora)

1. **Claude Maestro**
   - Define escopo (TF, ranges, N backtests)
   - Cria pasta `session_<timestamp>/segment_<n>/`
   - Inicializa logging real-time

2. **Claude 2 (Estrategista)**
   - Propõe 3 variações:
     ```json
     {
       "proposal_1": {"methods": ["macd_trend"], "params": {"atr_stop": 1.5, "hard_tp": 200}},
       "proposal_2": {"methods": ["ema_crossover"], "params": {"atr_stop": 2.0, "hard_tp": 400}},
       "proposal_3": {"methods": ["trend_breakout"], "params": {"atr_stop": 1.2, "hard_tp": 150}}
     }
     ```

3. **GPT-5 (Crítico)**
   - Escolhe 1 variação
   - Define métricas e teste rápido:
     ```json
     {
       "chosen": "proposal_1",
       "metricas": {"hit": ">=0.52", "payoff": ">=1.25", "maxdd": ">=-2000"},
       "teste_rapido": "100 microbacktests, feedback por episódio"
     }
     ```

4. **Execução Automática**
   - 100 backtests paralelos (16 simultâneos)
   - Cada backtest = múltiplas mini-partidas (120 barras)
   - ScoreEvent emitido a cada mini-partida

5. **GPT-5 lê logs durante execução**
   - A cada 10 mini-partidas, analisa
   - Se degradação (hit < 0.48), sinal de ajuste imediato

6. **Claude 2 gera ajustes**
   - Incrementais ou mata variações ruins

7. **Maestro aplica patches**
   - Reinicia próxima rodada com melhorias

### A Cada 2 Segmentos

**Entrega plano de 1 página**:
- Rotinas estabelecidas
- DRIs (Directly Responsible Individuals) por métrica
- Checagens de saúde do sistema

---

## ESTRUTURA DE ARQUIVOS

```
/sessions/
 ├─ session_2025-11-08_1200/
 │   ├─ session_config.json
 │   ├─ segment_1/
 │   │   ├─ segment_plan.json         # Plano dos 100 testes
 │   │   ├─ segment_results.json      # Resultados agregados
 │   │   ├─ seg1_test001/
 │   │   │   ├─ leaderboard_base.csv
 │   │   │   ├─ trades.jsonl          # TradeEvents stream
 │   │   │   ├─ scores.jsonl          # ScoreEvents stream
 │   │   │   └─ test.log
 │   │   ├─ seg1_test002/
 │   │   ├─ ...
 │   │   └─ seg1_test100/
 │   ├─ segment_2/
 │   │   ├─ ...
 │   ├─ segment_3/
 │   ├─ segment_4/
 │   ├─ segment_5/
 │   ├─ session_summary.md            # Relatório final
 │   └─ metrics_global.json           # Agregados da sessão
```

---

## INTEGRAÇÃO COM APRENDIZADOS GEN 2/3

### Baseline Inteligente

Ao invés de começar com parâmetros aleatórios, **usar descobertas Gen 2/3**:

**Configurações Promissoras** (de Gen 2/3):
```python
WINNING_CONFIGS = [
    # gen3_15m_macd: +277K, Sharpe +0.84
    {"tf": "15m", "method": "macd_trend", "period": "Jan 1-15"},

    # gen3_15m_trend: +194K, Sharpe +1.13
    {"tf": "15m", "method": "trend_breakout", "period": "Jan 1-15"},

    # rapid_feb_w2_macd: +259K, Sharpe +0.95
    {"tf": "1m", "method": "macd_trend", "period": "Feb week 2"},

    # rapid_w4_ema: +160K, Sharpe +0.81
    {"tf": "1m", "method": "ema_crossover", "period": "Jan week 4"}
]
```

**Estratégia Maestro**:
1. Segmento 1-2: Explorar configurações vencedoras Gen 2/3 com pequenas variações
2. Segmento 3: Testar novos métodos baseados em insights
3. Segmento 4: Walk-forward validation das melhores
4. Segmento 5: Ensemble e otimização final

---

## AUTO-AJUSTES AUTOMÁTICOS

### Regras de Ajuste

```python
ADJUSTMENT_RULES = {
    "hit_low": {
        "condition": "avg_hit < 0.48",
        "action": "atr_stop_mult += 0.3",
        "reason": "Stops muito apertados, aumentando"
    },

    "payoff_low": {
        "condition": "avg_payoff < 1.1",
        "action": "hard_tp_usd += 200",
        "reason": "Alvos muito próximos, expandindo"
    },

    "maxdd_high": {
        "condition": "avg_maxdd < -2000",
        "action": "timeout_bars -= 60",
        "reason": "Exposição excessiva, reduzindo"
    },

    "blunders_high": {
        "condition": "blunders_per_100 > 2",
        "action": "tighten_gates",
        "reason": "Muitos erros, fortalecer filtros"
    }
}
```

### Exemplo de Patch Automático

```python
# Se hit < 0.48 em 3 episódios consecutivos
if episodes_below_hit >= 3:
    current_mult = config["atr_stop_mult"]
    new_mult = current_mult + 0.3

    patch = {
        "parameter": "atr_stop_mult",
        "old_value": current_mult,
        "new_value": new_mult,
        "reason": "hit_below_target_3x",
        "applied_at": "episode_42"
    }

    # Maestro aplica
    apply_patch(patch)

    # Log PlanEvent
    log_plan_event({
        "episode_id": 42,
        "agent": "maestro",
        "action": "auto_adjust",
        "patch": patch
    })
```

---

## FORMATO DE MENSAGENS (JSON)

### Claude 2 (Estrategista) → Maestro

```json
{
  "agente": "estrategista",
  "turno": 1,
  "proposta": "testar_macd_15m_agressivo",
  "plano_curto": [
    "Usar 15m timeframe (melhor que 5m em Gen3)",
    "macd_trend com atr_stop_mult=2.0",
    "hard_tp_usd=400 para maior payoff"
  ],
  "metricas_esperadas": {"hit": 0.48, "payoff": 1.4, "sharpe": 0.8},
  "proximo_passo": "executar_100_testes_segmento_1"
}
```

### GPT-5 (Crítico) → Maestro

```json
{
  "agente": "critico",
  "turno": 1,
  "veredito": "aprovado_com_ajustes",
  "analise": {
    "hit_atual": 0.46,
    "payoff_atual": 1.31,
    "problema": "hit abaixo do alvo por 0.02"
  },
  "ajuste_proposto": {
    "parametro": "atr_stop_mult",
    "valor_atual": 2.0,
    "valor_novo": 2.3,
    "razao": "ampliar stops para aumentar hit"
  },
  "proximo_passo": "aplicar_ajuste_e_continuar"
}
```

### Maestro → Ambos

```json
{
  "agente": "maestro",
  "turno": 2,
  "acao": "patch_aplicado",
  "detalhes": {
    "patch": {"atr_stop_mult": "2.0 -> 2.3"},
    "aplicado_em": "seg1_test051-100"
  },
  "metricas_pos_patch": {"hit": 0.51, "payoff": 1.28},
  "status": "meta_hit_atingida",
  "proximo_passo": "prosseguir_segmento_2"
}
```

---

## IMPLEMENTAÇÃO

### Arquivos Criados

1. **maestro_session.py** - Orquestrador principal (completo)
2. **pilot_maestro.py** - Teste piloto 10 backtests (validado ✅)

### Próximos Passos

3. **Create real-time logger**:
   ```python
   # real_time_logger.py
   # Emite TradeEvent, ScoreEvent, PlanEvent
   ```

4. **Create multi-AI protocol**:
   ```python
   # multi_ai_protocol.py
   # Gerencia comunicação Claude 2 ↔ GPT-5 ↔ Maestro
   ```

5. **Create auto-adjustment engine**:
   ```python
   # auto_adjuster.py
   # Aplica patches baseados em regras + ML
   ```

6. **Run Segment 1** (100 testes):
   ```bash
   python3 maestro_session.py --segment 1
   ```

7. **Analyze + Iterate**:
   - Após cada segmento, gerar relatório
   - Ajustar próximo segmento baseado em feedback

---

## TIMELINE ESTIMADO

| Fase | Duração | Atividade |
|------|---------|-----------|
| **Setup** | 10 min | Criar loggers + protocol |
| **Segment 1** | 1h | 100 testes (15 dias cada, 5m) |
| **Análise 1** | 15 min | Maestro + Strategist + Critic |
| **Segment 2** | 1h | 100 testes (ajustados) |
| **Análise 2** | 15 min | Plano de 1 página |
| **Segment 3** | 1h | 100 testes |
| **Segment 4** | 1h | 100 testes |
| **Segment 5** | 1h | 100 testes |
| **Final** | 30 min | Session summary + melhores configs |
| **TOTAL** | **~6 horas** | 500 micro-backtests completos |

---

## SUCESSO ESPERADO

Baseado em Gen 2/3:
- **Gen 2**: 10% profitable (3/30)
- **Gen 3**: 14.3% profitable (3/21)
- **Com baseline inteligente**: **>20% profitable esperado**

Se 500 testes → **100+ configurações lucrativas** identificadas!

---

**SISTEMA PRONTO PARA EXECUÇÃO** 🎭

Próximo: Implementar loggers + protocol e rodar Segment 1!
