# 🤖 ESTRATÉGIA ML/DL - BOTSCALP V3
**Data**: 2025-11-08
**Status**: Definição de Arquitetura

---

## 📋 SITUAÇÃO ATUAL

### Componentes ML/DL Identificados

#### 1. **Módulos de Treinamento**
- `dl_heads_v8.py` - Sistema principal de DL (GRU, LSTM)
- `treinar_dl.py` - Script de treinamento
- `validar_dl.py` - Validação de modelos
- `gerar_dataset_dl.py` - Preparação de dados
- `model_signal_generator.py` - Gerador de sinais baseado em modelos

#### 2. **Orquestrador**
- `orchestrator.py` - Coordena treinamento local e remoto (GPU)
- Suporta treinamento distribuído em GPUs AWS
- Walk-forward optimization integrado

#### 3. **Status dos Processos em Execução**
- Múltiplos treinamentos rodando em paralelo (GRU/LSTM)
- GPUs remotas sendo utilizadas (54.172.227.79, 34.226.219.16)
- Problemas de dados em alguns processos (falta de klines)

---

## 🎯 OBJETIVOS DA INTEGRAÇÃO

### Curto Prazo (1-2 semanas)
1. **Consolidar Pipeline de Dados**
   - Unificar fontes de dados (klines, trades, features)
   - Padronizar formato de entrada/saída
   - Resolver problemas de path nos scripts

2. **Integrar DL com Trading**
   - Conectar saída dos modelos DL com selector21.py
   - Criar sistema de scoring híbrido (técnico + DL)

### Médio Prazo (3-4 semanas)
1. **Otimização de Modelos**
   - Fine-tuning de hiperparâmetros
   - Ensemble de múltiplos modelos
   - Feature engineering avançado

2. **Sistema de Produção**
   - Pipeline automatizado de retreinamento
   - Monitoramento de drift
   - A/B testing de modelos

---

## 🏗️ ARQUITETURA PROPOSTA

### Camada 1: Data Pipeline
```
Binance Data → Klines/Trades → Feature Engineering → Normalized Dataset
                                       ↓
                              [OHLCV, Volume Profile, Microstructure]
```

### Camada 2: Modelo Híbrido
```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ Technical Setup │     │  DL Models   │     │   Ensemble  │
│   (Selector21)  │────▶│  (GRU/LSTM)  │────▶│   Scorer    │
│  Win Rate: 75%  │     │ Prediction   │     │ Confidence  │
└─────────────────┘     └──────────────┘     └─────────────┘
                              ↓
                        Final Signal
                    (threshold > 0.7)
```

### Camada 3: Execução
```
Signal → Risk Management → Order Execution → Paper Trading → Live Trading
             ↓                    ↓                ↓
      Position Sizing      Slippage Model    Performance
```

---

## 📊 MODELOS E FEATURES

### Features de Entrada (60+ dimensões)
```python
# Preço e Volume
- OHLCV básico (5 features)
- Returns e volatilidade (10 features)
- Volume profile (5 features)

# Indicadores Técnicos
- EMAs múltiplos (12, 26, 50, 200)
- MACD e Signal
- RSI, Stochastic
- Bollinger Bands
- Keltner Channels

# Microestrutura
- Order flow imbalance
- Bid-ask spread
- Trade intensity
- Volume-weighted metrics

# Features Temporais
- Hour of day, day of week
- Rolling statistics (1h, 4h, 24h)
```

### Modelos Implementados
1. **GRU (Gated Recurrent Unit)**
   - Horizon: 3-15 candles
   - Lags: 30-60 períodos
   - Batch: 256-2048

2. **LSTM (Long Short-Term Memory)**
   - Configuração similar ao GRU
   - Melhor para dependências longas

3. **Ensemble** (Planejado)
   - Voting classifier
   - Stacking com meta-learner

---

## 🔧 IMPLEMENTAÇÃO PRÁTICA

### Fase 1: Correção e Padronização ✅
```bash
# Verificar e corrigir paths de dados
python3 dl_heads_v8.py \
    --data_file data/btc_5m.parquet \
    --tf 5m \
    --models gru \
    --device cuda
```

### Fase 2: Integração com Trading
```python
# selector21.py modificado
class DLEnhancedSelector:
    def __init__(self):
        self.base_selector = Selector21()
        self.dl_model = load_dl_model('models/best_gru.pkl')

    def get_signal(self, df):
        # Sinal base (técnico)
        base_signal = self.base_selector.get_signal(df)

        # Previsão DL
        dl_pred = self.dl_model.predict(df)

        # Ensemble
        confidence = 0.6 * base_signal + 0.4 * dl_pred

        return confidence > 0.7
```

### Fase 3: Walk-Forward Optimization
```bash
# Treinar com walk-forward
python3 orchestrator.py \
    --symbol BTCUSDT \
    --dl_models gru,lstm \
    --walkforward \
    --wf_train_months 3 \
    --wf_val_months 1 \
    --wf_step_months 1
```

---

## 📈 MÉTRICAS DE SUCESSO

### KPIs Principais
- **Win Rate**: 75% → 80-85% (meta)
- **Sharpe Ratio**: > 2.0
- **Max Drawdown**: < 10%
- **Profit Factor**: > 1.8
- **Recovery Time**: < 24h

### Monitoramento
- Dashboard em tempo real (visual/)
- Alertas de performance degradada
- Logs detalhados de todas as decisões

---

## 🚀 PRÓXIMOS PASSOS

### Imediato (Esta Semana)
1. [ ] Resolver problemas de dados nos treinamentos atuais
2. [ ] Documentar resultados dos modelos já treinados
3. [ ] Criar script de validação unificado

### Próxima Sprint
1. [ ] Implementar DLEnhancedSelector
2. [ ] Criar pipeline de feature engineering
3. [ ] Configurar retreinamento automático

### Roadmap Q1 2025
1. [ ] Deploy em produção com paper trading
2. [ ] A/B testing entre modelos
3. [ ] Otimização de latência para HFT

---

## 🔐 CONSIDERAÇÕES DE SEGURANÇA

- Modelos salvos com versionamento
- Rollback automático se performance < baseline
- Limites rígidos de risco por modelo
- Auditoria de todas as decisões

---

## 📚 RECURSOS

### Documentação
- `/DL_INTEGRATION_PROPOSAL.md` - Proposta inicial
- `/DL_WORKFLOW_STATUS.md` - Status do workflow
- `/GPU_WORKFLOW.md` - Setup de GPUs

### Scripts Principais
- `orchestrator.py` - Orquestrador principal
- `dl_heads_v8.py` - Motor de DL
- `selector21.py` - Trading engine base

### Dados
- `/data/` - Dados históricos
- `/models/` - Modelos treinados
- `/out/` - Resultados e logs

---

## 💡 NOTAS IMPORTANTES

1. **Não overfit**: Walk-forward é essencial
2. **Simplicidade primeiro**: GRU simples > modelo complexo
3. **Features > Modelo**: 80% do ganho vem de boas features
4. **Latência matters**: Em produção, < 100ms por decisão
5. **Risk first**: Nunca comprometer gestão de risco por sinal

---

**Última Atualização**: 2025-11-08 20:00 UTC
**Responsável**: Sistema Botscalp V3
**Status**: Em desenvolvimento ativo