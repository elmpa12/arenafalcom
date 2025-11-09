#!/bin/bash
################################################################################
# flabs --dialogue — Multi-Agent Debate System
# Claude vs Codex: Dialogam, debatem, chegam a melhor decisão
# Você fica assistindo o debate em tempo real
################################################################################

cat << 'DIALOGUE_SPEC'

═══════════════════════════════════════════════════════════════════════════════
                    MULTI-AGENT DIALOGUE SYSTEM
           Claude (Strategist) vs Codex (Engineer) em Tempo Real
═══════════════════════════════════════════════════════════════════════════════

🎯 CONCEITO:

  User: "Quero um detector de regime com ML"
  
  Sistema abre DEBATE onde:
  
    Claude:  "Estratégia: usar Kalman filter + ensemble"
    Codex:   "Performance: Kalman é slow, considerar FastMA"
    Claude:  "Mas Kalman dá smoothing melhor para regime"
    Codex:   "Concordo, mas com threshold adaptativo"
    Claude:  "Perfeito! Adiciona kelly criterion também?"
    Codex:   "Pronto! Vai estar <100ms com isso"
    
  Resultado: Spec CONSENSUADO com melhor solução


BENEFÍCIOS:

  ✅ Combina visão estratégica (Claude) + pragmatismo técnico (Codex)
  ✅ Evita decisões ruins (debate expõe fraquezas)
  ✅ Você aprende vendo o diálogo
  ✅ Consenso = implementação melhor
  ✅ Menos back-and-forth depois


FLUXO:

  1. User fornece requisito
  2. Claude propõe estratégia inicial
  3. Codex critica/aprimora com constraints técnicos
  4. Claude responde aos pontos técnicos
  5. Codex aceita/refuta com evidência
  6. Loop até CONSENSO
  7. Output: DIALOGUE.md + CONSENSUS_SPEC.md


PROMPTS ESPECÍFICOS:

  Claude (Strategist Mode):
    "Você é estrategista. Veja o que Codex disse.
     Concorda? Discorda? Por quê?
     Responda com 1-2 pontos principais."
  
  Codex (Engineer Mode):
    "Você é pragmático. Veja o que Claude disse.
    É viável? Qual o custo (latência/complexidade)?
    Contra-argumento com dados técnicos."


TEMPERATURA DE DEBATE:

  Claude: 0.6 (criativo, critica construtiva)
  Codex:  0.5 (pragmático, data-driven)
  
  (Mais altas que normal pra gerar debate saudável)


═══════════════════════════════════════════════════════════════════════════════

DIALOGUE_SPEC

echo ""
echo "📝 Spec para implementação de --dialogue mode"
echo ""
echo "Próximas ações:"
echo "1. Criar dialogue_engine.py (orquestra debate)"
echo "2. Estender flabs com --dialogue submodo"
echo "3. Criar visualizador de debate em tempo real"
echo "4. Testar com exemplo: flabs --dialogue 'regime detector'"
echo ""
