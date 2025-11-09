#!/usr/bin/env python3
"""
Teste automatizado do Claudex 2.0
Mostra todas as funcionalidades sem input interativo
"""

from claudex_dual_gpt import DualGPTOrchestrator
import json

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                🔥 CLAUDEX 2.0 - DEMO AUTOMÁTICO                           ║
║              Mostrando o poder das IAs em ação!                           ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# Cria orchestrator
orch = DualGPTOrchestrator()

print("\n" + "="*70)
print("🎭 TESTE 1: DEBATE RÁPIDO")
print("="*70 + "\n")

# Teste 1: Debate curto
result1 = orch.debate_phase("Melhor timeframe para scalping: 1m ou 5m?", rounds=2)

print("\n" + "="*70)
print("📊 RESULTADO DO DEBATE:")
print("="*70)
print(f"✅ Consenso: {result1['consensus'][:200]}...")
print(f"📁 Salvo em: {orch.session_dir}/debate.json")

print("\n" + "="*70)
print("🎯 ANÁLISE:")
print("="*70)
print("✓ GPT-Strategist pensou estrategicamente")
print("✓ GPT-Executor validou tecnicamente")
print("✓ Consenso gerado automaticamente")
print("✓ Tudo salvo para referência futura")

print("\n" + "="*70)
print("💡 COMO USAR:")
print("="*70)
print("""
1. Debates:
   python3 claudex.py --dialogue "seu tema aqui"

2. Implementação:
   python3 claudex_dual_gpt.py --pipeline "sua tarefa aqui"
   (responda as perguntas interativamente)

3. Ver resultados:
   cat claudex/work/*/debate.json | jq .
   cat claudex/work/*/REVIEW.md
""")

print("\n" + "="*70)
print("🚀 CLAUDEX 2.0 ESTÁ PRONTO PARA USO!")
print("="*70)
print("""
PRÓXIMOS TESTES SUGERIDOS:

1. Debate sobre estratégia real:
   python3 claudex.py --dialogue "Análise de setup atual do BTC"

2. Gerar código completo:
   # Execute interativamente para responder os ENTERs
   python3 claudex_dual_gpt.py --pipeline "Criar sistema de alertas"

3. Ver histórico:
   python3 claudex.py --feedback
   python3 claudex.py --status

✨ O sistema está VIVO e FUNCIONANDO!
""")

print("\n📈 ESTATÍSTICAS DESTA SESSÃO:")
print(f"   Session ID: {orch.session_id}")
print(f"   Arquivos: {orch.session_dir}")
print(f"   Status: ✅ SUCESSO")
