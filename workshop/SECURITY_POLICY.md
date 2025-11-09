#!/bin/bash
# Script para visualizar a política de segurança do Claudex

cat << 'SECURITY'

╔════════════════════════════════════════════════════════════════════════════════╗
║          🔒 SISTEMA DE PROTEÇÃO CLAUDEX - SEGURANÇA DE SCRIPTS              ║
╚════════════════════════════════════════════════════════════════════════════════╝

📌 OBJETIVO:

Proteger a integridade dos scripts base do Claudex enquanto permite que Claude e
GPT enriqueçam o sistema com novos conhecimentos, prompts customizados e feedback.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔐 PROTEGIDO (Read-Only):

Scripts Base (NUNCA podem ser editados):
  ✗ claudex.py
  ✗ claudex_editor.py
  ✗ claudex_security.py
  ✗ dupla_apresentacao.py
  ✗ dupla_aprendizado.py
  ✗ dupla_conversa.py
  ✗ dupla_conversa_fast.py
  ✗ feedback_em_acao.py
  ✗ MECANISMO_MOLDAGEM.py

Documentação Core (NUNCA pode ser editada):
  ✗ README.md
  ✗ claudex_prompt.md
  ✗ FEEDBACK_SYSTEM.md
  ✗ DUPLA_COMO_SE_MOLDAM.md
  ✗ CONVERSAS_README.md
  ✗ PERMISSIONS_UNRESTRICTED.md

Diretórios Protegidos (NUNCA podem ser alterados):
  ✗ backend/
  ✗ frontend/
  ✗ tools/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PERMITIDO PARA ENRIQUECIMENTO:

Diretórios Editáveis (Claude/GPT podem escrever):
  ✓ claudex/prompts_custom/     - Prompts customizados da IA
  ✓ claudex/knowledge/          - Base de conhecimento
  ✓ claudex/feedback/           - Feedback e aprendizado
  ✓ claudex/logs/               - Logs de execução

Arquivos Editáveis (Claude/GPT podem escrever):
  ✓ FEEDBACK_LOG.jsonl          - Histórico de feedback Y/N/?
  ✓ enhancements.json           - Enhancements e melhorias
  ✓ ai_learnings.json           - Aprendizados da IA
  ✓ performance_metrics.json    - Métricas de performance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 CASOS DE USO PERMITIDOS:

1. CRIAR NOVOS PROMPTS:
   $ claudex - write prompt claude
   → Salva em: claudex/prompts_custom/prompt_claude_TIMESTAMP.txt

2. ADICIONAR CONHECIMENTO:
   Claude/GPT podem criar arquivos em claudex/knowledge/:
   • estrategias_descobertas.json
   • padroes_mercado.json
   • regimes_identificados.json

3. REGISTRAR FEEDBACK:
   Sistema registra automaticamente em claudex/feedback/:
   • Respostas do usuário (Y/N/?)
   • Evolução da IA ao longo do tempo
   • Padrões de sucesso/falha

4. LOGS DE PERFORMANCE:
   Claude/GPT podem registrar em claudex/logs/:
   • Tempo de resposta
   • Qualidade das decisões
   • Evolução de resultados

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ CASOS BLOQUEADOS:

  ✗ Editar scripts base (claudex.py, dupla_*.py, etc)
  ✗ Modificar documentação core
  ✗ Alterar código do backend/frontend
  ✗ Deletar arquivos
  ✗ Editar configurações core
  ✗ Acessar código de outras IAs

⚠️  Se bloqueado, apareça mensagem clara explicando o porquê!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 VERIFICAR POLÍTICA:

$ claudex - security        # Mostra relatório completo de segurança
$ python3 claudex_security.py --report

Todos os acessos são registrados em:
  claudex/security_log.jsonl

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 FILOSOFIA:

"Proteger a integridade do código base enquanto permitimos que Claude e GPT
 aprendam, criem prompts customizados e documentem suas descobertas."

Sistema = Seguro ✅
Aprendizado = Livre ✅

SECURITY
