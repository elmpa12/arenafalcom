#!/usr/bin/env python3
"""
Claudex - Sistema de IA Conversacional Dupla
Claude + Codex trabalhando juntos com feedback Y/N

Uso:
  claudex                           # Editor em modo AUTO
  claudex --plan                    # Editor em modo PLAN
  claudex --implement               # Editor em modo IMPLEMENT
  claudex --help                    # Ajuda
  claudex --status                  # Status do sistema
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Importa sistema de orquestração
sys.path.insert(0, str(Path(__file__).parent))
from claudex_orchestrator import DuoOrchestrator, show_duo_help

# Importa diretório ao path
CLAUDEX_DIR = Path(__file__).parent.absolute() / "claudex"
if not CLAUDEX_DIR.exists():
    CLAUDEX_DIR = Path("/opt/botscalpv3/claudex")
sys.path.insert(0, str(CLAUDEX_DIR.parent))

def print_banner():
    """Exibe banner do Claudex"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                          🔥 CLAUDEX - READY 🔥                           ║
║                     Claude + Codex - Inteligência Dupla                   ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

def show_menu():
    """Menu interativo principal"""
    print_banner()
    print("""
┌─ O QUE VOCÊ QUER FAZER? ─────────────────────────────────────────────────┐

  1. Ver APRESENTAÇÃO (Claude + GPT se apresentam)
     $ python3 claudex/dupla_apresentacao.py

  2. Simular 90 DIAS (Como evoluem em 90 dias)
     $ python3 claudex/dupla_aprendizado.py

  3. Ver DEBATES FORMAIS (3 debates estruturados)
     $ python3 claudex/dupla_conversa.py

  4. Ver CHATS RÁPIDOS (4 conversas naturais)
     $ python3 claudex/dupla_conversa_fast.py

  5. Ver FEEDBACK EM AÇÃO (Como aprendem com Y/N)
     $ python3 claudex/feedback_em_acao.py

  6. LER DOCUMENTAÇÃO
     6a. Guia Completo (claudex_prompt.md)
     6b. Como Se Moldam (DUPLA_COMO_SE_MOLDAM.md)
     6c. Sistema de Feedback (FEEDBACK_SYSTEM.md)
     6d. README Claudex (README.md)

  7. VER STATUS DO SISTEMA
  8. VER HISTÓRICO DE FEEDBACK
  E. EDITOR DE TEXTO (✨ NOVO!)
  9. SAIR

└─────────────────────────────────────────────────────────────────────────┘
    """)

def show_status():
    """Exibe status do sistema"""
    print_banner()
    print("""
📊 STATUS DO SISTEMA CLAUDEX:
    """)
    
    files = {
        "claudex_prompt.md": "Guia Completo",
        "README.md": "Visão Geral",
        "FEEDBACK_SYSTEM.md": "Sistema Y/N",
        "dupla_apresentacao.py": "Apresentação",
        "dupla_aprendizado.py": "Evolução 90 dias",
        "dupla_conversa.py": "Debates Formais",
        "dupla_conversa_fast.py": "Chats Rápidos",
        "feedback_em_acao.py": "Feedback em Ação",
        "FEEDBACK_LOG.jsonl": "Histórico de Feedback",
    }
    
    print("✅ ARQUIVOS DO SISTEMA:\n")
    for filename, description in files.items():
        filepath = CLAUDEX_DIR / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        if not exists:
            fallback_path = Path("/opt/botscalpv3/claudex") / filename
            if fallback_path.exists():
                status = "✓"
                exists = True
        print(f"  {status} {filename:<30} ({description})")
    
    feedback_log = Path("/opt/botscalpv3/claudex") / "FEEDBACK_LOG.jsonl"
    if feedback_log.exists():
        with open(feedback_log) as f:
            entries = len(f.readlines())
        print(f"\n📈 FEEDBACK LOG: {entries} entradas registradas")
    else:
        print(f"\n📈 FEEDBACK LOG: Vazio (será criado com primeira resposta)")
    
    print(f"\n🎯 PRÓXIMO PASSO: Escolha uma opção do menu!")

def show_feedback_history():
    """Exibe histórico de feedback"""
    print_banner()
    print("📋 HISTÓRICO DE FEEDBACK:\n")
    
    feedback_log = CLAUDEX_DIR / "FEEDBACK_LOG.jsonl"
    
    if not feedback_log.exists():
        print("  ⚠️  Nenhum feedback registrado ainda.")
        print("  Execute uma resposta e valide com Y/N/? para começar!")
        return
    
    try:
        with open(feedback_log) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        
        if not entries:
            print("  ⚠️  Arquivo vazio.")
            return
        
        print(f"  Total de feedbacks: {len(entries)}\n")
        
        # Estatísticas
        stats = {"Y": 0, "N": 0, "?": 0, "Y+": 0, "N-": 0}
        for entry in entries:
            feedback = entry.get("user_satisfaction", "?")
            stats[feedback] = stats.get(feedback, 0) + 1
        
        print("  📊 Distribuição:")
        for fb_type, count in sorted(stats.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {fb_type}: {count} {'✓' if fb_type in ['Y', 'Y+'] else '✗' if fb_type in ['N', 'N-'] else '⚠️ '}")
        
        # Últimas 5 entradas
        print(f"\n  📝 Últimas {min(5, len(entries))} respostas:")
        for i, entry in enumerate(entries[-5:], 1):
            ts = entry.get("timestamp", "?")
            satisfaction = entry.get("user_satisfaction", "?")
            response_type = entry.get("response_type", "?")
            print(f"    {i}. [{satisfaction}] {response_type} - {ts}")
    
    except Exception as e:
        print(f"  ⚠️  Erro ao ler histórico: {e}")

def run_command(cmd):
    """Executa comando"""
    resolved_dir = Path("/opt/botscalpv3/claudex").resolve()
    cmd = cmd.replace(str(CLAUDEX_DIR), str(resolved_dir))
    os.system(cmd)

def main():
    """Função principal"""
    args = sys.argv[1:]
    
    # SEM ARGUMENTOS: vai pro modo AUTO
    if not args:
        run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --auto")
        return
    
    first_arg = args[0].lower()
    
    # HELP, STATUS, FEEDBACK - sempre executam
    if first_arg == "--help":
        show_duo_help()
        return
    
    elif first_arg == "--status":
        show_status()
        return
    
    elif first_arg == "--feedback":
        show_feedback_history()
        return
    
    # FLAGS COM LÓGICA: se tem argumento, executa; senão abre editor
    elif first_arg == "--plan":
        if len(args) > 1:
            requirement = " ".join(args[1:])
            orchestrator = DuoOrchestrator()
            result = orchestrator.plan_phase(requirement)
            print(f"\n✅ Planejamento concluído!")
        else:
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --auto --mode plan")
        return
    
    elif first_arg == "--implement":
        if len(args) > 1:
            spec_file = args[1]
            if Path(spec_file).exists():
                spec = json.loads(Path(spec_file).read_text())
                orchestrator = DuoOrchestrator()
                orchestrator.implement_phase(spec)
            else:
                print(f"⚠️  Arquivo não encontrado: {spec_file}")
        else:
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --auto --mode implement")
        return
    
    elif first_arg == "--review":
        if len(args) > 1:
            spec_file = args[1]
            impl_file = args[2] if len(args) > 2 else "implementation.json"
            orchestrator = DuoOrchestrator()
            if Path(spec_file).exists() and Path(impl_file).exists():
                spec = json.loads(Path(spec_file).read_text())
                implementation = json.loads(Path(impl_file).read_text())
                orchestrator.review_phase(spec, implementation)
            else:
                print(f"⚠️  Arquivos não encontrados")
        else:
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --auto --mode review")
        return
    
    elif first_arg == "--pipeline":
        if len(args) > 1:
            requirement = " ".join(args[1:])
            # Tenta usar Dual GPT (funciona sem Claude)
            try:
                from claudex_dual_gpt import DualGPTOrchestrator
                orchestrator = DualGPTOrchestrator()
                result = orchestrator.pipeline_full(requirement)
                print(f"\n✅ Pipeline concluído!")
                print(f"Session ID: {result['session_id']}")
            except Exception as e:
                print(f"⚠️  Erro: {e}")
                print("Tentando modo clássico...")
                orchestrator = DuoOrchestrator()
                result = orchestrator.pipeline_full(requirement)
                print(f"\n✅ Pipeline concluído!")
                print(f"Session ID: {result['session_id']}")
        else:
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --auto --mode pipeline")
        return
    
    elif first_arg == "--dialogue":
        if len(args) > 1:
            topic = " ".join(args[1:])
            # Usa Dual GPT (funciona sem Claude)
            try:
                from claudex_dual_gpt import DualGPTOrchestrator
                orchestrator = DualGPTOrchestrator()
                result = orchestrator.debate_phase(topic, rounds=3)
                print(f"\n✅ Debate concluído!")
            except Exception as e:
                print(f"⚠️  Erro: {e}")
                print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  🎭 DIALOGUE: Claude vs GPT                               ║
╚════════════════════════════════════════════════════════════════════════════╝

Tema: {topic}

Para debates reais, execute:
  python3 claudex_dual_gpt.py --debate "{topic}"
                """)
        else:
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --auto --mode dialogue")
        return
    
    elif first_arg == "--auto":
        target = " ".join(args[1:]) if len(args) > 1 else None
        if target:
            print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  🔍 AUTO-DETECT: {target}                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Analisando contexto...
  • Tipo: {Path(target).suffix if Path(target).exists() else 'prompt'}
  • Rota: Pipeline automático
  
[Claude + GPT analisariam juntos e determinariam próximos passos]
            """)
        else:
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --auto")
        return
    
    # COMANDOS LEGADOS (com traço)
    command = first_arg.lower()
    
    if command in ["h", "help"]:
        print("""
Claudex 2.0 - Sistema de sinergia Claude + GPT

PIPELINE COM ARGUMENTO (Ambos sempre juntos):
  claudex --plan "requisito"           # GPT organiza specs
  claudex --implement spec.json        # Claude executa código
  claudex --review spec.json impl.json # Cross-review (ambos)
  claudex --pipeline "tarefa"          # Tudo automatizado
  claudex --dialogue "tema"            # Debate em tempo real
  claudex --auto <arquivo>             # Auto-detect + rota

EDITOR (Sem argumentos ou com --flag):
  claudex                               # Editor em modo AUTO (padrão)
  claudex --plan                        # Editor em modo PLAN
  claudex --implement                   # Editor em modo IMPLEMENT
  claudex --review                      # Editor em modo REVIEW
  claudex --pipeline                    # Editor em modo PIPELINE
  claudex --dialogue                    # Editor em modo DIALOGUE
  claudex --auto                        # Editor em modo AUTO

SISTEMA:
  claudex --help                        # Esta mensagem
  claudex --status                      # Status do sistema
  claudex --feedback                    # Histórico de feedback
  claudex - help                        # Comandos legados
        """)
        return
    
    elif command in ["s", "status"]:
        show_status()
        return
    
    elif command in ["f", "feedback"]:
        show_feedback_history()
        return
    
    elif command in ["a", "apresentacao"]:
        run_command(f"python3 {CLAUDEX_DIR}/dupla_apresentacao.py")
        return
    
    elif command in ["m", "moldagem"]:
        run_command(f"python3 {CLAUDEX_DIR}/dupla_aprendizado.py")
        return
    
    elif command in ["d", "debates"]:
        run_command(f"python3 {CLAUDEX_DIR}/dupla_conversa.py")
        return
    
    elif command in ["c", "chats"]:
        run_command(f"python3 {CLAUDEX_DIR}/dupla_conversa_fast.py")
        return
    
    elif command in ["demo"]:
        run_command(f"python3 {CLAUDEX_DIR}/feedback_em_acao.py")
        return
    
    elif command in ["prompt", "prompts"]:
        run_command(f"less {CLAUDEX_DIR}/claudex_prompt.md")
        return
    
    elif command in ["sec", "security"]:
        run_command(f"python3 {Path(__file__).parent}/claudex_security.py --report")
        return
    
    elif command in ["write", "editor"]:
        editor_type = args[1] if len(args) > 1 else "edit"
        if editor_type == "prompt":
            system = args[2] if len(args) > 2 else "claude"
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --prompt {system}")
        elif editor_type == "question":
            topic = " ".join(args[2:]) if len(args) > 2 else ""
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --question {topic}")
        else:
            run_command(f"python3 {Path(__file__).parent}/claudex_editor.py --edit")
        return

if __name__ == "__main__":
    main()
