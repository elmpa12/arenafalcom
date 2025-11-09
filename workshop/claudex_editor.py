#!/usr/bin/env python3
"""
Claudex Text Editor - Mini editor para escrever prompts no shell
Suporta ENTER para quebrar linha e CTRL+Y para confirmar
Com proteção de segurança para scripts base
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Importa sistema de segurança
sys.path.insert(0, str(Path(__file__).parent))
from claudex_security import SecurityPolicy, protect_file_operation, SecurityLogger

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def clear_screen():
    """Limpa a tela"""
    os.system('clear' if os.name == 'posix' else 'cls')

def show_banner():
    """Mostra banner do editor"""
    print(f"""
{CYAN}╔═══════════════════════════════════════════════════════════════════════════╗
║                    ✏️  CLAUDEX TEXT EDITOR                              ║
║              (ENTER quebra linha, CTRL+Y confirma)                      ║
╚═══════════════════════════════════════════════════════════════════════════╝{RESET}
    """)

def show_help():
    """Mostra ajuda dos comandos"""
    print(f"""
{BLUE}📋 CONTROLES:{RESET}
  • ENTER              → Quebra linha / Nova linha
  • CTRL+C             → Cancela (sem salvar)
  • CTRL+D             → Confirma e salva

{YELLOW}💡 DICAS:{RESET}
  • Digite naturalmente, use ENTER para novas linhas
  • Pressione CTRL+D quando terminar
  • O texto será processado e devolvido
    """)

def text_editor(title="Escreva seu texto:", placeholder="", max_lines=None):
    """
    Mini editor de texto interativo - VERSÃO SIMPLIFICADA
    
    Args:
        title: Título do editor
        placeholder: Texto padrão
        max_lines: Máximo de linhas (None = ilimitado)
    
    Returns:
        str: Texto editado ou None se cancelado
    """
    clear_screen()
    show_banner()
    
    print(f"{BLUE}{title}{RESET}\n")
    print(f"{YELLOW}💬 Digite seu texto (CTRL+D para confirmar, CTRL+C para cancelar):{RESET}\n")
    
    lines = []
    line_num = 1
    
    try:
        while True:
            try:
                print(f"{CYAN}[{line_num:2d}]{RESET} ", end="", flush=True)
                line = input()
                
                lines.append(line)
                line_num += 1
                
                # Verifica limite
                if max_lines and line_num > max_lines:
                    print(f"\n{YELLOW}⚠️  Limite de {max_lines} linhas atingido!{RESET}\n")
                    break
                
            except KeyboardInterrupt:
                # CTRL+C
                print(f"\n{RED}❌ Editor cancelado (CTRL+C){RESET}")
                return None
            except EOFError:
                # CTRL+D
                break
        
        if not lines:
            print(f"\n{YELLOW}⚠️  Nenhum texto foi fornecido{RESET}")
            return None
        
        # Remove última linha se vazia (quando usar CTRL+D)
        while lines and not lines[-1]:
            lines.pop()
        
        result = '\n'.join(lines)
        
        # Mostra confirmação
        print(f"\n{GREEN}✅ Texto confirmado!{RESET}\n")
        print(f"{BLUE}Seu texto ({len(lines)} linhas, {len(result)} caracteres):{RESET}\n")
        print(f"{CYAN}{'─' * 70}{RESET}")
        for line in lines:
            print(f"{GREEN}│{RESET} {line}")
        print(f"{CYAN}{'─' * 70}{RESET}\n")
        
        return result
        
    except Exception as e:
        print(f"\n{RED}❌ Erro: {e}{RESET}")
        return None

def editor_prompt(system="claude"):
    """Editor específico para criar prompts de sistema"""
    title = f"📝 Criar prompt para {system.upper()}"
    
    print(f"\n{BLUE}{title}{RESET}\n")
    text = text_editor(title=title, max_lines=50)
    
    if text:
        # Salva em arquivo SEGURO (diretório de prompts customizados)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Cria diretório de prompts customizados se não existir
        custom_prompts_dir = Path("/opt/botscalpv3/claudex/prompts_custom")
        custom_prompts_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = custom_prompts_dir / f"prompt_{system}_{timestamp}.txt"
        
        # Verifica permissão de escrita
        if not protect_file_operation(filepath, "write"):
            print(f"{RED}❌ Não foi possível salvar o arquivo{RESET}")
            return None
        
        filepath.write_text(text)
        
        print(f"{GREEN}💾 Salvo em: {filepath}{RESET}\n")
        
        return text
    else:
        print(f"{YELLOW}⚠️  Edição cancelada{RESET}")
        return None

def editor_question(topic=""):
    """Editor para fazer perguntas"""
    if topic:
        title = f"❓ Fazer pergunta sobre: {topic}"
    else:
        title = "❓ Qual é sua pergunta?"
    
    text = text_editor(title=title, max_lines=30)
    return text

def editor_auto(mode="auto"):
    """
    Editor em modo AUTO com detecção de contexto
    Abre o editor e tenta sugerir o modo apropriado
    """
    clear_screen()
    show_banner()
    
    print(f"{YELLOW}🎯 Modo: {mode.upper()}{RESET}\n")
    
    if mode == "auto":
        print(f"{BLUE}💡 Digite seu conteúdo e o sistema vai sugerir o modo:{RESET}")
        print(f"   • Se detectar requisitos → --plan")
        print(f"   • Se detectar código → --implement")
        print(f"   • Se detectar especificação → --review")
        print()
    elif mode == "plan":
        print(f"{BLUE}📋 Modo PLANEJAMENTO (GPT vai organizar specs):{RESET}\n")
    elif mode == "implement":
        print(f"{BLUE}💻 Modo IMPLEMENTAÇÃO (Claude vai escrever código):{RESET}\n")
    elif mode == "review":
        print(f"{BLUE}✅ Modo REVIEW (Ambos vão revisar):{RESET}\n")
    elif mode == "pipeline":
        print(f"{BLUE}🔄 Modo PIPELINE (Plan → Implement → Review):{RESET}\n")
    elif mode == "dialogue":
        print(f"{BLUE}🎭 Modo DIÁLOGO (Claude vs GPT debate):{RESET}\n")
    
    text = text_editor(title="", max_lines=None)
    return text

def main():
    """Menu principal"""
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == "--prompt":
            system = sys.argv[2] if len(sys.argv) > 2 else "claude"
            editor_prompt(system)
        
        elif cmd == "--question":
            topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
            title = f"❓ Fazer pergunta sobre: {topic}" if topic else "❓ Qual é sua pergunta?"
            print(f"\n{BLUE}{title}{RESET}\n")
            editor_question(topic)
        
        elif cmd == "--edit":
            title = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Editar texto"
            text_editor(title=title)
        
        elif cmd == "--auto":
            mode = "auto"
            # Suporte para --mode plan, --mode implement, etc
            if len(sys.argv) > 2 and sys.argv[2] == "--mode" and len(sys.argv) > 3:
                mode = sys.argv[3]
            elif len(sys.argv) > 2:
                mode = sys.argv[2]
            
            editor_auto(mode)
        
        elif cmd == "--help":
            show_banner()
            show_help()
        
        else:
            text_editor(title="Escrever seu texto:")
    else:
        text_editor(title="Escreva seu texto:")

if __name__ == "__main__":
    main()
