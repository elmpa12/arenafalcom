#!/usr/bin/env python3
"""
Claudex Security System - Proteção de scripts base
Permite enriquecimento de IA mas protege código core
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Diretório raiz do Claudex
CLAUDEX_ROOT = Path("/opt/botscalpv3")
CLAUDEX_DIR = CLAUDEX_ROOT / "claudex"

class SecurityPolicy:
    """
    Política de segurança do Claudex
    
    Protegido (Read-Only):
    - Scripts base: claudex.py, claudex_editor.py, etc
    - Código core: backend/*, frontend/*
    - Documentação: *.md files
    
    Permitido para enriquecimento (Writable):
    - Diretório: claudex/knowledge/
    - Diretório: claudex/prompts_custom/
    - Diretório: claudex/feedback/
    - Arquivo: claudex/FEEDBACK_LOG.jsonl
    - Arquivo: claudex/enhancements.json
    """
    
    # Scripts base que NUNCA podem ser editados
    PROTECTED_FILES = {
        "claudex.py",
        "claudex_editor.py",
        "dupla_apresentacao.py",
        "dupla_aprendizado.py",
        "dupla_conversa.py",
        "dupla_conversa_fast.py",
        "feedback_em_acao.py",
        "MECANISMO_MOLDAGEM.py",
    }
    
    # Documentação base que NUNCA pode ser editada
    PROTECTED_DOCS = {
        "README.md",
        "claudex_prompt.md",
        "FEEDBACK_SYSTEM.md",
        "DUPLA_COMO_SE_MOLDAM.md",
        "CONVERSAS_README.md",
        "PERMISSIONS_UNRESTRICTED.md",
    }
    
    # Diretórios protegidos (apenas leitura)
    PROTECTED_DIRS = {
        "backend",
        "frontend",
        "tools",
    }
    
    # Diretórios permitidos para enriquecimento (escrita)
    WRITABLE_DIRS = {
        "knowledge",      # Base de conhecimento customizada
        "prompts_custom", # Prompts customizados da IA
        "feedback",       # Feedback e aprendizado
        "logs",           # Logs de execução
    }
    
    # Arquivos permitidos para enriquecimento (escrita)
    WRITABLE_FILES = {
        "FEEDBACK_LOG.jsonl",           # Log de feedback
        "enhancements.json",            # Enhancements realizados
        "ai_learnings.json",            # Aprendizados da IA
        "performance_metrics.json",     # Métricas de performance
    }
    
    @staticmethod
    def is_protected(filepath: Path) -> bool:
        """Verifica se um arquivo está protegido"""
        filepath = Path(filepath)
        filename = filepath.name
        
        # Verifica se é arquivo protegido
        if filename in SecurityPolicy.PROTECTED_FILES:
            return True
        
        if filename in SecurityPolicy.PROTECTED_DOCS:
            return True
        
        # Verifica se está em diretório protegido
        for protected_dir in SecurityPolicy.PROTECTED_DIRS:
            if protected_dir in filepath.parts:
                return True
        
        return False
    
    @staticmethod
    def is_writable(filepath: Path) -> bool:
        """Verifica se um arquivo pode ser escrito (enriquecimento)"""
        filepath = Path(filepath)
        filename = filepath.name
        
        # Arquivo permitido?
        if filename in SecurityPolicy.WRITABLE_FILES:
            return True
        
        # Está em diretório permitido?
        for writable_dir in SecurityPolicy.WRITABLE_DIRS:
            if writable_dir in filepath.parts:
                return True
        
        return False
    
    @staticmethod
    def check_permission(filepath: Path, operation: str = "read") -> Tuple[bool, str]:
        """
        Verifica permissão para operação
        
        Args:
            filepath: Caminho do arquivo
            operation: "read", "write", "delete"
        
        Returns:
            (allowed: bool, message: str)
        """
        filepath = Path(filepath)
        
        if operation == "read":
            return (True, "✅ Leitura permitida")
        
        elif operation == "write":
            if SecurityPolicy.is_protected(filepath):
                return (False, f"""
⛔ ACESSO NEGADO: Script protegido!

Arquivo: {filepath.name}
Motivo: Este é um arquivo base do Claudex que não pode ser editado

✅ PERMITIDO (Enriquecimento):
   • Criar prompts customizados em claudex/prompts_custom/
   • Adicionar conhecimento em claudex/knowledge/
   • Salvar feedback em claudex/feedback/
   • Logs de execução em claudex/logs/

🔒 PROTEGIDO (Apenas leitura):
   • Scripts base do sistema
   • Documentação core
   • Código da IA

💡 OBJETIVO: Proteger integridade do sistema enquanto permite IA enriquecer com novos conhecimentos
                """)
            
            elif SecurityPolicy.is_writable(filepath):
                return (True, "✅ Enriquecimento permitido")
            
            else:
                return (False, f"""
⚠️  ACESSO RESTRITO: Localização não permitida

Arquivo: {filepath}

✅ LOCALIZAÇÕES PERMITIDAS:
   • {CLAUDEX_DIR}/knowledge/
   • {CLAUDEX_DIR}/prompts_custom/
   • {CLAUDEX_DIR}/feedback/
   • {CLAUDEX_DIR}/logs/
                """)
        
        elif operation == "delete":
            return (False, "⛔ Deleção não permitida pelo Claudex")
        
        else:
            return (False, f"❓ Operação desconhecida: {operation}")


class SecurityLogger:
    """Registra tentativas de acesso"""
    
    LOG_FILE = CLAUDEX_DIR / "security_log.jsonl"
    
    @staticmethod
    def log_access(filepath: Path, operation: str, allowed: bool, reason: str):
        """Registra tentativa de acesso"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "filepath": str(filepath),
            "operation": operation,
            "allowed": allowed,
            "reason": reason,
        }
        
        try:
            with open(SecurityLogger.LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"⚠️  Erro ao registrar: {e}")


def protect_file_operation(filepath: Path, operation: str = "write") -> bool:
    """
    Protetor genérico para operações de arquivo
    
    Uso:
        if protect_file_operation(filepath, "write"):
            # Prosseguir com escrita
            ...
    """
    allowed, message = SecurityPolicy.check_permission(filepath, operation)
    
    SecurityLogger.log_access(filepath, operation, allowed, message)
    
    if not allowed:
        print(message)
        return False
    
    return True


def get_security_report() -> str:
    """Gera relatório de segurança"""
    report = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    🔒 CLAUDEX SECURITY REPORT                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

📋 CONFIGURAÇÃO DE SEGURANÇA:

Protected Scripts ({len(SecurityPolicy.PROTECTED_FILES)}):
{chr(10).join(f"  • {f}" for f in sorted(SecurityPolicy.PROTECTED_FILES))}

Protected Docs ({len(SecurityPolicy.PROTECTED_DOCS)}):
{chr(10).join(f"  • {f}" for f in sorted(SecurityPolicy.PROTECTED_DOCS))}

Protected Directories ({len(SecurityPolicy.PROTECTED_DIRS)}):
{chr(10).join(f"  • {d}/" for d in sorted(SecurityPolicy.PROTECTED_DIRS))}

Writable Directories ({len(SecurityPolicy.WRITABLE_DIRS)}):
{chr(10).join(f"  • {d}/" for d in sorted(SecurityPolicy.WRITABLE_DIRS))}

Writable Files ({len(SecurityPolicy.WRITABLE_FILES)}):
{chr(10).join(f"  • {f}" for f in sorted(SecurityPolicy.WRITABLE_FILES))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PERMITIDO PARA ENRIQUECIMENTO:

   • Criar novos prompts customizados
   • Adicionar conhecimento sobre mercado
   • Registrar feedback de operações
   • Salvar aprendizados e insights
   • Logs de execução e performance

❌ BLOQUEADO PARA PROTEÇÃO:

   • Editar scripts base
   • Modificar documentação core
   • Alterar código do sistema
   • Deletar arquivos importantes
   • Editar configurações core

🎯 OBJETIVO:

   Permitir que Claude/GPT enriqueçam o sistema com conhecimento
   enquanto protegem a integridade do código base.

═══════════════════════════════════════════════════════════════════════════
"""
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        print(get_security_report())
    else:
        # Test
        print(get_security_report())
