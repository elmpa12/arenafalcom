#!/usr/bin/env python3
################################################################################
# router.py — Roteador Inteligente para Dual-AI System
# Seleciona o melhor agente (Claude ou Codex) para cada tarefa
################################################################################

from typing import Literal, Dict, Any
from enum import Enum

class Task(str, Enum):
    """Tipos de tarefa que podem ser roteadas"""
    SPEC = "spec"           # Especificação
    IMPL = "impl"           # Implementação
    REVIEW = "review"       # Code review
    DOCS = "docs"           # Documentação
    ASSIMILATE = "assimilate"  # Assimilação de contexto
    REFACTOR = "refactor"   # Refatoração
    TESTS = "tests"         # Testes
    IDEATION = "ideation"   # Ideação criativa

class Agent(str, Enum):
    """Agentes disponíveis"""
    CLAUDE = "claude"
    CODEX = "codex"
    HYBRID = "hybrid"  # Ambos, em sequência

class RouterPolicy:
    """Política de roteamento baseada em heurísticas"""
    
    def __init__(self):
        self.rules = {
            Task.SPEC: {
                "primary": Agent.CLAUDE,
                "reason": "Claude excels em estrutura e síntese",
                "temperature": 0.3,
            },
            Task.IMPL: {
                "primary": Agent.CODEX,
                "reason": "Codex excels em implementação precisa",
                "temperature": 0.2,
            },
            Task.REVIEW: {
                "primary": Agent.CODEX,
                "reason": "Codex tem review técnico mais preciso",
                "temperature": 0.2,
            },
            Task.DOCS: {
                "primary": Agent.CLAUDE,
                "reason": "Claude sintetiza documentação melhor",
                "temperature": 0.5,
            },
            Task.ASSIMILATE: {
                "primary": Agent.CLAUDE,
                "reason": "Claude processa contexto longo com naturalidade",
                "temperature": 0.3,
            },
            Task.REFACTOR: {
                "primary": Agent.CODEX,
                "reason": "Codex otimiza código com precisão",
                "temperature": 0.2,
            },
            Task.TESTS: {
                "primary": Agent.CODEX,
                "reason": "Codex gera testes estruturados",
                "temperature": 0.2,
            },
            Task.IDEATION: {
                "primary": Agent.CLAUDE,
                "reason": "Claude é mais criativo em exploração",
                "temperature": 0.7,
            },
        }
    
    def route(
        self,
        task: Task,
        context_size_kloc: int = 0,
        needs_long_context: bool = False,
        requires_precision: bool = False,
    ) -> Dict[str, Any]:
        """
        Roteia a tarefa para o melhor agente.
        
        Args:
            task: Tipo de tarefa
            context_size_kloc: Tamanho do contexto em KLOC (mil linhas)
            needs_long_context: Se precisa ler muito código
            requires_precision: Se precisão é crítica
        
        Returns:
            {
                "agent": Agent,
                "profile": str (nome do perfil no .codex.toml),
                "temperature": float,
                "reason": str,
            }
        """
        
        rule = self.rules.get(task, self.rules[Task.IMPL])
        agent = rule["primary"]
        
        # Heurística 1: Tamanho de contexto grande → Claude
        if context_size_kloc > 50 and task in (Task.SPEC, Task.ASSIMILATE, Task.REVIEW):
            agent = Agent.CLAUDE
        
        # Heurística 2: Precisão crítica → Codex para implementação/review
        if requires_precision and task in (Task.IMPL, Task.REVIEW, Task.TESTS):
            agent = Agent.CODEX
        
        # Heurística 3: Contexto longo → Claude (sempre vence em contexto)
        if needs_long_context and task != Task.IMPL:
            agent = Agent.CLAUDE
        
        # Map agent to profile name
        profile_map = {
            (Agent.CLAUDE, Task.SPEC): "claude_spec",
            (Agent.CLAUDE, Task.DOCS): "claude_spec",
            (Agent.CLAUDE, Task.ASSIMILATE): "claude_spec",
            (Agent.CLAUDE, Task.REVIEW): "claude_review",
            (Agent.CLAUDE, Task.IDEATION): "claude_spec",
            (Agent.CODEX, Task.IMPL): "codex_impl",
            (Agent.CODEX, Task.REFACTOR): "codex_impl",
            (Agent.CODEX, Task.TESTS): "codex_impl",
            (Agent.CODEX, Task.REVIEW): "codex_review",
        }
        
        profile = profile_map.get(
            (agent, task),
            "codex_impl" if agent == Agent.CODEX else "claude_spec"
        )
        
        return {
            "agent": agent.value,
            "profile": profile,
            "temperature": rule["temperature"],
            "reason": rule["reason"],
        }


# Instância global
router = RouterPolicy()


def route(
    task: str,
    context_size_kloc: int = 0,
    needs_long_context: bool = False,
    requires_precision: bool = False,
) -> Dict[str, Any]:
    """Função pública de roteamento"""
    try:
        task_enum = Task[task.upper()]
    except KeyError:
        # Default to IMPL se task desconhecida
        task_enum = Task.IMPL
    
    return router.route(
        task_enum,
        context_size_kloc=context_size_kloc,
        needs_long_context=needs_long_context,
        requires_precision=requires_precision,
    )


# ============================================================
# CLI para teste rápido
# ============================================================

if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python3 router.py <task> [--context-kloc N] [--long] [--precise]")
        print("\nTasks:", ", ".join([t.value for t in Task]))
        sys.exit(1)
    
    task = sys.argv[1]
    context_kloc = 0
    needs_long = "--long" in sys.argv
    needs_precise = "--precise" in sys.argv
    
    # Parse --context-kloc
    if "--context-kloc" in sys.argv:
        idx = sys.argv.index("--context-kloc")
        if idx + 1 < len(sys.argv):
            context_kloc = int(sys.argv[idx + 1])
    
    result = route(
        task,
        context_size_kloc=context_kloc,
        needs_long_context=needs_long,
        requires_precision=needs_precise,
    )
    
    print(json.dumps(result, indent=2))
