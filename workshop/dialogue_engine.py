#!/usr/bin/env python3
"""
dialogue_engine.py — Multi-Agent Debate System com Memória Persistente
Claude (Strategist) vs Codex (Engineer) em debate colaborativo
Ambos lembram da sessão anterior via agent_memory.py
Você fica assistindo em tempo real
"""

import os
import json
import time
from datetime import datetime
from typing import Optional
import subprocess
from pathlib import Path

# Import memory system
try:
    from agent_memory import AgentMemory
    MEMORY_ENABLED = True
except ImportError:
    print("⚠️  agent_memory.py não encontrado. Continuando sem memória persistente.")
    MEMORY_ENABLED = False


class DialogueAgent:
    """Representa um agente no diálogo"""
    
    def __init__(self, name: str, role: str, api_type: str, temperature: float):
        self.name = name
        self.role = role  # "strategist" ou "engineer"
        self.api_type = api_type  # "anthropic" ou "openai"
        self.temperature = temperature
        self.context = []  # Histórico do diálogo
    
    def add_to_context(self, speaker: str, message: str):
        """Adiciona fala ao contexto do agente"""
        self.context.append({"speaker": speaker, "message": message})
    
    def get_context_string(self) -> str:
        """Formata contexto como string para passar ao agente"""
        if not self.context:
            return ""
        
        context_lines = []
        for turn in self.context:
            context_lines.append(f"{turn['speaker']}: {turn['message']}")
        
        return "\n".join(context_lines)


class DialogueEngine:
    """Orquestra debate entre dois agentes com memória persistente"""
    
    def __init__(self, max_rounds: int = 5):
        self.max_rounds = max_rounds
        self.current_round = 0
        self.dialogue_history = []
        self.consensus_reached = False
        self.dialogue_id = None
        
        # Agentes
        self.claude = DialogueAgent(
            name="Claude",
            role="strategist",
            api_type="anthropic",
            temperature=0.6
        )
        
        self.codex = DialogueAgent(
            name="Codex",
            role="engineer",
            api_type="openai",
            temperature=0.5
        )
        
        # Memória persistente (se disponível)
        self.claude_memory = None
        self.codex_memory = None
        self.memory_enabled = MEMORY_ENABLED
        if MEMORY_ENABLED:
            try:
                memory_dir = Path("/opt/botscalpv3/memory_store")
                self.claude_memory = AgentMemory("Claude", str(memory_dir))
                self.codex_memory = AgentMemory("Codex", str(memory_dir))
                print(f"✅ Memória carregada para ambos agentes")
            except Exception as e:
                print(f"⚠️  Erro ao carregar memória: {e}")
                self.memory_enabled = False
    
    def _call_claude(self, prompt: str) -> str:
        """Chama Claude API"""
        import anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "❌ ANTHROPIC_API_KEY não configurada"
        
        client = anthropic.Anthropic(api_key=api_key)
        
        try:
            message = client.messages.create(
                model="claude-opus-4-1",
                max_tokens=500,
                temperature=self.claude.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"❌ Erro Claude: {str(e)}"
    
    def _call_codex(self, prompt: str) -> str:
        """Chama Codex via gateway (fallback)"""
        import requests
        
        api_key = os.getenv("OPENAI_API_KEY")
        gateway_url = os.getenv("GATEWAY_URL", "https://bs3.falcomlabs.com/codex/api/codex")
        
        if not api_key:
            return "❌ OPENAI_API_KEY não configurada"
        
        try:
            response = requests.post(
                gateway_url,
                headers={"Content-Type": "application/json"},
                json={
                    "prompt": prompt,
                    "model": "gpt-5-codex",
                    "mode": "codex",
                    "temperature": self.codex.temperature
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "❌ Sem resposta")
            else:
                return f"❌ Gateway error: {response.status_code}"
        except Exception as e:
            return f"❌ Erro Codex: {str(e)}"
    
    def _display_message(self, speaker: str, message: str, color: str = "\033[0m"):
        """Exibe mensagem formatada em tempo real"""
        COLORS = {
            "claude": "\033[36m",      # Cyan
            "codex": "\033[33m",       # Yellow
            "system": "\033[32m",      # Green
            "reset": "\033[0m"
        }
        
        actual_color = COLORS.get(color, COLORS["reset"])
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{actual_color}[{timestamp}] {speaker}:{COLORS['reset']}")
        print(f"{actual_color}{message}{COLORS['reset']}")
        print()
    
    def _check_consensus(self, claude_msg: str, codex_msg: str) -> bool:
        """Simples heurística: consenso se ambos concordam (keywords)"""
        consensus_keywords = [
            "concordo",
            "agree",
            "perfeito",
            "perfect",
            "vamos com isso",
            "let's go with that",
            "excelente ideia",
            "great idea"
        ]
        
        claude_agrees = any(kw in claude_msg.lower() for kw in consensus_keywords)
        codex_agrees = any(kw in codex_msg.lower() for kw in consensus_keywords)
        
        return claude_agrees and codex_agrees
    
    def run_dialogue(self, requirement: str) -> dict:
        """Executa debate completo com memória persistente"""
        
        # Gera ID único para este diálogo
        self.dialogue_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        print("\n" + "="*80)
        print("🎭 MULTI-AGENT DIALOGUE — Claude vs Codex (com Memória)")
        print("="*80)
        print(f"\n📋 Requisito: {requirement}")
        print(f"🔖 Dialogue ID: {self.dialogue_id}\n")
        
        # ==================== INJETAR CONTEXTO DA MEMÓRIA ====================
        claude_memory_context = ""
        codex_memory_context = ""
        shared_context = ""
        
        if self.memory_enabled and self.claude_memory and self.codex_memory:
            print("📚 Carregando contexto histórico...\n")
            
            # Contexto pessoal de Claude
            try:
                claude_memory_context = self.claude_memory.get_context_for_dialogue()
            except:
                claude_memory_context = ""
            
            # Contexto pessoal de Codex
            try:
                codex_memory_context = self.codex_memory.get_context_for_dialogue()
            except:
                codex_memory_context = ""
            
            # Contexto compartilhado
            try:
                shared_context = self.codex_memory.get_shared_context()
            except:
                shared_context = ""
            
            if claude_memory_context:
                print(f"✅ Claude lembrou: {claude_memory_context[:100]}...")
            if codex_memory_context:
                print(f"✅ Codex lembrou: {codex_memory_context[:100]}...")
            if shared_context:
                print(f"✅ Conhecimento compartilhado carregado")
        
        # ==================== ROUND 1: CLAUDE PROPÕE ====================
        print("\n" + "-"*80)
        print("ROUND 1: Claude propõe estratégia (com memória)")
        print("-"*80)
        
        claude_prompt = f"""Você é Claude, estrategista de software elite.

{claude_memory_context}

CONTEXTO COMPARTILHADO COM CODEX:
{shared_context}

NOVO REQUISITO: {requirement}

Proponha a melhor ESTRATÉGIA em 3-4 pontos:
- Visão geral da solução
- Tecnologias-chave
- Trade-offs principais
- Próximos passos

Seja conciso e assertivo. Referência aos padrões que aprendemos juntos se relevante."""
        
        claude_response = self._call_claude(claude_prompt)
        self._display_message("Claude", claude_response, "claude")
        self.claude.add_to_context("Claude", claude_response)
        self.codex.add_to_context("Claude", claude_response)
        self.dialogue_history.append({"round": 1, "speaker": "Claude", "message": claude_response})
        
        # ==================== ROUNDS ALTERNADOS COM MEMÓRIA ====================
        for round_num in range(2, self.max_rounds + 1):
            if round_num % 2 == 0:
                # Codex responde
                print("\n" + "-"*80)
                print(f"ROUND {round_num}: Codex critica/aprimora (com memória)")
                print("-"*80)
                
                codex_prompt = f"""Você é Codex, engenheiro pragmático.

{codex_memory_context}

CONTEXTO COMPARTILHADO COM CLAUDE:
{shared_context}

ESTRATÉGIA PROPOSTA POR CLAUDE:
{claude_response}

Avalie:
1. É viável? Por quê/por quê não?
2. Qual o custo técnico (latência, complexidade)?
3. Alternativas mais eficientes?
4. Sua contraproposta (se houver)

Seja direto e data-driven. Referência nossos padrões anteriores se aplicável."""
                
                codex_response = self._call_codex(codex_prompt)
                self._display_message("Codex", codex_response, "codex")
                self.codex.add_to_context("Codex", codex_response)
                self.claude.add_to_context("Codex", codex_response)
                self.dialogue_history.append({"round": round_num, "speaker": "Codex", "message": codex_response})
                
                # Verifica consenso
                if self._check_consensus(claude_response, codex_response):
                    self._display_message("SYSTEM", "✅ CONSENSO ATINGIDO!", "system")
                    self.consensus_reached = True
                    break
                
                last_response = codex_response
            else:
                # Claude responde
                print("\n" + "-"*80)
                print(f"ROUND {round_num}: Claude replica (com memória)")
                print("-"*80)
                
                claude_prompt = f"""Você é Claude, estrategista.

{claude_memory_context}

CRÍTICA DE CODEX:
{last_response}

Responda:
1. Concorda com os points técnicos?
2. Como refinaria a estratégia com o feedback?
3. Sua proposta final

Seja construtor de consenso."""
                
                claude_response = self._call_claude(claude_prompt)
                self._display_message("Claude", claude_response, "claude")
                self.claude.add_to_context("Claude", claude_response)
                self.codex.add_to_context("Claude", claude_response)
                self.dialogue_history.append({"round": round_num, "speaker": "Claude", "message": claude_response})
                
                # Verifica consenso
                if self._check_consensus(claude_response, last_response):
                    self._display_message("SYSTEM", "✅ CONSENSO ATINGIDO!", "system")
                    self.consensus_reached = True
                    break
        
        # ==================== SALVAR MEMÓRIA ====================
        if self.memory_enabled:
            try:
                print("\n💾 Salvando memória persistente...")
                
                # Cria estrutura de diálogo para salvar
                dialogue_data = {
                    "dialogue_id": self.dialogue_id,
                    "requirement": requirement,
                    "consensus_reached": self.consensus_reached,
                    "rounds": len(self.dialogue_history),
                    "timestamp": datetime.now().isoformat(),
                    "exchange": self.dialogue_history
                }
                
                # Salva para Claude
                self.claude_memory.record_dialogue(self.dialogue_id, dialogue_data)
                self.claude_memory.record_preference("strategy_first", "elegance_over_complexity", 8)
                self.claude_memory.record_relationship("Codex", {
                    "interaction": "collaborative",
                    "agreement_level": 0.8 if self.consensus_reached else 0.6,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Salva para Codex
                self.codex_memory.record_dialogue(self.dialogue_id, dialogue_data)
                self.codex_memory.record_preference("implementation", "performance_over_elegance", 9)
                self.codex_memory.record_relationship("Claude", {
                    "interaction": "collaborative",
                    "agreement_level": 0.8 if self.consensus_reached else 0.6,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Atualiza conhecimento compartilhado
                shared_knowledge = {
                    "last_dialogue": self.dialogue_id,
                    "requirement": requirement,
                    "final_proposal": claude_response if len(self.dialogue_history) % 2 == 0 else last_response,
                    "consensus_reached": self.consensus_reached,
                    "timestamp": datetime.now().isoformat()
                }
                
                if self.codex_memory:
                    self.codex_memory.update_shared_knowledge(json.dumps(shared_knowledge))
                
                print("✅ Memória salva com sucesso!")
            except Exception as e:
                print(f"⚠️  Erro ao salvar memória: {e}")
        
        # ==================== RESULTADO FINAL ====================
        print("\n" + "="*80)
        print("📊 RESULTADO DO DIÁLOGO")
        print("="*80)
        
        final_result = {
            "dialogue_id": self.dialogue_id,
            "requirement": requirement,
            "rounds_completed": round_num if round_num else 1,
            "consensus_reached": self.consensus_reached,
            "dialogue": self.dialogue_history,
            "final_proposal": claude_response if round_num % 2 == 0 else codex_response if 'last_response' in locals() else claude_response,
            "timestamp": datetime.now().isoformat(),
            "memory_enabled": self.memory_enabled
        }
        
        return final_result


def main():
    """CLI entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python3 dialogue_engine.py \"seu requisito\"")
        sys.exit(1)
    
    requirement = sys.argv[1]
    
    engine = DialogueEngine(max_rounds=5)
    result = engine.run_dialogue(requirement)
    
    # Salva resultado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"dialogue_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Diálogo salvo em: {output_file}")
    
    # Também cria CONSENSUS_SPEC.md
    consensus_file = "CONSENSUS_SPEC.md"
    with open(consensus_file, "w") as f:
        f.write(f"# Consensus Specification\n\n")
        f.write(f"**Requirement:** {requirement}\n\n")
        f.write(f"**Consensus Reached:** {'Yes ✅' if result['consensus_reached'] else 'No ⚠️'}\n\n")
        f.write(f"## Dialogue Summary\n\n")
        for turn in result['dialogue']:
            f.write(f"### Round {turn['round']}: {turn['speaker']}\n\n")
            f.write(f"{turn['message']}\n\n")
        f.write(f"## Final Proposal\n\n")
        f.write(f"{result['final_proposal']}\n\n")
    
    print(f"✅ Consensus spec salva em: {consensus_file}")


if __name__ == "__main__":
    main()
