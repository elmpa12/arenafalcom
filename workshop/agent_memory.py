#!/usr/bin/env python3
"""
agent_memory.py — Persistent Memory System para Claude e Codex
Eles guardam tudo: diálogos, specs, decisões, preferências
Para sempre lembrar um do outro quando se encontram de novo
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class AgentMemory:
    """Sistema de memória persistente para agentes"""
    
    def __init__(self, agent_name: str, memory_dir: str = "/opt/botscalpv3/memory_store"):
        self.agent_name = agent_name
        self.memory_dir = Path(memory_dir)
        self.agent_dir = self.memory_dir / agent_name
        
        # Criar estrutura de diretórios
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        (self.agent_dir / "dialogues").mkdir(exist_ok=True)
        (self.agent_dir / "specs").mkdir(exist_ok=True)
        (self.agent_dir / "decisions").mkdir(exist_ok=True)
        (self.agent_dir / "preferences").mkdir(exist_ok=True)
        (self.agent_dir / "relationships").mkdir(exist_ok=True)
        (self.memory_dir / "shared").mkdir(exist_ok=True)
        
        # Carregar perfil do agente (imutável)
        self.profile = self._load_agent_profile()
        
        # Carregar histórico pessoal
        self.dialogue_history = self._load_dialogue_history()
        self.past_specs = self._load_past_specs()
        self.past_decisions = self._load_past_decisions()
        self.preferences = self._load_preferences()
        self.relationships = self._load_relationships()
    
    def _load_agent_profile(self) -> Dict:
        """Carrega perfil imutável do agente"""
        profile_file = self.agent_dir / "PROFILE.json"
        
        if profile_file.exists():
            with open(profile_file, "r") as f:
                return json.load(f)
        
        # Profile padrão
        profiles = {
            "Claude": {
                "name": "Claude",
                "role": "Strategist",
                "personality": "Visão holística, pensador estratégico, aprecia nuances",
                "strengths": ["Big picture thinking", "Long-term strategy", "Conceptual clarity"],
                "weaknesses": ["Sometimes imprecise technically", "Can miss implementation details"],
                "temperature": 0.6,
                "created_date": datetime.now().isoformat(),
                "version": 1.0
            },
            "Codex": {
                "name": "Codex",
                "role": "Engineer",
                "personality": "Pragmático, data-driven, foco em viabilidade",
                "strengths": ["Technical precision", "Performance optimization", "Implementation"],
                "weaknesses": ["Sometimes misses strategic vision", "Overly focused on constraints"],
                "temperature": 0.5,
                "created_date": datetime.now().isoformat(),
                "version": 1.0
            }
        }
        
        profile = profiles.get(self.agent_name, {})
        
        # Salvar para futuro
        with open(profile_file, "w") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        
        return profile
    
    def _load_dialogue_history(self) -> List[Dict]:
        """Carrega histórico de diálogos passados"""
        history_file = self.agent_dir / "dialogues" / "history.jsonl"
        
        if not history_file.exists():
            return []
        
        history = []
        with open(history_file, "r") as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
        
        return history
    
    def _load_past_specs(self) -> Dict:
        """Carrega specs criadas/revisadas no passado"""
        specs_file = self.agent_dir / "specs" / "index.json"
        
        if not specs_file.exists():
            return {}
        
        with open(specs_file, "r") as f:
            return json.load(f)
    
    def _load_past_decisions(self) -> Dict:
        """Carrega decisões tomadas no passado"""
        decisions_file = self.agent_dir / "decisions" / "index.json"
        
        if not decisions_file.exists():
            return {}
        
        with open(decisions_file, "r") as f:
            return json.load(f)
    
    def _load_preferences(self) -> Dict:
        """Carrega preferências e padrões de comportamento"""
        prefs_file = self.agent_dir / "preferences" / "index.json"
        
        if not prefs_file.exists():
            return {}
        
        with open(prefs_file, "r") as f:
            return json.load(f)
    
    def _load_relationships(self) -> Dict:
        """Carrega relacionamento com outros agentes"""
        rel_file = self.agent_dir / "relationships" / "index.json"
        
        if not rel_file.exists():
            return {}
        
        with open(rel_file, "r") as f:
            return json.load(f)
    
    def record_dialogue(self, dialogue_id: str, dialogue_data: Dict):
        """Registra um diálogo na memória"""
        entry = {
            "dialogue_id": dialogue_id,
            "timestamp": datetime.now().isoformat(),
            "agent_role": self.profile.get("role"),
            "dialogue": dialogue_data,
            "my_contribution": dialogue_data.get(f"{self.agent_name}_response"),
            "other_agent": "Codex" if self.agent_name == "Claude" else "Claude"
        }
        
        history_file = self.agent_dir / "dialogues" / "history.jsonl"
        with open(history_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Também salva resumo em arquivo separado
        summary_file = self.agent_dir / "dialogues" / f"{dialogue_id}.json"
        with open(summary_file, "w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
    
    def record_spec(self, spec_id: str, spec_content: str, contribution: str = "reviewed"):
        """Registra um spec na memória"""
        spec_hash = hashlib.md5(spec_content.encode()).hexdigest()[:8]
        
        self.past_specs[spec_id] = {
            "timestamp": datetime.now().isoformat(),
            "hash": spec_hash,
            "contribution": contribution,  # "created" ou "reviewed"
            "size": len(spec_content)
        }
        
        # Salvar spec completo
        spec_file = self.agent_dir / "specs" / f"{spec_id}.md"
        with open(spec_file, "w") as f:
            f.write(spec_content)
        
        # Salvar índice
        index_file = self.agent_dir / "specs" / "index.json"
        with open(index_file, "w") as f:
            json.dump(self.past_specs, f, indent=2, ensure_ascii=False)
    
    def record_decision(self, decision_id: str, decision_data: Dict):
        """Registra uma decisão importante"""
        self.past_decisions[decision_id] = {
            "timestamp": datetime.now().isoformat(),
            "reasoning": decision_data.get("reasoning"),
            "choice": decision_data.get("choice"),
            "alternatives_considered": decision_data.get("alternatives", []),
            "impact": decision_data.get("impact")
        }
        
        # Salvar índice
        index_file = self.agent_dir / "decisions" / "index.json"
        with open(index_file, "w") as f:
            json.dump(self.past_decisions, f, indent=2, ensure_ascii=False)
    
    def record_preference(self, category: str, preference: str, strength: int = 5):
        """Registra uma preferência (1-10, onde 10 é fortíssimo)"""
        if category not in self.preferences:
            self.preferences[category] = []
        
        self.preferences[category].append({
            "preference": preference,
            "strength": min(10, max(1, strength)),
            "timestamp": datetime.now().isoformat()
        })
        
        # Salvar
        index_file = self.agent_dir / "preferences" / "index.json"
        with open(index_file, "w") as f:
            json.dump(self.preferences, f, indent=2, ensure_ascii=False)
    
    def record_relationship(self, other_agent: str, interaction: Dict):
        """Registra uma interação/impressão de outro agente"""
        if other_agent not in self.relationships:
            self.relationships[other_agent] = []
        
        self.relationships[other_agent].append({
            "timestamp": datetime.now().isoformat(),
            "observation": interaction.get("observation"),
            "sentiment": interaction.get("sentiment"),  # positive/neutral/negative
            "agrees_on": interaction.get("agrees_on", []),
            "disagrees_on": interaction.get("disagrees_on", [])
        })
        
        # Salvar
        index_file = self.agent_dir / "relationships" / "index.json"
        with open(index_file, "w") as f:
            json.dump(self.relationships, f, indent=2, ensure_ascii=False)
    
    def get_context_for_dialogue(self) -> str:
        """Gera contexto histórico para incluir em novo diálogo"""
        context_lines = [
            f"=== MEMORY CONTEXT FOR {self.agent_name.upper()} ===\n",
            f"Profile: {self.profile.get('role')}\n",
            f"Personality: {self.profile.get('personality')}\n"
        ]
        
        # Últimos 3 diálogos
        if self.dialogue_history:
            context_lines.append("\nRecent dialogues:")
            for dialogue in self.dialogue_history[-3:]:
                context_lines.append(f"  - {dialogue.get('dialogue_id')}: {dialogue.get('timestamp')}")
        
        # Preferências
        if self.preferences:
            context_lines.append("\nPreferences:")
            for category, prefs in self.preferences.items():
                strong_prefs = [p for p in prefs if p.get('strength', 0) >= 7]
                if strong_prefs:
                    context_lines.append(f"  {category}: {strong_prefs[0].get('preference')}")
        
        # Relacionamento com outro agente
        other_agent = "Codex" if self.agent_name == "Claude" else "Claude"
        if other_agent in self.relationships:
            context_lines.append(f"\nRelationship with {other_agent}:")
            recent = self.relationships[other_agent][-2:]
            for rel in recent:
                context_lines.append(f"  {rel.get('sentiment')}: {rel.get('observation')}")
        
        return "\n".join(context_lines)
    
    def get_shared_context(self) -> str:
        """Retorna contexto compartilhado (ambos agentes conhecem)"""
        shared_file = self.memory_dir / "shared" / "common_knowledge.md"
        
        if not shared_file.exists():
            return ""
        
        with open(shared_file, "r") as f:
            return f.read()
    
    def update_shared_knowledge(self, knowledge: str):
        """Atualiza conhecimento compartilhado"""
        shared_file = self.memory_dir / "shared" / "common_knowledge.md"
        
        with open(shared_file, "a") as f:
            f.write(f"\n\n--- Updated {datetime.now().isoformat()} ---\n")
            f.write(knowledge)
    
    def create_memory_report(self) -> str:
        """Gera relatório visual da memória do agente"""
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║  AGENT MEMORY REPORT: {self.agent_name.upper()}
╚════════════════════════════════════════════════════════════════╝

PROFILE
───────
Name: {self.profile.get('name')}
Role: {self.profile.get('role')}
Personality: {self.profile.get('personality')}

STATISTICS
──────────
Dialogues participated: {len(self.dialogue_history)}
Specs created/reviewed: {len(self.past_specs)}
Decisions made: {len(self.past_decisions)}
Strong preferences: {sum(1 for cat in self.preferences.values() for p in cat if p.get('strength', 0) >= 7)}

RECENT DIALOGUES
────────────────
"""
        for dialogue in self.dialogue_history[-3:]:
            report += f"• {dialogue.get('dialogue_id')} ({dialogue.get('timestamp')})\n"
        
        report += f"\nSTRONG PREFERENCES\n──────────────────\n"
        for category, prefs in self.preferences.items():
            strong = [p for p in prefs if p.get('strength', 0) >= 7]
            if strong:
                report += f"• {category}: {strong[0].get('preference')} (strength: {strong[0].get('strength')})\n"
        
        other_agent = "Codex" if self.agent_name == "Claude" else "Claude"
        if other_agent in self.relationships:
            report += f"\nRELATIONSHIP WITH {other_agent}\n"
            report += "─" * 30 + "\n"
            for rel in self.relationships[other_agent][-3:]:
                report += f"• [{rel.get('sentiment')}] {rel.get('observation')}\n"
        
        return report


def main():
    """Test memory system"""
    print("Testing Agent Memory System...\n")
    
    # Create memories for both agents
    claude_mem = AgentMemory("Claude")
    codex_mem = AgentMemory("Codex")
    
    # Show profiles
    print(claude_mem.create_memory_report())
    print(codex_mem.create_memory_report())
    
    # Record a dialogue
    dialogue_data = {
        "Claude_response": "Kalman filter is best for smoothing",
        "Codex_response": "Agreed, but needs optimization"
    }
    claude_mem.record_dialogue("dialogue_001", dialogue_data)
    codex_mem.record_dialogue("dialogue_001", dialogue_data)
    
    print("\n✅ Memory system initialized successfully!")


if __name__ == "__main__":
    main()
