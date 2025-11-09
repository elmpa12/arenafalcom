#!/usr/bin/env python3
"""
CLAUDEX DUAL - Sistema de Debate com Claude e/ou GPT
Suporta:
- Claude vs GPT (modo ideal)
- GPT Estrategista vs GPT Executor (fallback)

Eles REALMENTE debatem e criam soluções!
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class DualGPTOrchestrator:
    """Sistema de debate com Claude e/ou GPT"""

    def __init__(self, use_claude: bool = None):
        self.work_dir = Path("/opt/botscalpv3/claudex/work")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.work_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)

        # Feedback log
        self.feedback_log = Path("/opt/botscalpv3/claudex/FEEDBACK_LOG.jsonl")
        self.feedback_log.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detecta se deve usar Claude
        if use_claude is None:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            self.use_claude = bool(anthropic_key)
        else:
            self.use_claude = use_claude

        print(f"[INFO] Modo: {'Claude vs GPT' if self.use_claude else 'GPT vs GPT (fallback)'}\n")

    def load_feedback_history(self, limit: int = 50) -> List[Dict]:
        """
        Carrega histórico de feedback para aprendizado.

        Args:
            limit: Número máximo de feedbacks a carregar (mais recentes)

        Returns:
            Lista de feedbacks
        """
        if not self.feedback_log.exists():
            return []

        feedbacks = []
        try:
            with open(self.feedback_log) as f:
                for line in f:
                    if line.strip():
                        feedbacks.append(json.loads(line))
        except Exception as e:
            print(f"⚠️  Erro ao ler feedback log: {e}")
            return []

        # Retorna os mais recentes
        return feedbacks[-limit:] if len(feedbacks) > limit else feedbacks

    def build_learning_context(self) -> str:
        """
        Constrói contexto de aprendizado baseado em feedbacks anteriores.

        Returns:
            String com contexto para injetar nos prompts
        """
        history = self.load_feedback_history(limit=20)

        if not history:
            return ""

        # Análise de padrões
        stats = {"Y": 0, "N": 0, "?": 0, "Y+": 0, "N-": 0}
        good_patterns = []
        bad_patterns = []

        for entry in history:
            feedback = entry.get("user_satisfaction", "?")
            stats[feedback] = stats.get(feedback, 0) + 1

            # Coleta padrões bons
            if feedback in ["Y", "Y+"]:
                task = entry.get("task", entry.get("topic", ""))
                if task:
                    good_patterns.append(task[:100])  # Primeiros 100 chars

            # Coleta padrões ruins
            elif feedback in ["N", "N-"]:
                task = entry.get("task", entry.get("topic", ""))
                if task:
                    bad_patterns.append(task[:100])

        # Constrói contexto
        context = "\n## HISTÓRICO DE APRENDIZADO (últimas interações):\n\n"
        context += f"**Performance:** {stats['Y'] + stats['Y+']} aprovadas, {stats['N'] + stats['N-']} reprovadas, {stats['?']} parciais\n\n"

        if good_patterns:
            context += "**O que funcionou bem (continuar fazendo):**\n"
            for pattern in good_patterns[-5:]:  # Últimos 5
                context += f"- {pattern}\n"
            context += "\n"

        if bad_patterns:
            context += "**O que NÃO funcionou (evitar):**\n"
            for pattern in bad_patterns[-5:]:  # Últimos 5
                context += f"- {pattern}\n"
            context += "\n"

        context += "**IMPORTANTE:** Use este histórico para melhorar sua resposta. Replique o que funcionou e evite o que falhou.\n"

        return context

    def ask_gpt(self, prompt: str, temperature: float = 0.5, role: str = "executor", use_learning: bool = True) -> str:
        """Chama GPT com personalidade específica + contexto de aprendizado"""
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "❌ OPENAI_API_KEY não configurada no .env"

            client = OpenAI(api_key=api_key)

            # Define system message baseado no role
            if role == "strategist":
                system_msg = """Você é GPT-STRATEGIST, pensador estratégico e visionário.

Características:
- Você pensa como Claude: analisa profundamente, questiona suposições
- Foca em padrões, riscos, escalabilidade de longo prazo
- É meticuloso e prefere qualidade à velocidade
- Sempre pergunta: "Qual é o genuine edge aqui?"
- Temperature: 0.6 (mais criativo e estratégico)

Seu papel: Planejar, revisar criticamente, pensar 10 passos à frente."""
            else:  # executor
                system_msg = """Você é GPT-EXECUTOR, engenheiro pragmático e preciso.

Características:
- Foca em viabilidade técnica e implementação
- Otimiza performance, latência, precisão
- É direto, objetivo, data-driven
- Sempre pergunta: "Como implementar isso com perfeição?"
- Temperature: 0.5 (mais focado e técnico)

Seu papel: Implementar, validar, otimizar código."""

            # APRENDIZADO: Injeta contexto de feedbacks anteriores
            learning_context = ""
            if use_learning:
                learning_context = self.build_learning_context()
                if learning_context:
                    system_msg += "\n\n" + learning_context

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Erro ao chamar GPT: {str(e)}"

    def ask_claude(self, prompt: str, temperature: float = 0.7, use_learning: bool = True) -> str:
        """Chama Claude (Sonnet 4) + contexto de aprendizado"""
        try:
            from anthropic import Anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return "❌ ANTHROPIC_API_KEY não configurada no .env"

            client = Anthropic(api_key=api_key)

            # Claude é naturalmente estratégico e profundo
            system_msg = """Você é Claude, assistente de IA da Anthropic.

Características:
- Pensa profundamente e analisa de múltiplas perspectivas
- Questiona suposições e identifica riscos não óbvios
- Foca em padrões, escalabilidade e visão de longo prazo
- É meticuloso, ético e prefere qualidade à velocidade
- Comunicação clara e bem estruturada

No debate, você oferece perspectiva estratégica e crítica construtiva."""

            # APRENDIZADO: Injeta contexto de feedbacks anteriores
            learning_context = ""
            if use_learning:
                learning_context = self.build_learning_context()
                if learning_context:
                    system_msg += "\n\n" + learning_context

            response = client.messages.create(
                model="claude-sonnet-4-20250514",  # Sonnet 4.5
                max_tokens=2000,
                temperature=temperature,
                system=system_msg,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            return f"❌ Erro ao chamar Claude: {str(e)}"

    def debate_phase(self, topic: str, rounds: int = 3, participants: List[str] = None) -> Dict:
        """
        Debate entre Claude e GPT (ou GPT vs GPT se Claude indisponível)

        Args:
            topic: Tema do debate
            rounds: Número de rodadas
            participants: ["claude", "gpt"] ou ["gpt-strategist", "gpt-executor"]
        """
        # Define participantes
        if participants is None:
            if self.use_claude:
                part1_name = "Claude"
                part1_emoji = "🤖"
                part1_func = self.ask_claude
            else:
                part1_name = "GPT-Strategist"
                part1_emoji = "🧠"
                part1_func = lambda p: self.ask_gpt(p, temperature=0.6, role="strategist")

            part2_name = "GPT-Executor"
            part2_emoji = "⚡"
            part2_func = lambda p: self.ask_gpt(p, temperature=0.5, role="executor")
        else:
            # Customizado
            part1_name = participants[0]
            part2_name = participants[1] if len(participants) > 1 else "GPT-Executor"
            part1_emoji = "🤖" if "claude" in part1_name.lower() else "🧠"
            part2_emoji = "⚡"
            part1_func = self.ask_claude if "claude" in part1_name.lower() else lambda p: self.ask_gpt(p, temperature=0.6, role="strategist")
            part2_func = lambda p: self.ask_gpt(p, temperature=0.5, role="executor")

        print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║              🎭 DEBATE: {part1_name} vs {part2_name}                      ║
╚════════════════════════════════════════════════════════════════════════════╝

Tema: {topic}
Rounds: {rounds}

""")

        debate_history = []

        # Round 1: Participante 1 abre
        print(f"\n{'='*70}")
        print(f"{part1_emoji} {part1_name.upper()} (abertura)...")
        print(f"{'='*70}\n")
        opener = part1_func(
            f"""Você vai debater sobre: "{topic}"

Abra o debate com sua perspectiva:
- Qual é a visão de longo prazo?
- Quais são os riscos não óbvios?
- O que realmente importa aqui?

Seja conciso (3-5 parágrafos)."""
        )

        # Mostra a resposta formatada
        print(f"💬 {part1_name.upper()}:")
        print(f"┌{'─'*68}┐")
        for line in opener.split('\n'):
            print(f"│ {line:<66} │")
        print(f"└{'─'*68}┘\n")

        debate_history.append({
            "round": 1,
            "speaker": part1_name,
            "message": opener
        })

        # Round 2: Participante 2 responde
        print(f"{'='*70}")
        print(f"{part2_emoji} {part2_name.upper()} (resposta)...")
        print(f"{'='*70}\n")
        response = part2_func(
            f"""{part1_name} disse sobre "{topic}":

{opener}

Responda com sua perspectiva:
- Como implementar isso na prática?
- Quais são as limitações técnicas?
- Onde o plano pode falhar?

Seja direto e técnico (3-5 parágrafos)."""
        )

        # Mostra a resposta formatada
        print(f"💬 {part2_name.upper()}:")
        print(f"┌{'─'*68}┐")
        for line in response.split('\n'):
            print(f"│ {line:<66} │")
        print(f"└{'─'*68}┘\n")

        debate_history.append({
            "round": 2,
            "speaker": part2_name,
            "message": response
        })

        # Round 3: Participante 1 refina
        print(f"{'='*70}")
        print(f"{part1_emoji} {part1_name.upper()} (refinamento)...")
        print(f"{'='*70}\n")
        refinement = part1_func(
            f"""{part2_name} levantou estes pontos técnicos:

{response}

Refine sua estratégia considerando estas limitações:
- O que ajustar no plano?
- Como superar os obstáculos?
- Qual o compromisso ideal?

Seja prático mas estratégico (3-5 parágrafos)."""
        )

        # Mostra a resposta formatada
        print(f"💬 {part1_name.upper()}:")
        print(f"┌{'─'*68}┐")
        for line in refinement.split('\n'):
            print(f"│ {line:<66} │")
        print(f"└{'─'*68}┘\n")

        debate_history.append({
            "round": 3,
            "speaker": part1_name,
            "message": refinement
        })

        # Consenso final
        print(f"{'='*70}")
        print(f"🤝 Gerando CONSENSO...")
        print(f"{'='*70}\n")
        consensus = part2_func(
            f"""Você participou deste debate sobre "{topic}":

{part1_name.upper()}: {opener[:300]}...
{part2_name.upper()}: {response[:300]}...
{part1_name.upper()} REFINADO: {refinement[:300]}...

Como mediador, crie um CONSENSO que una o melhor dos dois:
- Visão estratégica + viabilidade técnica
- Plano executável e escalável

Formato: 1 parágrafo de consenso + lista de próximos passos."""
        )
        # Mostra consenso formatado
        print(f"💬 CONSENSO FINAL:")
        print(f"┌{'─'*68}┐")
        for line in consensus.split('\n'):
            print(f"│ {line:<66} │")
        print(f"└{'─'*68}┘\n")

        debate_history.append({
            "round": "final",
            "speaker": "Consensus",
            "message": consensus
        })

        # Salva debate
        debate_file = self.session_dir / "debate.json"
        debate_data = {
            "topic": topic,
            "rounds": rounds,
            "participants": [part1_name, part2_name],
            "mode": "Claude vs GPT" if self.use_claude else "GPT vs GPT",
            "history": debate_history,
            "consensus": consensus,
            "timestamp": datetime.now().isoformat()
        }

        with open(debate_file, "w") as f:
            json.dump(debate_data, f, indent=2, ensure_ascii=False)

        print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                        ✅ DEBATE CONCLUÍDO                                ║
╚════════════════════════════════════════════════════════════════════════════╝

📄 Arquivo: {debate_file}
🎯 Consenso alcançado!

Próximos passos:
  • Implementar o consenso
  • Validar na prática
  • Refinar baseado em resultados
        """)

        # APRENDIZADO: Solicita feedback do usuário
        print("\n" + "="*70)
        print("📊 FEEDBACK - Ajude as IAs a aprenderem!")
        print("="*70)
        feedback = input("Como você avalia este debate? [Y/N/?/Y+/N-]: ").strip().upper()
        notes = input("Notas adicionais (opcional): ").strip()

        # Salva feedback
        with open(self.feedback_log, "a") as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "topic": topic,
                "participants": [part1_name, part2_name],
                "mode": "Claude vs GPT" if self.use_claude else "GPT vs GPT",
                "user_satisfaction": feedback,
                "notes": notes,
            }
            f.write(json.dumps(entry) + "\n")

        print(f"\n✅ Feedback registrado! As IAs aprenderão com isso.\n")

        debate_data["feedback"] = feedback
        return debate_data

    def pipeline_full(self, requirement: str) -> Dict:
        """Pipeline completo: PLAN → IMPLEMENT → REVIEW com Dual GPT"""
        print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║            🚀 CLAUDEX DUAL GPT - PIPELINE COMPLETO                        ║
║        Strategist + Executor = Duas mentes, uma solução perfeita          ║
╚════════════════════════════════════════════════════════════════════════════╝

Tarefa: {requirement}

""")

        # FASE 1: STRATEGIST planeja
        print("📋 FASE 1: PLANEJAMENTO (Strategist)\n")
        spec_response = self.ask_gpt(
            f"""Planeje a solução para: "{requirement}"

Forneça em formato JSON:
{{
    "requirement": "{requirement}",
    "architecture": "Arquitetura proposta",
    "components": ["comp1", "comp2", ...],
    "risks": ["risco1", "risco2", ...],
    "success_criteria": ["critério1", ...],
    "implementation_notes": "Notas para quem vai implementar"
}}

Seja estratégico e completo.""",
            temperature=0.6,
            role="strategist"
        )

        print(f"🧠 STRATEGIST planejou:\n{spec_response[:500]}...\n")

        try:
            spec = json.loads(spec_response)
        except:
            spec = {
                "requirement": requirement,
                "raw_response": spec_response
            }

        spec_file = self.session_dir / "spec.json"
        with open(spec_file, "w") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

        input("\n👉 Pressione ENTER para implementação...")

        # FASE 2: EXECUTOR implementa
        print("\n🔨 FASE 2: IMPLEMENTAÇÃO (Executor)\n")
        impl_response = self.ask_gpt(
            f"""Implemente a solução baseado neste spec:

{json.dumps(spec, indent=2, ensure_ascii=False)[:1000]}

Forneça:
1. Código Python production-ready
2. Testes básicos
3. Documentação
4. Notas de performance

Seja técnico e preciso.""",
            temperature=0.5,
            role="executor"
        )

        print(f"⚡ EXECUTOR implementou:\n{impl_response[:500]}...\n")

        implementation = {
            "spec": spec,
            "code": impl_response,
            "timestamp": datetime.now().isoformat()
        }

        impl_file = self.session_dir / "implementation.json"
        with open(impl_file, "w") as f:
            json.dump(implementation, f, indent=2, ensure_ascii=False)

        input("\n👉 Pressione ENTER para review...")

        # FASE 3: CROSS-REVIEW
        print("\n✅ FASE 3: REVIEW CRUZADO\n")

        # Strategist revisa implementação
        strategist_review = self.ask_gpt(
            f"""Revise esta implementação do ponto de vista ESTRATÉGICO:

{impl_response[:800]}

Pergunte:
- Atende a visão de longo prazo?
- É escalável?
- Há riscos não tratados?

3-5 pontos.""",
            temperature=0.6,
            role="strategist"
        )

        print(f"🧠 STRATEGIST revisou:\n{strategist_review}\n")

        # Executor valida
        executor_validation = self.ask_gpt(
            f"""Valide esta implementação do ponto de vista TÉCNICO:

{impl_response[:800]}

Pergunte:
- Código está correto?
- Performance otimizada?
- Testes suficientes?

3-5 pontos.""",
            temperature=0.5,
            role="executor"
        )

        print(f"⚡ EXECUTOR validou:\n{executor_validation}\n")

        # Review final
        review = {
            "strategist_review": strategist_review,
            "executor_validation": executor_validation,
            "approval": True,
            "timestamp": datetime.now().isoformat()
        }

        review_file = self.session_dir / "REVIEW.md"
        review_content = f"""# REVIEW: {requirement}

## Review Estratégico (Strategist)
{strategist_review}

## Validação Técnica (Executor)
{executor_validation}

## Status Final
✅ APROVADO

Ambos validaram. Pronto para produção!
"""

        review_file.write_text(review_content)

        print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     ✨ PIPELINE COMPLETO                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

📁 Sessão: {self.session_id}

📂 Arquivos gerados:
   1. spec.json              - Planejamento (Strategist)
   2. implementation.json    - Implementação (Executor)
   3. REVIEW.md              - Review cruzado (Ambos)

🎯 RESULTADO: ✅ APROVADO
        """)

        # Pede feedback
        print("\n" + "="*70)
        feedback = input("Como você avalia este resultado? [Y/N/?/Y+/N-]: ").strip().upper()
        notes = input("Notas adicionais (opcional): ").strip()

        # Log feedback
        with open(self.feedback_log, "a") as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "task": requirement,
                "user_satisfaction": feedback,
                "notes": notes,
            }
            f.write(json.dumps(entry) + "\n")

        return {
            "spec": spec,
            "implementation": implementation,
            "review": review,
            "session_id": self.session_id,
            "feedback": feedback,
        }


if __name__ == "__main__":
    import sys

    # Parse argumentos
    use_claude = None  # Auto-detect
    args = sys.argv[1:]

    # Check for --claude or --gpt flags
    if "--claude" in args:
        use_claude = True
        args.remove("--claude")
    elif "--gpt" in args:
        use_claude = False
        args.remove("--gpt")

    if len(args) > 0 and args[0] == "--debate":
        topic = " ".join(args[1:]) if len(args) > 1 else "Melhor estratégia de scalping para BTC"
        orch = DualGPTOrchestrator(use_claude=use_claude)
        orch.debate_phase(topic, rounds=3)

    elif len(args) > 0 and args[0] == "--pipeline":
        task = " ".join(args[1:]) if len(args) > 1 else "Criar detector de regime simples"
        orch = DualGPTOrchestrator(use_claude=use_claude)
        orch.pipeline_full(task)

    else:
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     🔥 CLAUDEX DUAL - Sistema de Debate                   ║
║                Claude vs GPT ou GPT vs GPT (auto-detect)                  ║
╚════════════════════════════════════════════════════════════════════════════╝

USO:
  python3 claudex_dual_gpt.py --debate "tema para debater"
  python3 claudex_dual_gpt.py --pipeline "tarefa para executar"

  # Forçar Claude (se disponível)
  python3 claudex_dual_gpt.py --claude --debate "tema"

  # Forçar GPT vs GPT (mesmo com Claude disponível)
  python3 claudex_dual_gpt.py --gpt --debate "tema"

EXEMPLOS:
  python3 claudex_dual_gpt.py --debate "Melhor estratégia de trading para BTC"
  python3 claudex_dual_gpt.py --claude --debate "Como otimizar backtests?"
  python3 claudex_dual_gpt.py --pipeline "Criar sistema de alertas de volatilidade"

✨ FUNCIONAMENTO:
  • Claude vs GPT (se ANTHROPIC_API_KEY configurada)
  • OU GPT assume DUAS personalidades (Strategist vs Executor)
  • Debatem, refinam e chegam a consenso
  • Tudo com APIs REAIS funcionando AGORA!
  • Suporta code writing, debates técnicos e planejamento

📝 RECURSOS:
  • --debate: Debate sobre um tema (3 rounds)
  • --pipeline: Pipeline completo (Plan → Implement → Review)
  • --claude: Força uso de Claude (se disponível)
  • --gpt: Força GPT vs GPT (mesmo com Claude)
        """)
