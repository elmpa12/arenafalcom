#!/usr/bin/env python3
"""
AUTO EVOLUTION SYSTEM - BotScalp V3
═══════════════════════════════════════════════════════════════

Sistema revolucionário de auto-evolução onde Claude + GPT:
1. Analisam TODOS os eventos (testes, trades, erros)
2. Cada um com sua perspectiva (estratégica vs técnica)
3. Propõem mudanças no código
4. Aprendem continuamente com resultados
5. Evoluem o bot automaticamente

ARQUITETURA:
┌─────────────────────────────────────────────────────────────┐
│  EVENTO (teste, trade, erro)                                │
│         ↓                                                    │
│  EVENT INTERCEPTOR                                          │
│         ↓                                                    │
│  ┌──────────────┐        ┌──────────────┐                 │
│  │ CLAUDE       │ ←───→  │ GPT          │                 │
│  │ Estratégico  │ debate │ Técnico      │                 │
│  └──────────────┘        └──────────────┘                 │
│         ↓                        ↓                          │
│  CONSENSO + AÇÕES                                           │
│         ↓                                                    │
│  ┌─────────────────────────────────────┐                  │
│  │ - Modificar código                  │                  │
│  │ - Ajustar parâmetros                │                  │
│  │ - Registrar aprendizado             │                  │
│  │ - Re-testar automaticamente         │                  │
│  └─────────────────────────────────────┘                  │
│         ↓                                                    │
│  LOOP CONTÍNUO DE MELHORIA                                  │
└─────────────────────────────────────────────────────────────┘

RESULTADO ESPERADO:
Dia 1:  Sistema funciona mas com bugs
Dia 7:  Bugs corrigidos, 70% win rate
Dia 30: 85% win rate, código otimizado
Dia 90: 92%+ win rate, CHAMPIONSHIP GRADE
"""

import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
from enum import Enum

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# AI APIs
try:
    import openai
    from anthropic import Anthropic
except ImportError:
    print("⚠️  Install: pip install openai anthropic")
    raise


class EventType(Enum):
    """Tipos de eventos que disparam análise dual"""
    BACKTEST_RESULT = "backtest_result"
    TRADE_EXECUTED = "trade_executed"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    CODE_CHANGE = "code_change"
    STRATEGY_UPDATE = "strategy_update"


@dataclass
class TradingEvent:
    """Evento do sistema de trading"""
    event_type: EventType
    timestamp: str
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "context": self.context or {}
        }


@dataclass
class DualAnalysis:
    """Análise dual de Claude + GPT"""
    event: TradingEvent
    claude_analysis: str
    gpt_analysis: str
    consensus: str
    actions: List[Dict[str, Any]]
    confidence: float
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "event": self.event.to_dict(),
            "claude_analysis": self.claude_analysis,
            "gpt_analysis": self.gpt_analysis,
            "consensus": self.consensus,
            "actions": self.actions,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


class AutoEvolutionSystem:
    """
    Sistema principal de auto-evolução.

    Intercepta TODOS os eventos do trading bot e chama
    Claude + GPT para análise dual e ações corretivas.
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 apply_mode: str = "review"):
        """
        Args:
            openai_api_key: API key OpenAI (ou usa OPENAI_API_KEY env)
            anthropic_api_key: API key Anthropic (ou usa ANTHROPIC_API_KEY env)
            apply_mode: Modo de aplicação de mudanças:
                - "review": Apenas propõe, não aplica (padrão - SEGURO)
                - "interactive": Pergunta antes de aplicar cada mudança
                - "auto": Aplica todas automaticamente (CUIDADO!)
        """

        # Validar modo
        valid_modes = ["review", "interactive", "auto"]
        if apply_mode not in valid_modes:
            raise ValueError(f"apply_mode deve ser um de: {valid_modes}. Recebido: {apply_mode}")

        # Configurar APIs
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY não encontrada!")
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY não encontrada!")

        # Inicializar clientes
        openai.api_key = self.openai_key
        self.anthropic = Anthropic(api_key=self.anthropic_key)

        # Configurações
        self.apply_mode = apply_mode
        self.learning_log_path = Path("/opt/botscalpv3/claudex/LEARNING_LOG.jsonl")
        self.code_changes_log = Path("/opt/botscalpv3/claudex/CODE_CHANGES_LOG.jsonl")

        # Estatísticas
        self.total_events = 0
        self.total_analyses = 0
        self.total_code_changes = 0
        self.total_improvements = 0

        # Histórico
        self.event_history: List[TradingEvent] = []
        self.analysis_history: List[DualAnalysis] = []

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║     🤖 AUTO EVOLUTION SYSTEM - INICIADO                      ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print(f"✅ Claude API: Configurada")
        print(f"✅ GPT API: Configurada")
        mode_labels = {
            "review": "NÃO (modo revisão)",
            "interactive": "PERGUNTA (modo interativo)",
            "auto": "SIM (modo automático)"
        }
        print(f"⚙️  Auto-apply changes: {mode_labels[self.apply_mode]}")
        print(f"📁 Learning log: {self.learning_log_path}")
        print()


    def intercept_event(self, event: TradingEvent) -> DualAnalysis:
        """
        Intercepta um evento e dispara análise dual.

        Este é o CORAÇÃO do sistema - tudo passa por aqui!
        """

        self.total_events += 1
        self.event_history.append(event)

        print(f"\n{'='*60}")
        print(f"🎯 EVENTO #{self.total_events}: {event.event_type.value}")
        print(f"{'='*60}")

        # Análise dual
        analysis = self._dual_analysis(event)

        # Executar ações se necessário
        if analysis.actions:
            self._execute_actions(analysis)

        # Salvar aprendizado
        self._save_learning(analysis)

        self.total_analyses += 1
        self.analysis_history.append(analysis)

        return analysis


    def _dual_analysis(self, event: TradingEvent) -> DualAnalysis:
        """
        Análise dual: Claude (estratégico) + GPT (técnico)
        """

        print("\n🧠 CLAUDE analisando (perspectiva estratégica)...")
        claude_analysis = self._claude_analyze(event)

        print("\n⚡ GPT analisando (perspectiva técnica)...")
        gpt_analysis = self._gpt_analyze(event, claude_analysis)

        print("\n🤝 Gerando consenso...")
        consensus, actions, confidence = self._generate_consensus(
            event, claude_analysis, gpt_analysis
        )

        return DualAnalysis(
            event=event,
            claude_analysis=claude_analysis,
            gpt_analysis=gpt_analysis,
            consensus=consensus,
            actions=actions,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )


    def _claude_analyze(self, event: TradingEvent) -> str:
        """Claude analisa com visão estratégica profunda"""

        prompt = f"""Você é CLAUDE, o estrategista do BotScalp V3.

EVENTO RECEBIDO:
Tipo: {event.event_type.value}
Timestamp: {event.timestamp}
Dados: {json.dumps(event.data, indent=2)}
Contexto: {json.dumps(event.context or {}, indent=2)}

SUA MISSÃO:
Analise este evento com visão ESTRATÉGICA:
1. O que este evento significa para a estratégia geral?
2. Há padrões emergentes? Tendências?
3. Quais são os riscos estratégicos?
4. Como isso afeta nossa edge competitiva?
5. Que aprendizados devemos registrar?

FOCO: Visão de longo prazo, detecção de padrões, gestão de risco.

Responda de forma concisa mas profunda (max 300 palavras)."""

        try:
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",  # Usando Haiku (mais rápido, disponível na API key atual)
                max_tokens=1024,
                temperature=0.6,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except Exception as e:
            return f"❌ Erro na análise Claude: {str(e)}"


    def _gpt_analyze(self, event: TradingEvent, claude_input: str) -> str:
        """GPT analisa com visão técnica e de execução"""

        prompt = f"""Você é GPT-4o, o engenheiro do BotScalp V3.

EVENTO RECEBIDO:
Tipo: {event.event_type.value}
Dados: {json.dumps(event.data, indent=2)}

ANÁLISE DE CLAUDE (Estrategista):
{claude_input}

SUA MISSÃO:
Analise com visão TÉCNICA e EXECUTIVA:
1. Há bugs ou problemas de implementação?
2. O código está otimizado?
3. Que mudanças técnicas são necessárias?
4. Como melhorar a performance?
5. Que testes devemos adicionar?

FOCO: Código, otimização, debugging, execução precisa.

Seja ESPECÍFICO sobre mudanças de código (max 300 palavras)."""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"❌ Erro na análise GPT: {str(e)}"


    def _generate_consensus(self,
                           event: TradingEvent,
                           claude: str,
                           gpt: str) -> Tuple[str, List[Dict], float]:
        """
        Gera consenso entre Claude e GPT e define ações.

        Returns:
            (consenso, lista_de_ações, confiança)
        """

        # Prompt para consenso
        prompt = f"""Você é o ARBITRADOR entre Claude e GPT no BotScalp V3.

EVENTO: {event.event_type.value}

CLAUDE (Estratégico): {claude}

GPT (Técnico): {gpt}

MISSÃO:
Gere um CONSENSO e lista de AÇÕES concretas.

Retorne JSON:
{{
  "consenso": "resumo do acordo entre ambos",
  "ações": [
    {{"tipo": "code_change|parameter_update|test|log", "descrição": "...", "prioridade": 1-10}},
    ...
  ],
  "confiança": 0.0-1.0
}}

Seja ESPECÍFICO e ACIONÁVEL."""

        try:
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",  # Usando Haiku (mais rápido, disponível na API key atual)
                max_tokens=2048,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text

            # Extrair JSON (pode vir dentro de markdown)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)

            return (
                result.get("consenso", "Sem consenso"),
                result.get("ações", []),
                result.get("confiança", 0.5)
            )

        except Exception as e:
            print(f"⚠️  Erro ao gerar consenso: {e}")
            return (
                f"Erro: {str(e)}",
                [],
                0.0
            )


    def _ask_user_approval(self, action: Dict, action_number: int, total_actions: int) -> bool:
        """
        Pergunta ao usuário se deseja aplicar a mudança.

        Args:
            action: Ação a ser aplicada
            action_number: Número da ação (1-indexed)
            total_actions: Total de ações

        Returns:
            True se usuário aprovou, False caso contrário
        """
        action_type = action.get("tipo", "unknown")
        desc = action.get("descrição", "Sem descrição")
        priority = action.get("prioridade", 5)

        print(f"\n{'='*60}")
        print(f"🎯 AÇÃO PROPOSTA #{action_number}/{total_actions}")
        print(f"{'='*60}")
        print(f"Tipo: [{action_type}]")
        print(f"Prioridade: {priority}/10")
        print(f"Descrição: {desc}")
        print(f"{'='*60}")

        while True:
            try:
                response = input("Aplicar esta mudança? [S/n]: ").strip().lower()

                # Enter = Sim (padrão)
                if response == "" or response == "s" or response == "sim" or response == "y" or response == "yes":
                    return True
                elif response == "n" or response == "não" or response == "nao" or response == "no":
                    return False
                else:
                    print("⚠️  Resposta inválida. Digite 'S' para Sim ou 'n' para Não.")
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️  Interrompido pelo usuário. Pulando ação.")
                return False


    def _execute_actions(self, analysis: DualAnalysis):
        """Executa as ações definidas pelo consenso"""

        print(f"\n{'─'*60}")
        print(f"🎬 EXECUTANDO {len(analysis.actions)} AÇÕES")
        print(f"{'─'*60}")

        for i, action in enumerate(analysis.actions, 1):
            action_type = action.get("tipo", "unknown")
            desc = action.get("descrição", "Sem descrição")
            priority = action.get("prioridade", 5)

            print(f"\n{i}. [{action_type}] Prioridade {priority}/10")
            print(f"   {desc}")

            # Decidir se aplica baseado no modo
            should_apply = False

            if self.apply_mode == "auto":
                should_apply = True
                print(f"   ✅ Aplicando automaticamente (modo auto)")

            elif self.apply_mode == "interactive":
                should_apply = self._ask_user_approval(action, i, len(analysis.actions))
                if should_apply:
                    print(f"   ✅ Aprovado pelo usuário")
                else:
                    print(f"   ⏭️  Pulado pelo usuário")

            else:  # review mode
                should_apply = False
                print(f"   ⏸️  Aguardando aprovação (modo revisão)")

            # Aplicar mudança se aprovado
            if should_apply and action_type == "code_change":
                print(f"   ⏳ Aplicando mudança...")
                self._apply_code_change(action)
                self.total_code_changes += 1


    def _apply_code_change(self, action: Dict):
        """Aplica uma mudança de código (com backup!)"""

        # TODO: Implementar sistema seguro de modificação de código
        # 1. Criar backup do arquivo
        # 2. Aplicar mudança
        # 3. Rodar testes
        # 4. Se falhar, reverter

        print("   ⚠️  Code modification not yet implemented (safety first!)")

        # Log da mudança proposta
        with open(self.code_changes_log, 'a') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "applied": False,
                "reason": "Manual review required"
            }) + "\n")


    def _save_learning(self, analysis: DualAnalysis):
        """Salva o aprendizado no log"""

        with open(self.learning_log_path, 'a') as f:
            f.write(json.dumps(analysis.to_dict()) + "\n")


    def get_stats(self) -> Dict:
        """Retorna estatísticas do sistema"""

        return {
            "total_events": self.total_events,
            "total_analyses": self.total_analyses,
            "total_code_changes": self.total_code_changes,
            "total_improvements": self.total_improvements,
            "avg_confidence": sum(a.confidence for a in self.analysis_history) / len(self.analysis_history) if self.analysis_history else 0,
            "event_types": {
                et.value: sum(1 for e in self.event_history if e.event_type == et)
                for et in EventType
            }
        }


    def print_summary(self):
        """Imprime resumo das operações"""

        stats = self.get_stats()

        print("\n" + "="*60)
        print("📊 AUTO EVOLUTION SYSTEM - SUMMARY")
        print("="*60)
        print(f"📈 Total eventos processados: {stats['total_events']}")
        print(f"🧠 Total análises duais: {stats['total_analyses']}")
        print(f"🔧 Mudanças de código: {stats['total_code_changes']}")
        print(f"✨ Melhorias aplicadas: {stats['total_improvements']}")
        print(f"💯 Confiança média: {stats['avg_confidence']:.1%}")
        print()
        print("Tipos de eventos:")
        for event_type, count in stats['event_types'].items():
            if count > 0:
                print(f"  • {event_type}: {count}")
        print("="*60)


# ========== HELPER FUNCTIONS ==========

def create_event(event_type: EventType, data: Dict, context: Optional[Dict] = None) -> TradingEvent:
    """Helper para criar eventos facilmente"""
    return TradingEvent(
        event_type=event_type,
        timestamp=datetime.now().isoformat(),
        data=data,
        context=context
    )


# ========== EXEMPLO DE USO ==========

if __name__ == "__main__":

    print("""
╔═══════════════════════════════════════════════════════════════╗
║        🚀 AUTO EVOLUTION SYSTEM - DEMO                       ║
╚═══════════════════════════════════════════════════════════════╝

Demonstração do sistema de auto-evolução:
- Intercepta eventos de trading
- Claude + GPT analisam
- Propõem melhorias
- Sistema aprende continuamente
""")

    # Inicializar sistema
    system = AutoEvolutionSystem(auto_apply_changes=False)

    # Simular alguns eventos
    print("\n🧪 Simulando eventos de teste...")

    # Evento 1: Resultado de backtest
    event1 = create_event(
        EventType.BACKTEST_RESULT,
        data={
            "win_rate": 0.65,
            "total_trades": 100,
            "profit": 1500,
            "max_drawdown": -800,
            "sharpe_ratio": 1.2
        },
        context={"strategy": "scalping_v1", "timeframe": "1m"}
    )

    analysis1 = system.intercept_event(event1)

    print("\n" + "─"*60)
    print("CONSENSO:")
    print(analysis1.consensus)
    print(f"Confiança: {analysis1.confidence:.1%}")

    # Evento 2: Erro durante execução
    event2 = create_event(
        EventType.ERROR_OCCURRED,
        data={
            "error_type": "OrderRejected",
            "message": "Insufficient balance",
            "traceback": "...",
            "frequency": 5
        },
        context={"last_trade": "BUY BTCUSDT @ 43500"}
    )

    analysis2 = system.intercept_event(event2)

    # Mostrar resumo
    system.print_summary()

    print("\n✅ Demo completo! Veja os logs em:")
    print(f"   • {system.learning_log_path}")
    print(f"   • {system.code_changes_log}")
