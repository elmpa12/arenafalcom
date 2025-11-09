#!/usr/bin/env python3
"""
BACKTEST DESIGN DEBATE - Claude + GPT

As IAs debatem e formulam a melhor estratégia de backtest.
Elas trocam ideias, aprendem uma com a outra, e chegam a um consenso.

Uso:
    python3 backtest_design_debate.py
"""

import os
import json
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Importar as IAs
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()


class BacktestDesignDebate:
    """Sistema de debate entre Claude e GPT para formular backtest."""

    def __init__(self):
        """Inicializa as IAs."""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        self.debate_history = []
        self.consensus = None

    def _ask_claude(self, prompt: str, context: str = "") -> str:
        """Pergunta para Claude (perspectiva estratégica)."""
        full_prompt = f"""Você é CLAUDE, especialista em estratégia de trading e análise quantitativa.

CONTEXTO DO PROJETO:
- BotScalp V3 - Trading bot com auto-evolution
- Dados: 2 anos de BTCUSDT (aggTrades + klines 1m/5m/15m)
- Modelos ML: XGBoost, RandomForest, LogReg, Ensemble
- Objetivo: Formular backtest robusto e exigente

{context}

{prompt}

Responda de forma estratégica, pensando em:
1. Robustez estatística
2. Métricas de qualidade
3. Validação rigorosa
4. Evitar overfitting
"""

        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2000,
            messages=[{"role": "user", "content": full_prompt}]
        )

        return response.content[0].text

    def _ask_gpt(self, prompt: str, context: str = "") -> str:
        """Pergunta para GPT (perspectiva técnica)."""
        full_prompt = f"""Você é GPT, especialista em implementação técnica e machine learning.

CONTEXTO DO PROJETO:
- BotScalp V3 - Trading bot com auto-evolution
- Dados: 2 anos de BTCUSDT (aggTrades + klines 1m/5m/15m)
- Modelos ML: XGBoost, RandomForest, LogReg, Ensemble
- Objetivo: Formular backtest robusto e exigente

{context}

{prompt}

Responda de forma técnica, pensando em:
1. Implementação prática
2. Eficiência computacional
3. Features engineering
4. Validação cruzada
"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=2000,
            temperature=0.7
        )

        return response.choices[0].message.content

    def debate_round(self, topic: str, round_num: int, previous_responses: List[str] = None) -> Dict:
        """Uma rodada de debate sobre um tópico."""
        print(f"\n{'='*70}")
        print(f"🎯 RODADA {round_num}: {topic}")
        print(f"{'='*70}\n")

        # Contexto das respostas anteriores
        context = ""
        if previous_responses:
            context = "DISCUSSÃO ANTERIOR:\n" + "\n".join(previous_responses)

        # Claude responde primeiro (estratégico)
        print("🤖 CLAUDE (Estratégico):")
        print("-" * 70)
        claude_response = self._ask_claude(topic, context)
        print(claude_response)
        print()

        # GPT responde considerando Claude (técnico)
        print("🤖 GPT (Técnico):")
        print("-" * 70)
        gpt_context = context + f"\n\nCLAUDE disse:\n{claude_response}"
        gpt_response = self._ask_gpt(topic, gpt_context)
        print(gpt_response)
        print()

        # Claude rebate considerando GPT
        print("🤖 CLAUDE (Rebatendo):")
        print("-" * 70)
        claude_rebuttal_context = gpt_context + f"\n\nGPT disse:\n{gpt_response}"
        claude_rebuttal = self._ask_claude(
            "Considerando a resposta do GPT, o que você complementa ou ajusta na sua proposta?",
            claude_rebuttal_context
        )
        print(claude_rebuttal)
        print()

        # Salvar rodada
        round_data = {
            "round": round_num,
            "topic": topic,
            "claude_initial": claude_response,
            "gpt_response": gpt_response,
            "claude_rebuttal": claude_rebuttal,
            "timestamp": datetime.now().isoformat()
        }

        self.debate_history.append(round_data)

        return round_data

    def generate_consensus(self) -> Dict:
        """Gera consenso final entre as IAs."""
        print(f"\n{'='*70}")
        print("🎯 GERANDO CONSENSO FINAL")
        print(f"{'='*70}\n")

        # Resumir todo o debate
        debate_summary = "\n\n".join([
            f"RODADA {r['round']}: {r['topic']}\n"
            f"Claude: {r['claude_initial'][:200]}...\n"
            f"GPT: {r['gpt_response'][:200]}...\n"
            f"Claude (rebate): {r['claude_rebuttal'][:200]}..."
            for r in self.debate_history
        ])

        # Claude gera consenso estratégico
        print("🤖 CLAUDE - Consenso Estratégico:")
        print("-" * 70)
        claude_consensus = self._ask_claude(
            "Com base em TODA a discussão, gere um PLANO FINAL consolidado de backtest. "
            "Liste claramente: (1) Métricas principais, (2) Período de teste, (3) Validação, "
            "(4) Features, (5) Critérios de sucesso.",
            debate_summary
        )
        print(claude_consensus)
        print()

        # GPT gera consenso técnico
        print("🤖 GPT - Consenso Técnico:")
        print("-" * 70)
        gpt_consensus = self._ask_gpt(
            "Com base em TODA a discussão, gere um PLANO TÉCNICO consolidado de implementação. "
            "Liste claramente: (1) Estrutura de código, (2) Pipeline de dados, (3) Features engineering, "
            "(4) Validação cruzada, (5) Métricas computacionais.",
            debate_summary + f"\n\nCLAUDE CONSENSO:\n{claude_consensus}"
        )
        print(gpt_consensus)
        print()

        # Consenso final combinado
        consensus = {
            "strategic_plan": claude_consensus,
            "technical_plan": gpt_consensus,
            "debate_rounds": len(self.debate_history),
            "timestamp": datetime.now().isoformat()
        }

        self.consensus = consensus
        return consensus

    def save_results(self, filename: str = "backtest_design_consensus.json"):
        """Salva resultados do debate."""
        output = {
            "debate_history": self.debate_history,
            "consensus": self.consensus,
            "metadata": {
                "total_rounds": len(self.debate_history),
                "timestamp": datetime.now().isoformat(),
                "models": {
                    "claude": "claude-3-haiku-20240307",
                    "gpt": "gpt-4o"
                }
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Resultados salvos em: {filename}")


def main():
    """Executa debate completo sobre design de backtest."""
    print("\n" + "🚀" * 35)
    print("BACKTEST DESIGN DEBATE - Claude + GPT")
    print("🚀" * 35)
    print()
    print("As IAs vão debater e formular a melhor estratégia de backtest.")
    print("DEBATE EXTENSO: 20+ trocas de ideias!")
    print()

    debate = BacktestDesignDebate()

    # ========== BLOCO 1: MÉTRICAS (5 rodadas) ==========

    # Rodada 1: Métricas principais
    debate.debate_round(
        topic="Quais as MÉTRICAS mais importantes para avaliar um backtest de trading? "
               "Considere: win rate, Sharpe ratio, max drawdown, profit factor, etc. "
               "Quais devemos priorizar e por quê?",
        round_num=1
    )

    # Rodada 2: Métricas de risco
    debate.debate_round(
        topic="Vamos focar em MÉTRICAS DE RISCO. Como avaliar: drawdown, volatilidade, "
               "tail risk, value at risk (VaR)? Quais são críticas para evitar blowup?",
        round_num=2,
        previous_responses=[f"R1: {debate.debate_history[0]['claude_initial'][:100]}..."]
    )

    # Rodada 3: Métricas de consistência
    debate.debate_round(
        topic="E quanto à CONSISTÊNCIA? Como medir: estabilidade de returns, "
               "win/loss streaks, monthly consistency, recovery time? "
               "O que indica um sistema robusto vs sortudo?",
        round_num=3,
        previous_responses=[f"R2: {debate.debate_history[1]['gpt_response'][:100]}..."]
    )

    # Rodada 4: Métricas de eficiência
    debate.debate_round(
        topic="EFICIÊNCIA DE CAPITAL: Como avaliar profit factor, return/risk ratio, "
               "Sortino ratio, Calmar ratio? Qual threshold mínimo para cada?",
        round_num=4,
        previous_responses=[f"R3: {debate.debate_history[2]['claude_rebuttal'][:100]}..."]
    )

    # Rodada 5: Trade-offs entre métricas
    debate.debate_round(
        topic="TRADE-OFFS: Se precisar escolher entre 90% win rate com baixo Sharpe "
               "vs 60% win rate com alto Sharpe, o que escolher? "
               "Como balancear métricas conflitantes?",
        round_num=5,
        previous_responses=[f"R4: {debate.debate_history[3]['gpt_response'][:100]}..."]
    )

    # ========== BLOCO 2: VALIDAÇÃO (5 rodadas) ==========

    # Rodada 6: Estrutura de validação
    debate.debate_round(
        topic="ESTRUTURA DE VALIDAÇÃO: Walk-forward? Expanding window? "
               "Rolling window? Purged k-fold? Qual a melhor abordagem e por quê?",
        round_num=6,
        previous_responses=[f"R5: {debate.debate_history[4]['claude_initial'][:100]}..."]
    )

    # Rodada 7: Tamanho de janelas
    debate.debate_round(
        topic="TAMANHO DE JANELAS: Quanto de dados para treino vs validação? "
               "6 meses treino / 1 mês validação? 1 ano / 3 meses? "
               "Como balancear dados suficientes vs regime changes?",
        round_num=7,
        previous_responses=[f"R6: {debate.debate_history[5]['gpt_response'][:100]}..."]
    )

    # Rodada 8: Evitar overfitting
    debate.debate_round(
        topic="COMBATER OVERFITTING: Regularização? Early stopping? "
               "Feature selection? Ensemble methods? Cross-validation? "
               "Quais técnicas são ESSENCIAIS?",
        round_num=8,
        previous_responses=[f"R7: {debate.debate_history[6]['claude_rebuttal'][:100]}..."]
    )

    # Rodada 9: Detecção de regime changes
    debate.debate_round(
        topic="REGIME CHANGES: Como detectar quando o mercado mudou? "
               "Como invalidar modelos antigos? Quando retreinar? "
               "Como adaptar dinamicamente?",
        round_num=9,
        previous_responses=[f"R8: {debate.debate_history[7]['gpt_response'][:100]}..."]
    )

    # Rodada 10: Out-of-sample testing
    debate.debate_round(
        topic="TESTE OUT-OF-SAMPLE: Guardar últimos 3-6 meses sem tocar? "
               "Usar dados de outro par (BTC/ETH)? "
               "Como garantir teste verdadeiramente independente?",
        round_num=10,
        previous_responses=[f"R9: {debate.debate_history[8]['claude_initial'][:100]}..."]
    )

    # ========== BLOCO 3: FEATURES (5 rodadas) ==========

    # Rodada 11: Features de price action
    debate.debate_round(
        topic="PRICE ACTION FEATURES: Returns, volatility, momentum, mean reversion? "
               "Quais timeframes? Como combinar aggTrades + klines? "
               "Evitar look-ahead bias?",
        round_num=11,
        previous_responses=[f"R10: {debate.debate_history[9]['gpt_response'][:100]}..."]
    )

    # Rodada 12: Features de volume
    debate.debate_round(
        topic="VOLUME FEATURES: Volume, VWAP, volume profile, buy/sell imbalance? "
               "Volume é informativo em crypto? Como usar aggTrades de forma eficaz?",
        round_num=12,
        previous_responses=[f"R11: {debate.debate_history[10]['claude_rebuttal'][:100]}..."]
    )

    # Rodada 13: Indicadores técnicos
    debate.debate_round(
        topic="INDICADORES TÉCNICOS: RSI, MACD, Bollinger Bands, ATR? "
               "São úteis ou apenas noise? Como evitar overfitting em indicadores? "
               "Quais são robustos?",
        round_num=13,
        previous_responses=[f"R12: {debate.debate_history[11]['gpt_response'][:100]}..."]
    )

    # Rodada 14: Features temporais
    debate.debate_round(
        topic="FEATURES TEMPORAIS: Hour of day, day of week, month? "
               "Sazonalidade importa em crypto? Como capturar padrões temporais "
               "sem overfitting?",
        round_num=14,
        previous_responses=[f"R13: {debate.debate_history[12]['claude_initial'][:100]}..."]
    )

    # Rodada 15: Microstructure
    debate.debate_round(
        topic="MARKET MICROSTRUCTURE: Bid-ask spread, order flow, trade size distribution? "
               "Podemos extrair de aggTrades? Vale a pena ou é muito granular?",
        round_num=15,
        previous_responses=[f"R14: {debate.debate_history[13]['gpt_response'][:100]}..."]
    )

    # ========== BLOCO 4: IMPLEMENTAÇÃO (5 rodadas) ==========

    # Rodada 16: Arquitetura de código
    debate.debate_round(
        topic="ARQUITETURA DE CÓDIGO: Como estruturar pipeline de backtest? "
               "Separar data loading, feature engineering, model training, evaluation? "
               "Modular vs monolítico?",
        round_num=16,
        previous_responses=[f"R15: {debate.debate_history[14]['claude_rebuttal'][:100]}..."]
    )

    # Rodada 17: Otimização de performance
    debate.debate_round(
        topic="PERFORMANCE COMPUTACIONAL: Vectorização vs loops? "
               "Paralelização? Usar GPUs? Como processar 900M+ trades rapidamente?",
        round_num=17,
        previous_responses=[f"R16: {debate.debate_history[15]['gpt_response'][:100]}..."]
    )

    # Rodada 18: Logging e debugging
    debate.debate_round(
        topic="LOGGING E DEBUGGING: Como rastrear cada trade? "
               "Salvar features usadas? Checkpoint de modelos? "
               "Como debugar quando métricas são ruins?",
        round_num=18,
        previous_responses=[f"R17: {debate.debate_history[16]['claude_initial'][:100]}..."]
    )

    # Rodada 19: Integração com auto-evolution
    debate.debate_round(
        topic="INTEGRAÇÃO AUTO-EVOLUTION: Como as IAs vão analisar cada backtest? "
               "Quais eventos disparar? O que logar para aprendizado? "
               "Como modificar código automaticamente?",
        round_num=19,
        previous_responses=[f"R18: {debate.debate_history[17]['gpt_response'][:100]}..."]
    )

    # Rodada 20: Critérios de graduação
    debate.debate_round(
        topic="CRITÉRIOS FINAIS PARA GRADUAÇÃO: Quando sair de backtest para paper trading? "
               "Quais métricas mínimas? Quantos períodos de validação? "
               "Como garantir que está REALMENTE pronto?",
        round_num=20,
        previous_responses=[f"R19: {debate.debate_history[18]['claude_rebuttal'][:100]}..."]
    )

    # Gerar consenso final
    consensus = debate.generate_consensus()

    # Salvar resultados
    debate.save_results("backtest_design_consensus.json")

    # Resumo final
    print("\n" + "=" * 70)
    print("📊 RESUMO DO DEBATE")
    print("=" * 70)
    print(f"\n✅ {len(debate.debate_history)} rodadas de debate completadas")
    print(f"✅ Consenso estratégico gerado por Claude")
    print(f"✅ Consenso técnico gerado por GPT")
    print(f"✅ Plano completo salvo em backtest_design_consensus.json")
    print()
    print("🎯 PRÓXIMOS PASSOS:")
    print("1. Revisar backtest_design_consensus.json")
    print("2. Implementar o plano consensuado")
    print("3. Rodar backtests com os critérios definidos")
    print("4. Usar auto-evolution para melhorar continuamente")
    print()


if __name__ == "__main__":
    main()
