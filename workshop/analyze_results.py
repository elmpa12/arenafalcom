#!/usr/bin/env python3
"""
ANÁLISE DETALHADA DOS RESULTADOS - BotScalp V3

As IAs (Claude + GPT) analisam os resultados do selector21 e DISCUTEM:
- O que funcionou bem
- O que não funcionou
- Métricas críticas
- Propostas de melhoria
- Próximos experimentos

Uso:
    python3 analyze_results.py
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()


def read_results():
    """Lê todos os resultados do selector21."""
    results = {}

    # CSVs principais
    csv_files = [
        "wf_leaderboard_all.csv",
        "wf_leaderboard_base.csv",
        "wf_leaderboard_combos.csv",
        "wf_best_trades.csv",
        "wf_report.csv",
        "wf_ml.csv"
    ]

    for csv in csv_files:
        path = f"./resultados/{csv}"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                results[csv] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "head": df.head(10).to_dict('records') if len(df) > 0 else [],
                    "summary": df.describe().to_dict() if len(df) > 0 else {}
                }
                print(f"✓ Lido: {csv} ({len(df)} linhas)")
            except Exception as e:
                print(f"✗ Erro lendo {csv}: {e}")

    # JSONs
    json_files = ["selection_report.json", "runtime_config.json"]
    for jf in json_files:
        path = f"./resultados/{jf}"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    results[jf] = json.load(f)
                print(f"✓ Lido: {jf}")
            except Exception as e:
                print(f"✗ Erro lendo {jf}: {e}")

    return results


def analyze_with_ais(results: dict, num_rounds: int = 5):
    """Claude e GPT analisam e DISCUTEM os resultados."""

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Prepara sumário dos resultados
    summary = {
        "files_read": list(results.keys()),
        "total_strategies": 0,
        "top_performers": [],
        "key_metrics": {}
    }

    if "wf_leaderboard_all.csv" in results:
        lb = results["wf_leaderboard_all.csv"]
        summary["total_strategies"] = lb["rows"]
        if lb["head"]:
            summary["top_performers"] = lb["head"][:5]

    results_str = json.dumps(summary, indent=2, default=str)[:8000]  # Limita tamanho

    print("\n" + "="*80)
    print("🤖 ANÁLISE E DISCUSSÃO PELAS IAs")
    print("="*80)
    print(f"\nRodadas de debate: {num_rounds}")
    print(f"Resultados carregados: {len(results)} arquivos")
    print()

    conversation = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"📊 RODADA {round_num}/{num_rounds}")
        print("="*80)

        if round_num == 1:
            # Primeira rodada: análise inicial
            claude_prompt = f"""Você é um especialista em backtesting de trading systems.

TAREFA: Analisar os RESULTADOS REAIS do selector21 que acabou de rodar.

RESULTADOS:
{results_str}

ANÁLISE INICIAL (Rodada 1/{num_rounds}):
1. **Visão Geral**: O que os números estão dizendo?
2. **Top Performers**: Quais estratégias se destacaram?
3. **Métricas Críticas**: Sharpe, Win Rate, Drawdown, PnL
4. **Sinais de Overfitting**: Há evidências?
5. **Primeiras Observações**: O que chama atenção?

Seja OBJETIVO e CRÍTICO. Foque nos DADOS."""

            gpt_prompt = claude_prompt.replace("ANÁLISE INICIAL", "ANÁLISE TÉCNICA")

        else:
            # Rodadas seguintes: debate e aprofundamento
            prev_context = "\n\n".join([
                f"[{entry['round']}] {entry['speaker']}: {entry['response'][:500]}..."
                for entry in conversation[-4:]  # Últimas 2 trocas
            ])

            topics = [
                "Robustez das estratégias top",
                "Combos vs Base: qual performou melhor?",
                "Machine Learning: agregou valor?",
                "Walk-Forward: consistência entre janelas?",
                "Próximos experimentos sugeridos"
            ]
            topic = topics[(round_num - 2) % len(topics)]

            claude_prompt = f"""Continuando a análise dos resultados do selector21.

CONTEXTO ANTERIOR:
{prev_context}

RODADA {round_num}/{num_rounds} - FOCO: {topic}

Considerando o que foi discutido até agora, aprofunde em:
- {topic}
- Contra-argumentos às análises anteriores
- Evidências nos dados que apoiam ou contradizem hipóteses
- Sugestões práticas de melhorias

Seja ESPECÍFICO com os dados."""

            gpt_prompt = claude_prompt

        # Claude analisa
        print("🔵 Claude analisando...")
        try:
            claude_response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": claude_prompt}]
            )
            claude_text = claude_response.content[0].text
            conversation.append({
                "round": round_num,
                "speaker": "Claude",
                "response": claude_text
            })
            print(f"✓ Claude ({len(claude_text)} chars)")
        except Exception as e:
            print(f"✗ Claude falhou: {e}")
            claude_text = f"[ERRO: {e}]"

        # GPT responde
        print("🟢 GPT analisando...")
        try:
            gpt_context = gpt_prompt + f"\n\nCLAUDE DISSE:\n{claude_text}\n\nAgora dê sua análise considerando o que Claude disse."
            gpt_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": gpt_context}],
                max_tokens=2000
            )
            gpt_text = gpt_response.choices[0].message.content
            conversation.append({
                "round": round_num,
                "speaker": "GPT",
                "response": gpt_text
            })
            print(f"✓ GPT ({len(gpt_text)} chars)")
        except Exception as e:
            print(f"✗ GPT falhou: {e}")
            gpt_text = f"[ERRO: {e}]"

        # Claude rebate (exceto última rodada)
        if round_num < num_rounds:
            print("🔵 Claude rebatendo...")
            try:
                rebuttal_prompt = f"""GPT respondeu:

{gpt_text}

Responda brevemente (max 500 chars):
- Concorda ou discorda?
- Evidências que GPT pode ter perdido?
- Refinamento da análise"""

                claude_reb = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=800,
                    messages=[{"role": "user", "content": rebuttal_prompt}]
                )
                claude_reb_text = claude_reb.content[0].text
                conversation.append({
                    "round": round_num,
                    "speaker": "Claude (rebate)",
                    "response": claude_reb_text
                })
                print(f"✓ Claude rebateu ({len(claude_reb_text)} chars)")
            except Exception as e:
                print(f"✗ Claude rebate falhou: {e}")

    # Consenso final
    print(f"\n{'='*80}")
    print("🎯 CONSENSO FINAL")
    print("="*80)
    print("\n🟢 GPT gerando consenso...")

    full_debate = "\n\n".join([
        f"[{e['round']}] {e['speaker']}:\n{e['response']}"
        for e in conversation
    ])

    try:
        consensus_prompt = f"""Com base em todo o debate sobre os resultados do selector21:

{full_debate[:6000]}

Gere um CONSENSO FINAL com:

1. **Principais Descobertas** (3-5 pontos)
2. **Estratégias Recomendadas** (top 3 e porquê)
3. **Melhorias Prioritárias** (3-5 itens)
4. **Próximos Experimentos** (3-5 ideias concretas)
5. **Parâmetros a Ajustar** (específico)

Formato: Markdown estruturado, objetivo, acionável."""

        consensus = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": consensus_prompt}],
            max_tokens=2500
        )
        consensus_text = consensus.choices[0].message.content
        print("✓ Consenso gerado")
    except Exception as e:
        print(f"✗ Consenso falhou: {e}")
        consensus_text = f"[ERRO: {e}]"

    # Salva tudo
    output = {
        "results_summary": summary,
        "conversation": conversation,
        "consensus": consensus_text,
        "metadata": {
            "num_rounds": num_rounds,
            "total_exchanges": len(conversation)
        }
    }

    with open("resultados/ai_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    with open("resultados/ai_consensus.md", "w") as f:
        f.write(f"# Consenso das IAs - Análise Selector21\n\n")
        f.write(consensus_text)

    print(f"\n{'='*80}")
    print("💾 ANÁLISE SALVA")
    print("="*80)
    print("\n  • resultados/ai_analysis.json (debate completo)")
    print("  • resultados/ai_consensus.md (consenso final)")
    print()

    return output


if __name__ == "__main__":
    print("="*80)
    print("🤖 ANÁLISE DETALHADA DOS RESULTADOS")
    print("="*80)
    print()

    # Lê resultados
    print("📂 Lendo resultados do selector21...")
    results = read_results()

    if not results:
        print("\n⚠️  Nenhum resultado encontrado em ./resultados/")
        print("   Execute o selector21 primeiro!")
        exit(1)

    print(f"\n✓ {len(results)} arquivos carregados")

    # Análise pelas IAs
    analysis = analyze_with_ais(results, num_rounds=5)

    print("\n" + "="*80)
    print("✅ ANÁLISE COMPLETA!")
    print("="*80)
    print()
    print("📊 Resumo:")
    print(f"  • Rodadas de debate: {analysis['metadata']['num_rounds']}")
    print(f"  • Total de trocas: {analysis['metadata']['total_exchanges']}")
    print()
    print("📖 Próximo passo:")
    print("   cat resultados/ai_consensus.md")
    print()
