#!/usr/bin/env python3
"""
EVOLUÇÃO EXPONENCIAL DE ESTRATÉGIAS - BotScalp V3

As IAs analisam resultados, aprendem padrões e GERAM automaticamente
a próxima geração de testes MELHORADOS.

Uso:
    python3 evolve_strategy.py --generation 1
"""

import os
import json
import glob
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()


def load_all_results(test_dirs):
    """Carrega TODOS os resultados de todos os testes."""
    all_results = {}

    for test_dir in test_dirs:
        test_name = os.path.basename(test_dir)

        # Ler CSVs
        csvs = {}
        for csv_file in glob.glob(f"{test_dir}/*.csv"):
            csv_name = os.path.basename(csv_file)
            try:
                df = pd.read_csv(csv_file)
                csvs[csv_name] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "data": df.to_dict('records') if len(df) > 0 else []
                }
            except Exception as e:
                csvs[csv_name] = {"error": str(e)}

        # Ler JSONs
        jsons = {}
        for json_file in glob.glob(f"{test_dir}/*.json"):
            json_name = os.path.basename(json_file)
            try:
                with open(json_file) as f:
                    jsons[json_name] = json.load(f)
            except Exception as e:
                jsons[json_name] = {"error": str(e)}

        all_results[test_name] = {
            "csvs": csvs,
            "jsons": jsons
        }

    return all_results


def analyze_and_evolve(results, generation):
    """Claude + GPT analisam e propõem próxima geração."""

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Preparar sumário dos resultados
    summary = {}
    for test_name, data in results.items():
        lb = data["csvs"].get("leaderboard_base.csv", {})
        if "data" in lb and lb["data"]:
            best = lb["data"][0] if len(lb["data"]) > 0 else {}
            summary[test_name] = {
                "num_strategies": lb.get("rows", 0),
                "best_method": best.get("method", "N/A"),
                "best_sharpe": best.get("sharpe", "N/A"),
                "best_hit": best.get("hit", "N/A"),
                "best_pnl": best.get("total_pnl", "N/A"),
                "best_trades": best.get("n_trades", "N/A")
            }

    results_str = json.dumps(summary, indent=2, default=str)[:6000]

    print("\n" + "="*80)
    print(f"🧬 GERAÇÃO {generation} - ANÁLISE E EVOLUÇÃO")
    print("="*80)
    print(f"\nTestes analisados: {len(results)}")
    print()

    # FASE 1: Análise de Padrões (Claude)
    print("🔵 Claude identificando padrões...")

    claude_prompt = f"""Você é um especialista em trading quantitativo.

GERAÇÃO {generation} - ANÁLISE DE RESULTADOS

RESULTADOS DOS TESTES:
{results_str}

TAREFA: Identifique PADRÕES nos resultados.

1. **O QUE FUNCIONOU?**
   - Quais métodos tiveram melhor performance?
   - Quais timeframes?
   - Quais períodos?
   - Por quê você acha que funcionaram?

2. **O QUE NÃO FUNCIONOU?**
   - Quais métodos perderam dinheiro?
   - Por quê falharam?
   - Sharpe negativo indica quê?

3. **HIPÓTESES**:
   - Overfitting?
   - Períodos inadequados?
   - Métodos inadequados para o mercado?
   - Falta de stops adequados?

4. **APRENDIZADOS CHAVE** (3-5 pontos):
   - O que aprendemos desta geração?

Seja ESPECÍFICO e BASE-SE NOS DADOS."""

    try:
        claude_resp = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            messages=[{"role": "user", "content": claude_prompt}]
        )
        claude_analysis = claude_resp.content[0].text
        print(f"✓ Claude completou ({len(claude_analysis)} chars)")
    except Exception as e:
        print(f"✗ Claude falhou: {e}")
        # Fallback para modelo antigo
        try:
            claude_resp = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=3000,
                messages=[{"role": "user", "content": claude_prompt}]
            )
            claude_analysis = claude_resp.content[0].text
            print(f"✓ Claude (haiku) completou ({len(claude_analysis)} chars)")
        except Exception as e2:
            claude_analysis = f"[ERRO: {e2}]"

    # FASE 2: Propostas de Melhoria (GPT)
    print("\n🟢 GPT propondo melhorias...")

    gpt_prompt = f"""GERAÇÃO {generation} - PROPOSTAS DE MELHORIA

ANÁLISE DO CLAUDE:
{claude_analysis[:2000]}

RESULTADOS:
{results_str[:2000]}

TAREFA: Propor MELHORIAS CONCRETAS para Geração {generation+1}.

1. **AJUSTES DE PARÂMETROS**:
   - Quais parâmetros devemos testar?
   - Valores específicos?
   - Por quê esses valores?

2. **NOVOS MÉTODOS/COMBOS**:
   - Quais métodos testar?
   - Combos promissores?

3. **PERÍODOS ALTERNATIVOS**:
   - Testar outros períodos?
   - Por quê esses períodos?

4. **FILTROS/STOPS**:
   - Como melhorar risk management?
   - Stops dinâmicos?

5. **TOP 5 EXPERIMENTOS** para Geração {generation+1}:
   - Liste 5 testes concretos
   - Para cada um: objetivo, config, hipótese

Seja ESPECÍFICO e ACIONÁVEL."""

    try:
        gpt_resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": gpt_prompt}],
            max_tokens=3000
        )
        gpt_proposals = gpt_resp.choices[0].message.content
        print(f"✓ GPT completou ({len(gpt_proposals)} chars)")
    except Exception as e:
        print(f"✗ GPT falhou: {e}")
        gpt_proposals = f"[ERRO: {e}]"

    # FASE 3: Consenso e Geração de Testes
    print("\n🎯 Gerando próxima geração de testes...")

    consensus_prompt = f"""GERAÇÃO {generation+1} - CRIAR NOVOS TESTES

ANÁLISE CLAUDE:
{claude_analysis[:1500]}

PROPOSTAS GPT:
{gpt_proposals[:1500]}

TAREFA: Gerar configuração Python para Geração {generation+1}.

Crie 5-10 NOVOS testes baseados nos aprendizados.

Formato:
```python
TESTS_GEN{generation+1} = [
    {{
        "name": "gen{generation+1}_test1_nome",
        "desc": "Descrição clara do objetivo",
        "hypothesis": "Por que achamos que vai funcionar",
        "args": [
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "YYYY-MM-DD",
            "--end", "YYYY-MM-DD",
            # ... TODOS os argumentos necessários
            "--out_root", "./resultados/gen{generation+1}/test1"
        ]
    }},
    # ... mais testes
]
```

IMPORTANTE:
- Use argumentos REAIS do selector21.py
- Varie períodos, métodos, timeframes baseado no aprendizado
- Teste hipóteses específicas
- Cada teste deve ter OBJETIVO claro"""

    try:
        final_resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": consensus_prompt}],
            max_tokens=4000
        )
        next_gen_code = final_resp.choices[0].message.content
        print(f"✓ Geração {generation+1} criada ({len(next_gen_code)} chars)")
    except Exception as e:
        print(f"✗ Geração falhou: {e}")
        next_gen_code = f"[ERRO: {e}]"

    # Salvar tudo
    output = {
        "generation": generation,
        "num_tests_analyzed": len(results),
        "claude_analysis": claude_analysis,
        "gpt_proposals": gpt_proposals,
        "next_generation_code": next_gen_code,
        "summary": summary
    }

    gen_dir = f"./evolution/gen{generation}"
    os.makedirs(gen_dir, exist_ok=True)

    with open(f"{gen_dir}/analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    with open(f"{gen_dir}/next_generation.py", "w") as f:
        f.write(f"# GERAÇÃO {generation+1} - Auto-gerada pelas IAs\n\n")
        f.write(next_gen_code)

    with open(f"{gen_dir}/LEARNING.md", "w") as f:
        f.write(f"# APRENDIZADOS - GERAÇÃO {generation}\n\n")
        f.write("## Análise Claude\n\n")
        f.write(claude_analysis)
        f.write("\n\n## Propostas GPT\n\n")
        f.write(gpt_proposals)
        f.write("\n\n## Próxima Geração\n\n")
        f.write(next_gen_code)

    print(f"\n{'='*80}")
    print("💾 EVOLUÇÃO SALVA")
    print("="*80)
    print(f"\n  • evolution/gen{generation}/analysis.json")
    print(f"  • evolution/gen{generation}/next_generation.py")
    print(f"  • evolution/gen{generation}/LEARNING.md")
    print()

    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, default=1, help="Número da geração")
    parser.add_argument("--test_dir", type=str, default="./resultados/test*",
                        help="Pattern dos diretórios de teste")
    args = parser.parse_args()

    print("="*80)
    print("🧬 EVOLUÇÃO EXPONENCIAL DE ESTRATÉGIAS")
    print("="*80)
    print(f"\nGeração atual: {args.generation}")
    print()

    # Carregar resultados
    test_dirs = glob.glob(args.test_dir)
    if not test_dirs:
        print(f"⚠️  Nenhum diretório encontrado: {args.test_dir}")
        return

    print(f"📂 Carregando resultados de {len(test_dirs)} testes...")
    results = load_all_results(test_dirs)
    print(f"✓ {len(results)} testes carregados")

    # Analisar e evoluir
    evolution = analyze_and_evolve(results, args.generation)

    print("\n" + "="*80)
    print("✅ EVOLUÇÃO COMPLETA!")
    print("="*80)
    print(f"\n📊 Próximo passo:")
    print(f"   1. Revisar: cat evolution/gen{args.generation}/LEARNING.md")
    print(f"   2. Executar: python3 evolution/gen{args.generation}/next_generation.py")
    print()


if __name__ == "__main__":
    main()
