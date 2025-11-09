#!/usr/bin/env python3
"""
ESTUDO PROFUNDO DO SELECTOR21.PY

As IAs vão estudar TODAS as funções, fluxo de execução,
e condições de output LINHA POR LINHA.

Uso:
    python3 estudo_profundo_selector.py
"""

import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()


def extract_functions(code: str):
    """Extrai todas as funções definidas no código."""
    pattern = r'^def\s+(\w+)\s*\((.*?)\):'
    functions = []
    for match in re.finditer(pattern, code, re.MULTILINE):
        func_name = match.group(1)
        func_args = match.group(2)
        line_num = code[:match.start()].count('\n') + 1
        functions.append({
            "name": func_name,
            "args": func_args,
            "line": line_num
        })
    return functions


def study_selector_deep():
    """As IAs estudam o selector21.py COMPLETO."""

    print("="*80)
    print("🧠 ESTUDO PROFUNDO DO SELECTOR21.PY")
    print("="*80)
    print()

    # Ler código completo
    with open("selector21.py") as f:
        code = f.read()

    # Extrair funções
    functions = extract_functions(code)

    print(f"📊 Código: {len(code):,} caracteres")
    print(f"📊 Linhas: {len(code.splitlines()):,}")
    print(f"📊 Funções: {len(functions)}")
    print()

    # Top 30 funções por linha
    print("🔍 Top 30 funções principais:")
    for i, func in enumerate(functions[:30], 1):
        print(f"  {i:2}. {func['name']:30} (linha {func['line']:4}) - args: {func['args'][:50]}")
    print()

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # FASE 1: Mapeamento de todas as funções
    print("="*80)
    print("FASE 1: MAPEAMENTO COMPLETO DE FUNÇÕES")
    print("="*80)
    print()

    functions_str = "\n".join([
        f"{i}. {f['name']}({f['args']}) [linha {f['line']}]"
        for i, f in enumerate(functions[:50], 1)
    ])

    mapping_prompt = f"""Você é um especialista em Python e sistemas de trading.

TAREFA: Analisar COMPLETAMENTE as funções do selector21.py

FUNÇÕES IDENTIFICADAS (top 50):
{functions_str}

Para cada categoria, identifique quais funções são responsáveis:

1. **DATA LOADING** - Funções que carregam dados
2. **FEATURE ENGINEERING** - Funções que calculam features/indicadores
3. **STRATEGY EXECUTION** - Funções que executam estratégias
4. **COMBOS** - Funções que geram combinações
5. **WALK-FORWARD** - Funções relacionadas a WF optimization
6. **MACHINE LEARNING** - Funções de ML
7. **OUTPUT/SAVING** - Funções que salvam resultados (CRÍTICO!)
8. **METRICS** - Funções que calculam métricas
9. **FILTERING** - Funções que filtram estratégias
10. **MAIN FLOW** - Função main() e fluxo principal

Seja ESPECÍFICO. Liste funções por categoria e explique o que cada uma faz."""

    print("🔵 Claude mapeando funções...")
    try:
        claude_map = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": mapping_prompt}]
        )
        claude_mapping = claude_map.content[0].text
        print(f"✓ Claude completou ({len(claude_mapping)} chars)")
    except Exception as e:
        print(f"✗ Claude falhou: {e}")
        claude_mapping = f"[ERRO: {e}]"

    print()
    print("🟢 GPT mapeando funções...")
    try:
        gpt_map = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": mapping_prompt + f"\n\nCLAUDE MAPEOU:\n{claude_mapping}\n\nAgora faça seu mapeamento complementando o de Claude."
            }],
            max_tokens=4000
        )
        gpt_mapping = gpt_map.choices[0].message.content
        print(f"✓ GPT completou ({len(gpt_mapping)} chars)")
    except Exception as e:
        print(f"✗ GPT falhou: {e}")
        gpt_mapping = f"[ERRO: {e}]"

    # FASE 2: Estudo das funções de OUTPUT (críticas!)
    print()
    print("="*80)
    print("FASE 2: FUNÇÕES DE OUTPUT (CRÍTICAS)")
    print("="*80)
    print()

    # Buscar funções de output no código
    output_funcs = [f for f in functions if any(x in f['name'].lower()
                    for x in ['save', 'write', 'output', 'print', 'export', 'dump'])]

    output_str = "\n".join([
        f"- {f['name']}({f['args']}) [linha {f['line']}]"
        for f in output_funcs
    ])

    output_prompt = f"""CRÍTICO: Analise as funções de OUTPUT do selector21.py

FUNÇÕES DE OUTPUT IDENTIFICADAS:
{output_str}

PERGUNTAS:
1. Quais dessas funções são chamadas no main()?
2. Sob quais CONDIÇÕES elas são chamadas?
3. Quais ARGUMENTOS de CLI controlam essas funções?
4. O que impede os CSVs de serem gerados?
5. Como garantir que TODOS os outputs sejam gerados?

Seja EXTREMAMENTE ESPECÍFICO. Liste as condições exatas."""

    print("🔵 Claude analisando outputs...")
    try:
        claude_out = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            messages=[{"role": "user", "content": output_prompt}]
        )
        claude_output = claude_out.content[0].text
        print(f"✓ Claude completou ({len(claude_output)} chars)")
    except Exception as e:
        print(f"✗ Claude falhou: {e}")
        claude_output = f"[ERRO: {e}]"

    print()
    print("🟢 GPT analisando outputs...")
    try:
        gpt_out = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": output_prompt + f"\n\nCLAUDE ANALISOU:\n{claude_output}\n\nAgora complemente a análise."
            }],
            max_tokens=3000
        )
        gpt_output = gpt_out.choices[0].message.content
        print(f"✓ GPT completou ({len(gpt_output)} chars)")
    except Exception as e:
        print(f"✗ GPT falhou: {e}")
        gpt_output = f"[ERRO: {e}]"

    # FASE 3: Consenso e Recomendações
    print()
    print("="*80)
    print("FASE 3: CONSENSO E COMANDO CORRETO")
    print("="*80)
    print()

    consensus_prompt = f"""Com base no estudo completo:

MAPEAMENTO DE FUNÇÕES:
{claude_mapping[:2000]}...

ANÁLISE DE OUTPUTS:
{claude_output[:2000]}...
{gpt_output[:2000]}...

GERE:

1. **COMANDO TESTE MÍNIMO** (deve rodar em ~5min e gerar CSVs)
   - Inclua TODOS os argumentos de output necessários
   - Explique o que cada argumento faz
   - smoke_months=1 para teste rápido

2. **CHECKLIST DE OUTPUTS** - Como verificar que funcionou:
   - Quais CSVs devem ser gerados?
   - Qual o conteúdo esperado de cada um?
   - Como interpretar os resultados?

3. **PRÓXIMAS ITERAÇÕES** - Comandos progressivos:
   - Do mais simples ao mais complexo
   - Critérios para avançar de fase

Formato: Markdown, comandos em bash code blocks."""

    print("🟢 GPT gerando consenso...")
    try:
        consensus = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": consensus_prompt}],
            max_tokens=3000
        )
        consensus_text = consensus.choices[0].message.content
        print(f"✓ Consenso gerado ({len(consensus_text)} chars)")
    except Exception as e:
        print(f"✗ Consenso falhou: {e}")
        consensus_text = f"[ERRO: {e}]"

    # Salvar tudo
    study_output = {
        "phase1_mapping": {
            "claude": claude_mapping,
            "gpt": gpt_mapping
        },
        "phase2_outputs": {
            "claude": claude_output,
            "gpt": gpt_output
        },
        "phase3_consensus": consensus_text,
        "functions_found": len(functions),
        "code_size": len(code)
    }

    with open("ESTUDO_COMPLETO_SELECTOR.md", "w") as f:
        f.write("# ESTUDO COMPLETO DO SELECTOR21.PY\n\n")
        f.write(f"**Funções identificadas**: {len(functions)}\n")
        f.write(f"**Código**: {len(code):,} caracteres\n\n")
        f.write("---\n\n")
        f.write("## FASE 1: MAPEAMENTO DE FUNÇÕES\n\n")
        f.write("### Claude:\n\n")
        f.write(claude_mapping)
        f.write("\n\n### GPT:\n\n")
        f.write(gpt_mapping)
        f.write("\n\n---\n\n")
        f.write("## FASE 2: ANÁLISE DE OUTPUTS\n\n")
        f.write("### Claude:\n\n")
        f.write(claude_output)
        f.write("\n\n### GPT:\n\n")
        f.write(gpt_output)
        f.write("\n\n---\n\n")
        f.write("## FASE 3: CONSENSO E COMANDO CORRETO\n\n")
        f.write(consensus_text)

    print()
    print("="*80)
    print("💾 ESTUDO SALVO")
    print("="*80)
    print()
    print("  • ESTUDO_COMPLETO_SELECTOR.md")
    print("  • LEARNING_SELECTOR_PROFUNDO.md")
    print()
    print("📖 Próximo passo:")
    print("   cat ESTUDO_COMPLETO_SELECTOR.md")
    print()

    return study_output


if __name__ == "__main__":
    study_selector_deep()
