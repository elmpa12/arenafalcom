#!/usr/bin/env python3
"""
IAS STUDY SELECTOR - BotScalp V3

As IAs (Claude + GPT) estudam COMPLETAMENTE o selector21.py
e TODAS as suas configurações antes de decidir parâmetros.

Uso:
    python3 ias_study_selector.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()


def study_selector21_complete():
    """As IAs estudam o selector21.py COMPLETO."""

    # Ler selector21.py COMPLETO
    with open("selector21.py") as f:
        selector_code = f.read()

    # Ler help completo
    import subprocess
    help_output = subprocess.run(
        ["python3", "selector21.py", "--help"],
        capture_output=True,
        text=True
    ).stdout

    print("\n" + "="*80)
    print("🤖 IAs ESTUDANDO SELECTOR21.PY COMPLETO")
    print("="*80)
    print(f"\nCódigo: {len(selector_code)} caracteres")
    print(f"Help: {len(help_output)} caracteres")
    print(f"\nParâmetros identificados:")
    params = help_output.count("--")
    print(f"  Total: ~{params} parâmetros configuráveis")
    print()

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Prompt para análise COMPLETA
    analysis_prompt = f"""Você é um especialista em backtesting de trading systems.

TAREFA: Analisar COMPLETAMENTE o selector21.py e DECIDIR TODOS os parâmetros ideais.

HELP DO SELECTOR21:
{help_output}

CATEGORIAS DE PARÂMETROS:

1. DATA LOADING:
   - data_dir, data_glob, symbol, interval, start, end
   - umcsv_root, force_fallback, smoke_months, max_rows_loader

2. EXECUTION RULES:
   - exec_rules (timeframes: 1m, 5m, 15m)
   - methods (14 métodos disponíveis)
   - long_only, run_base, run_combos

3. COMBOS:
   - combo_ops (AND, MAJ, SEQ)
   - combo_cap, combo_window, combo_min_votes

4. RISK MANAGEMENT:
   - contracts, contract_value, fee_perc, slippage
   - tick_size, max_hold, futures

5. STOPS & TAKE-PROFITS:
   - use_atr_stop, atr_stop_len, atr_stop_mult, trailing
   - timeout_mode, atr_timeout_len, atr_timeout_mult
   - use_atr_tp, atr_tp_len, atr_tp_mult
   - hard_stop_usd, hard_tp_usd
   - use_candle_stop, candle_stop_lookback

6. WALK-FORWARD:
   - walkforward (true/false)
   - wf_train_months, wf_val_months, wf_step_months
   - wf_grid_mode (light/medium/full)
   - wf_top_n, wf_expand

7. MACHINE LEARNING:
   - run_ml (true/false)
   - ml_model_kind (auto, xgb, rf, logreg)
   - ml_horizon, ml_ret_thr, ml_lags
   - ml_use_agg, ml_use_depth
   - ml_opt_thr, ml_thr_grid, ml_thr_fixed
   - ml_neutral_band
   - ml_add_base_feats, ml_add_combo_feats
   - ml_calibrate, ml_recency_mode

8. FILTERS:
   - atr_z_min (threshold ATR z-score)
   - vhf_min (Vertical Horizontal Filter)
   - cvd_slope_min, imbalance_min

9. PERFORMANCE:
   - n_jobs, par_backend (process/thread)
   - lowmem, mem_lookback_bars

10. OUTPUT:
    - min_trades, min_hit, min_pnl, min_sharpe, max_dd
    - out_root, out_report, out_wf_ml
    - ml_save_dir, print_top10

PARA CADA CATEGORIA, DECIDA:
1. Valores ideais para backtest ROBUSTO
2. Justificativa técnica
3. Trade-offs considerados

FORMATO DA RESPOSTA:
```
CATEGORIA: <nome>
PARÂMETROS:
  --param1 <valor>  # Justificativa
  --param2 <valor>  # Justificativa
  ...
```

Seja COMPLETO. Cubra TODAS as 10 categorias.
"""

    print("📊 CLAUDE analisando...")
    claude_response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4000,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    claude_analysis = claude_response.content[0].text

    print("✅ Claude completou análise")
    print()
    print("📊 GPT analisando...")

    gpt_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": analysis_prompt + f"\n\nCLAUDE DISSE:\n{claude_analysis}\n\nAgora dê sua análise considerando o que Claude disse."
        }],
        max_tokens=4000
    )
    gpt_analysis = gpt_response.choices[0].message.content

    print("✅ GPT completou análise")
    print()

    # Salvar análises
    with open("selector_analysis_claude.txt", "w") as f:
        f.write(claude_analysis)

    with open("selector_analysis_gpt.txt", "w") as f:
        f.write(gpt_analysis)

    print("="*80)
    print("📝 ANÁLISES SALVAS")
    print("="*80)
    print("\n  • selector_analysis_claude.txt")
    print("  • selector_analysis_gpt.txt")
    print()

    # Gerar comando final consensuado
    print("🎯 Gerando comando final consensuado...")

    consensus_prompt = f"""Com base nas duas análises:

CLAUDE:
{claude_analysis[:2000]}...

GPT:
{gpt_analysis[:2000]}...

Gere um COMANDO COMPLETO do selector21.py com TODOS os parâmetros decididos.

Formato:
```bash
python3 selector21.py \\
  --symbol BTCUSDT \\
  --start 2024-01-01 \\
  --end 2024-06-01 \\
  --data_dir ./data_monthly \\
  [... TODOS OS OUTROS PARÂMETROS ...]
```

Seja COMPLETO. Inclua TODOS os parâmetros importantes das 10 categorias.
"""

    final_cmd = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": consensus_prompt}],
        max_tokens=2000
    ).choices[0].message.content

    with open("SELECTOR_COMANDO_FINAL.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Comando gerado pelas IAs após análise COMPLETA do selector21.py\n\n")
        f.write(final_cmd)

    print("✅ Comando salvo em: SELECTOR_COMANDO_FINAL.sh")
    print()
    print("="*80)
    print("🎯 ESTUDO COMPLETO!")
    print("="*80)
    print("\nArquivos gerados:")
    print("  1. selector_analysis_claude.txt - Análise do Claude")
    print("  2. selector_analysis_gpt.txt - Análise do GPT")
    print("  3. SELECTOR_COMANDO_FINAL.sh - Comando completo consensuado")
    print()

    return {
        "claude": claude_analysis,
        "gpt": gpt_analysis,
        "command": final_cmd
    }


if __name__ == "__main__":
    study_selector21_complete()
