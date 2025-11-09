#!/usr/bin/env python3
"""
GENERATION 2 ANALYSIS - 30 Ultra-Fast Tests
Analyzes rapid test results and generates Generation 3 hypotheses
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def load_all_results():
    """Load all 30 rapid test results"""
    results = []
    rapid_dir = Path("resultados/rapid")

    for test_dir in rapid_dir.glob("*/"):
        csv_path = test_dir / "leaderboard_base.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row['test_name'] = test_dir.name
                results.append(row)

    return pd.DataFrame(results)

def analyze_patterns(df):
    """Identify what worked BEST (even if negative)"""

    print("=" * 80)
    print("GENERATION 2 - ANÁLISE DE 30 TESTES ULTRA-RÁPIDOS")
    print("=" * 80)

    # 1. Rankings
    print("\n### TOP 10 POR PnL (Menos Negativo = Melhor)")
    top_pnl = df.nlargest(10, 'total_pnl')[['test_name', 'method', 'total_pnl', 'sharpe', 'hit', 'n_trades']]
    print(top_pnl.to_string(index=False))

    print("\n### TOP 10 POR SHARPE (Menos Negativo = Melhor)")
    top_sharpe = df.nlargest(10, 'sharpe')[['test_name', 'method', 'sharpe', 'total_pnl', 'hit', 'n_trades']]
    print(top_sharpe.to_string(index=False))

    print("\n### TOP 10 POR HIT RATE")
    top_hit = df.nlargest(10, 'hit')[['test_name', 'method', 'hit', 'total_pnl', 'sharpe', 'n_trades']]
    print(top_hit.to_string(index=False))

    # 2. Aggregate by method
    print("\n### PERFORMANCE POR MÉTODO (Média)")
    method_stats = df.groupby('method').agg({
        'total_pnl': 'mean',
        'sharpe': 'mean',
        'hit': 'mean',
        'n_trades': 'mean'
    }).round(2)
    print(method_stats.sort_values('total_pnl', ascending=False))

    # 3. Extract week/period patterns
    print("\n### ANÁLISE POR PERÍODO")
    week_performance = defaultdict(list)

    for _, row in df.iterrows():
        test_name = row['test_name']

        # Extract period from test name
        if 'w1' in test_name:
            period = 'Semana 1 (Jan)'
        elif 'w2' in test_name:
            period = 'Semana 2 (Jan)'
        elif 'w3' in test_name:
            period = 'Semana 3 (Jan)'
        elif 'w4' in test_name:
            period = 'Semana 4 (Jan)'
        elif 'feb' in test_name:
            period = 'Fevereiro'
        elif 'mar' in test_name:
            period = 'Março'
        else:
            period = 'Unknown'

        week_performance[period].append(row['total_pnl'])

    for period, pnls in sorted(week_performance.items()):
        avg_pnl = sum(pnls) / len(pnls)
        print(f"{period:20s} → Avg PnL: {avg_pnl:>12,.2f} ({len(pnls)} testes)")

    # 4. Identify best combinations
    print("\n### MELHORES COMBINAÇÕES (Method + Period)")
    combos = []
    for _, row in df.iterrows():
        test_name = row['test_name']
        period = 'w1' if 'w1' in test_name else 'w2' if 'w2' in test_name else 'feb' if 'feb' in test_name else 'mar' if 'mar' in test_name else 'other'
        combos.append({
            'combo': f"{row['method']}_{period}",
            'pnl': row['total_pnl'],
            'sharpe': row['sharpe'],
            'hit': row['hit']
        })

    combo_df = pd.DataFrame(combos).nlargest(10, 'pnl')
    print(combo_df.to_string(index=False))

    return {
        'best_method': method_stats['total_pnl'].idxmax(),
        'best_period': max(week_performance.items(), key=lambda x: sum(x[1])/len(x[1]))[0],
        'top_10_pnl': top_pnl.to_dict('records'),
        'top_10_sharpe': top_sharpe.to_dict('records'),
        'method_stats': method_stats.to_dict(),
        'week_performance': {k: sum(v)/len(v) for k, v in week_performance.items()}
    }

def generate_hypotheses(analysis):
    """Generate Generation 3 hypotheses based on patterns"""

    print("\n" + "=" * 80)
    print("GERAÇÃO 3 - HIPÓTESES E EXPERIMENTOS")
    print("=" * 80)

    best_method = analysis['best_method']
    best_period = analysis['best_period']

    hypotheses = [
        f"### HIPÓTESE 1: Método {best_method} tem melhor potencial",
        f"   → Gen 2 mostrou que {best_method} teve menor perda média",
        f"   → Testar {best_method} com diferentes parâmetros de risco",
        "",
        f"### HIPÓTESE 2: Período {best_period} é mais favorável",
        f"   → Gen 2 mostrou melhor desempenho relativo neste período",
        f"   → Focar testes em períodos similares (volatilidade/condições de mercado)",
        "",
        "### HIPÓTESE 3: Hit Rate vs Payoff tradeoff",
        "   → Alguns métodos têm hit >40% mas perdem dinheiro",
        "   → Problema: payoff ratio insuficiente ou stops mal calibrados",
        "   → Testar diferentes configurações de stop/target",
        "",
        "### HIPÓTESE 4: Timeframe 1m pode ser ruidoso demais",
        "   → Todos os testes usaram 1m",
        "   → Testar 5m e 15m para reduzir ruído",
        "",
        "### HIPÓTESE 5: Período de teste muito curto",
        "   → 1 semana = 50-300 trades",
        "   → Aumentar para 2-4 semanas para melhor amostra estatística",
    ]

    for h in hypotheses:
        print(h)

    return hypotheses

def generate_gen3_tests():
    """Generate Generation 3 test configurations"""

    print("\n" + "=" * 80)
    print("GERAÇÃO 3 - CONFIGURAÇÃO DE TESTES")
    print("=" * 80)

    gen3_tests = []

    # Test 1-10: Best method from Gen2 with different periods
    best_methods = ['rsi_reversion', 'ema_crossover']  # Based on hit rate analysis
    periods = [
        ("2024-01-01", "2024-01-15", "jan_w12"),
        ("2024-01-15", "2024-02-01", "jan_w34"),
        ("2024-02-01", "2024-02-15", "feb_w12"),
        ("2024-02-15", "2024-03-01", "feb_w34"),
        ("2024-03-01", "2024-03-15", "mar_w12"),
    ]

    for method in best_methods:
        for start, end, period_name in periods:
            gen3_tests.append({
                "name": f"gen3_{period_name}_{method.split('_')[0]}",
                "args": [
                    "--umcsv_root", "./data_monthly",
                    "--symbol", "BTCUSDT",
                    "--start", start,
                    "--end", end,
                    "--exec_rules", "1m",
                    "--methods", method,
                    "--run_base",
                    "--n_jobs", "2",
                    "--out_root", f"./resultados/gen3/gen3_{period_name}_{method.split('_')[0]}"
                ]
            })

    # Test 11-20: Different timeframes (5m, 15m)
    for tf in ["5m", "15m"]:
        for method in ['rsi_reversion', 'ema_crossover', 'trend_breakout', 'macd_trend', 'vwap_reversion']:
            gen3_tests.append({
                "name": f"gen3_{tf}_{method.split('_')[0]}",
                "args": [
                    "--umcsv_root", "./data_monthly",
                    "--symbol", "BTCUSDT",
                    "--start", "2024-01-01",
                    "--end", "2024-01-15",
                    "--exec_rules", tf,
                    "--methods", method,
                    "--run_base",
                    "--n_jobs", "2",
                    "--out_root", f"./resultados/gen3/gen3_{tf}_{method.split('_')[0]}"
                ]
            })

    # Test 21-30: Alternative methods not tested in Gen2
    alternative_methods = [
        'bollinger_breakout', 'keltner_breakout', 'donchian_breakout',
        'opening_range_breakout', 'opening_range_reversal', 'ema_pullback',
        'volume_breakout', 'pivot_reversion', 'pivot_breakout', 'rsi_ema_combo'
    ]

    for i, method in enumerate(alternative_methods[:10]):
        gen3_tests.append({
            "name": f"gen3_alt_{method.split('_')[0]}",
            "args": [
                "--umcsv_root", "./data_monthly",
                "--symbol", "BTCUSDT",
                "--start", "2024-01-01",
                "--end", "2024-01-15",
                "--exec_rules", "1m",
                "--methods", method,
                "--run_base",
                "--n_jobs", "2",
                "--out_root", f"./resultados/gen3/gen3_alt_{method.split('_')[0]}"
            ]
        })

    print(f"\n✅ Gerado {len(gen3_tests)} testes para Geração 3")
    print(f"   - {len([t for t in gen3_tests if 'w12' in t['name'] or 'w34' in t['name']])} testes com períodos mais longos")
    print(f"   - {len([t for t in gen3_tests if '5m' in t['name'] or '15m' in t['name']])} testes com timeframes maiores")
    print(f"   - {len([t for t in gen3_tests if 'alt' in t['name']])} testes com métodos alternativos")

    # Save config
    with open("gen3_tests_config.json", "w") as f:
        json.dump(gen3_tests, f, indent=2)

    print(f"\n💾 Configuração salva em: gen3_tests_config.json")

    return gen3_tests

def main():
    # Load results
    print("📂 Carregando resultados dos 30 testes rápidos...")
    df = load_all_results()
    print(f"✅ {len(df)} resultados carregados\n")

    # Analyze
    analysis = analyze_patterns(df)

    # Generate hypotheses
    hypotheses = generate_hypotheses(analysis)

    # Generate Gen 3 tests
    gen3_tests = generate_gen3_tests()

    # Save analysis
    Path("evolution/gen2").mkdir(parents=True, exist_ok=True)

    with open("evolution/gen2/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    with open("evolution/gen2/HYPOTHESES.md", "w") as f:
        f.write("# GERAÇÃO 2 → GERAÇÃO 3: APRENDIZADOS E HIPÓTESES\n\n")
        f.write("## Análise de 30 Testes Ultra-Rápidos\n\n")
        f.write(f"**Melhor Método (menor perda)**: {analysis['best_method']}\n")
        f.write(f"**Melhor Período**: {analysis['best_period']}\n\n")
        f.write("## Hipóteses para Geração 3\n\n")
        f.write("\n".join(hypotheses))
        f.write("\n\n## Próximos Passos\n\n")
        f.write("```bash\n")
        f.write("# Rodar Geração 3 (30 testes, ~5-15s cada, 20 paralelos)\n")
        f.write("python3 run_from_config.py gen3_tests_config.json --parallel 20\n")
        f.write("```\n")

    print("\n" + "=" * 80)
    print("✅ ANÁLISE COMPLETA")
    print("=" * 80)
    print(f"📊 Análise salva em: evolution/gen2/analysis.json")
    print(f"📝 Hipóteses salvas em: evolution/gen2/HYPOTHESES.md")
    print(f"⚙️  Config Gen 3: gen3_tests_config.json")
    print("\n🚀 Próximo comando:")
    print("   python3 run_from_config.py gen3_tests_config.json --parallel 20")

if __name__ == "__main__":
    main()
