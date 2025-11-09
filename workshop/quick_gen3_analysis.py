#!/usr/bin/env python3
"""Quick Gen 3 Analysis"""
import pandas as pd
from pathlib import Path

results = []
for test_dir in Path("resultados/gen3").glob("*/"):
    csv_path = test_dir / "leaderboard_base.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            row = df.iloc[0].to_dict()
            row['test_name'] = test_dir.name
            results.append(row)

if not results:
    print("❌ Nenhum resultado Gen 3 encontrado")
    exit(1)

df = pd.DataFrame(results)

print("=" * 80)
print(f"GERAÇÃO 3 - ANÁLISE RÁPIDA ({len(df)} testes)")
print("=" * 80)

print("\n### TOP 10 POR PnL")
top = df.nlargest(10, 'total_pnl')[['test_name', 'timeframe', 'total_pnl', 'sharpe', 'hit', 'n_trades']]
print(top.to_string(index=False))

print("\n### PERFORMANCE POR TIMEFRAME")
tf_stats = df.groupby('timeframe').agg({
    'total_pnl': ['mean', 'max', 'min'],
    'sharpe': 'mean'
}).round(2)
print(tf_stats)

# Count profitable
profitable = df[df['total_pnl'] > 0]
print(f"\n### ESTRATÉGIAS LUCRATIVAS: {len(profitable)}/{len(df)}")
if len(profitable) > 0:
    print(profitable[['test_name', 'timeframe', 'total_pnl', 'sharpe']].to_string(index=False))

print("\n" + "=" * 80)
print(f"✅ Gen 3 completo: {len(df)} testes analisados")
print(f"⭐ Lucrativas: {len(profitable)} ({100*len(profitable)/len(df):.1f}%)")
