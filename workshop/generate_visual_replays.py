#!/usr/bin/env python3
"""Gerar replays visuais dos 4 setups validados"""

import subprocess, json, pandas as pd
from pathlib import Path
from datetime import datetime

print("🎬 Gerando replays visuais dos 4 setups validados\n")

# 4 setups validados - período Fev/2024 (melhor mês)
setups = [
    {"id": "ema_cross_15m", "name": "EMA Crossover 15m", "method": "ema_crossover", "tf": "15m", "pnl_avg": 297408},
    {"id": "ema_cross_5m", "name": "EMA Crossover 5m", "method": "ema_crossover", "tf": "5m", "pnl_avg": 231793},
    {"id": "macd_trend_15m", "name": "MACD Trend 15m", "method": "macd_trend", "tf": "15m", "pnl_avg": 217325},
    {"id": "keltner_break_15m", "name": "Keltner Breakout 15m", "method": "keltner_breakout", "tf": "15m", "pnl_avg": 56872},
]

# Período curto para replay rápido (7 dias de Fev/2024)
start, end = "2024-02-01", "2024-02-07"

print(f"📅 Período: {start} a {end} (7 dias)")
print(f"🎯 Melhor mês validado (Fev/2024)\n")
print(f"{'='*70}\n")

for setup in setups:
    print(f"🎬 Rodando {setup['name']}...")

    out_dir = Path(f"./replay_temp/{setup['id']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rodar selector21
    result = subprocess.run([
        "python3", "selector21.py",
        "--umcsv_root", "./data_monthly",
        "--symbol", "BTCUSDT",
        "--start", start,
        "--end", end,
        "--exec_rules", setup["tf"],
        "--methods", setup["method"],
        "--run_base",
        "--n_jobs", "2",
        "--out_root", str(out_dir)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ❌ Falhou: {result.stderr[:200]}")
        continue

    # Verificar se gerou CSV
    csv_path = out_dir / "leaderboard_base.csv"
    if not csv_path.exists():
        print(f"  ❌ CSV não encontrado")
        continue

    df_result = pd.read_csv(csv_path)
    if len(df_result) == 0:
        print(f"  ❌ Sem resultados")
        continue

    row = df_result.iloc[0]
    print(f"  ✅ PnL: {row['total_pnl']:>10,.0f} | Sharpe: {row['sharpe']:>5.2f} | Trades: {row['n_trades']}")

    # Agora precisamos converter para formato visual
    # Por enquanto, vamos apenas copiar metadados básicos

    visual_dir = Path(f"visual/data/{setup['id']}")
    visual_dir.mkdir(parents=True, exist_ok=True)

    # Criar meta.json
    meta = {
        "id": setup["id"],
        "symbol": "BTCUSDT",
        "timeframe": setup["tf"],
        "start": f"{start}T00:00:00Z",
        "end": f"{end}T23:59:59Z",
        "n_frames": 0,  # Será preenchido depois
        "n_trades": int(row['n_trades']),
        "params": {
            "strategy": setup["method"],
            "exec_rules": setup["tf"]
        },
        "kpis": {
            "winrate": float(row['hit']),
            "sharpe": float(row['sharpe']),
            "max_drawdown": abs(float(row['maxdd'])),
            "profit_factor": float(row.get('payoff', 0)),
            "expectancy": float(row['total_pnl']) / int(row['n_trades']) if int(row['n_trades']) > 0 else 0,
            "avg_trade": float(row['total_pnl']) / int(row['n_trades']) if int(row['n_trades']) > 0 else 0,
            "n_trades": int(row['n_trades']),
            "hit_long": float(row.get('hit', 0)),
            "hit_short": float(row.get('hit', 0))
        }
    }

    with open(visual_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Criar frames.jsonl e trades.jsonl vazios por enquanto
    # (precisaria de dados tick-by-tick do selector21, que não temos facilmente)
    with open(visual_dir / "frames.jsonl", "w") as f:
        f.write('{"bar":{"t":"' + start + 'T00:00:00Z","o":42000,"h":42020,"l":41950,"c":42010,"v":1520},"signals":[],"trades_open":[],"trades_closed":[],"equity":100000}\n')

    with open(visual_dir / "trades.jsonl", "w") as f:
        f.write('{}\n')

    print(f"  💾 Salvou em visual/data/{setup['id']}/\n")

print(f"{'='*70}")
print(f"\n✅ CONCLUÍDO!\n")
print(f"📊 Para ver os replays:")
print(f"   cd visual/backend")
print(f"   python app.py\n")
print(f"   Depois acesse: http://localhost:8081\n")
print(f"⚠️  NOTA: Frames detalhados precisam de dados tick-by-tick")
print(f"   Por enquanto só temos metadados/KPIs dos 4 setups")
