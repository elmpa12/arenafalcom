# Falcom BotScalp — Workspace

## Objetivo imediato
1) Arrumar a casa (refatorar/organizar diretórios e imports)
2) Entregar pipeline mínimo de Paper Trading (selector → executor simulado → relatórios)

## Estrutura-alvo (ajustável à realidade do repo)
core/
  datafeeds/      # klines, aggtrades, depth, normalização, NPZ
  selectors/      # sinais/ensembles, merges DL, thresholds
  execution/      # executor simulado (custos/risco), depois live gateado
  backtesting/    # WFO, métricas por janela e consolidado
  validation/     # sanity checks (ECE, leakage, ranges)
  evolution/      # tuning/ablation/seeds, pipelines WFO
  analysis/       # relatórios e gráficos
configs/
  system.yaml
  trading.json        # parâmetros de paper (notional, fee, slippage, tick, stops)
  exchange_testnet.json
tools/
  merge_dl.py         # (existe) merge com selector
  replay_merged.py    # (NOVO) converter MERGED_meta_*.csv em PnL/trades simulados
logs/
reports/
  daily.md
  weekly.md
  paper_config.md

## KPIs paper
PnL, Winrate, Profit Factor, Sharpe, MDD, trades/dia, turnover, dispersão por janela.

## Definição de pronto (DoD)
- Estrutura navegável; imports OK
- Pipeline paper executa fim-a-fim com custos/risco
- Relatórios automáticos gerados (daily/weekly)
- Documentos .md atualizados
