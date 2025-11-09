# Falcom BotScalp — Arquitetura Técnica (Resumo)

Status: fase final → foco em refatoração do `workspace/` e paper trading.
Mandato das IAs: finalizar, validar, preparar produção supervisionada (sem execução autônoma real).
Controle do Mentor: total (pausar, refazer, reverter, priorizar).

## Camadas
- datafeeds: klines/aggTrades/depth → NPZ por janela (evita parquet remoto e dependency hell; AMI com CUDA/stack DL).
- selectors: sinais/ensembles, thresholds, neutral_band, merges com DL.
- execution (sim): executor para paper trading (custos, fees, slippage, tick, stops/timeout).
- backtesting & validation: WFO (multi-janela), métricas por janela e consolidado.
- evolution & analysis: tuning (seeds, GA/Grid), calibração, relatórios md/png.

## Observabilidade Remota (GPU/treinos DL)
- tail -F (200 últimas):
  `ssh -i ~/.ssh/id_botscalp root@94.72.167.122 'tail -n 200 -F /root/botscalpv3/out/training_follow.log'`
- EPOCH/OOS/SUMMARY:
  `ssh -i ~/.ssh/id_botscalp root@94.72.167.122 'tail -F /root/botscalpv3/out/training_follow.log | rg --line-buffered -e "EPOCH|OOS|SUMMARY|===="'`
- less + follow:
  `ssh -t -i ~/.ssh/id_botscalp root@94.72.167.122 'less +F /root/botscalpv3/out/training_follow.log'`
- status processo:
  `ssh -i ~/.ssh/id_botscalp root@94.72.167.122 'bash -lc "ps -fp $(cat /root/botscalpv3/out/training_follow.pid) 2>/dev/null || echo not-running"'`
- parar treino:
  `ssh -i ~/.ssh/id_botscalp root@94.72.167.122 'bash -lc "kill $(cat /root/botscalpv3/out/training_follow.pid) 2>/dev/null || true; rm -f /root/botscalpv3/out/training_follow.pid"'`
- GPU ao vivo:
  `ssh -t -i ~/.ssh/id_botscalpv3 root@94.72.167.122 'nvidia-smi -l 2 || watch -n 5 date'`

## Onde olhar resultados DL
- `/root/botscalpv3/out/dl/`: oos_probs_*.csv, metrics_*.json, DL_LEADERBOARD.csv, DL_SUMMARY.csv
- logs: `/root/botscalpv3/out/training_tmux.log`, `/root/botscalpv3/out/training_follow.log`
- baixar artefatos:
  `rsync -az -e "ssh -i ~/.ssh/id_botscalpv3" root@94.72.167.122:/root/botscalpv3/out/dl/ out/dl-remote-run/`

## PnL “na prática”
- WFO simula execução por janela → gera `leaderboard_base.csv`, `leaderboard_combos.csv` (total_pnl, n_trades, sharpe, maxdd…)
- trades detalhados: `best_trades.csv` (entry_t, entry_px, exit_t, exit_px, side, pnl…)
- configs de execução: `runtime_config.json` (contracts, contract_value, fee_perc, slippage, max_hold…)
- DL produz probabilidades, não PnL → precisa de merge (MERGED_meta_*.csv) e replay/executor para virar trades e métricas.

## Dicas rápidas (estabilidade)
- Calibração (isotônica/platt) + target trades/dia → estabilidade.
- neutral_band > 0; thresholds simétricos (long/short).
- gating por ATR/VHF/regime.
- evitar “janela campeã”: agregação entre janelas (mediana/consenso).
