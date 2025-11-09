# DL Ops — Observabilidade rápida

Logs:
- `tail -n 200 -F /root/botscalpv3/out/training_follow.log`
- `rg "EPOCH|OOS|SUMMARY|===="`
- `nvidia-smi -l 2`

Interpretar:
- **EPOCH**: brier (↓), acc, auc/pr_auc, sps
- **OOS**: metrics por janela → `metrics_<model>_<tf>_winXX.json`
- **SUMMARY**: médias por modelo

Artefatos:
- `out/dl/` : `oos_probs_*.csv`, `metrics_*.json`, `DL_LEADERBOARD.csv`, `DL_SUMMARY.csv`
- `MERGED_meta_*.csv` → precisa de replay/executor p/ virar PnL

Erros comuns:
- **OOM**: reduzir batch/lags ou #models
- **Métricas piorando**: revisar label (horizon), neutral_band, range temporal
- **AUC alto x acc alto**: olhar pr_auc e ece (calibração)
