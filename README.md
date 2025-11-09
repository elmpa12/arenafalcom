# Falcom Arena & BotScalp Workspace

This repository mirrors the production tree at `/opt/falcom`:

| Folder       | Purpose |
|--------------|---------|
| `arena/`     | Falcom Arena (Flask + Socket.IO + Eventlet) with web UI, mentor controls, `/say/*` API, CLI. |
| `workspace/` | BotScalp V3 code (selectors, execution, paper trading tools, reports). |
| `workshop/`  | R&D sandbox (scripts, docs, prototypes) with heavy data directories ignored via `.gitignore`. |
| `ops/`       | Systemd + nginx manifests used in prod. |
| `scripts/`   | Helper shell scripts (start/stop/reset/backup). |
| `docs/`      | Architecture notes, journals, plans. |

## 1. Requirements
- Ubuntu 22.04+
- Python 3.12+
- systemd + nginx + certbot
- `/opt/falcom/env/.env` containing at least:
  ```bash
  OPENAI_API_KEY=...
  ANTHROPIC_API_KEY=...
  ARENA_API_TOKEN=super-secret
  ```

## 2. Install / Update
```bash
python -m venv /opt/falcom/.venv
source /opt/falcom/.venv/bin/activate
pip install -r arena/requirements.txt   # list flask, flask-socketio, eventlet, requests, websocket-client, python-dotenv, etc.
```

## 3. Systemd service
```bash
sudo cp ops/systemd/falcom-arena.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now falcom-arena
sudo systemctl status falcom-arena -n 50
```
The unit sets `FALCOM_HOME=/opt/falcom`, `FALCOM_PORT=5051` and runs `arena/arena.py` inside the venv.

## 4. Nginx + TLS
```bash
sudo cp ops/nginx/falcom-arena.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/falcom-arena.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d beta.falcomlabs.com   # replace host
```
Traffics hits nginx → `127.0.0.1:5051`, with gzip enabled and WebSocket upgrade headers preserved.

## 5. Arena CLI (`arena`)
Symlink `/usr/local/bin/arena → arena/tools/arena_cli.py`.
```bash
export ARENA_API_TOKEN=super-secret
export ARENA_HOST=beta.falcomlabs.com     # or set ARENA_BASE_URL=https://beta.falcomlabs.com
export FALCOM_PORT=5051
```
Commands:
```bash
arena status
arena say @ernest "[ANÁLISE] mapear workspace"
arena say @garapa "Criticar o plano"
arena say @alfred --dryrun "Gerar inventário CSV"
```
Output is JSON (`ok`, `agent`, `text`, `executed`).

## 6. HTTP API (all require `X-Auth: $ARENA_API_TOKEN`)
| Endpoint | Method | Body | Notes |
|----------|--------|------|-------|
| `/say/<agent>` | POST | `{"text": "...", "dryrun": false}` | Agents: `ernest`, `garapa`, `alfred`. Dry-run prevents executor. |
| `/status` | GET | – | Returns `running`, `sandbox`, `supervised_only`, `approval_mode`, `allow_exec`, `allow_global_access`, `workspace_root`, `models`. |
| `/approval_mode` | POST | `{"mode": "read_only|auto|full"}` | Adjusts supervision tier (auto-tunes exec + FS scope). |
| `/workspace_access` | POST | `{"allow_global": true|false}` | Explicit FS scope override. |
| `/toggle_supervision` | POST | `{}` or `{"enabled": bool}` | Same as Mentor toggle. |
| `/toggle_exec` | POST | `{}` or `{"enabled": bool}` | Enables/disables executor pipeline. |
| `/restart` | POST | optional config overrides | Stops current debate loop and restarts. |
| `/revert` | POST | – | Resets config to defaults. |
| `/inject`, `/start`, `/stop`, `/toggle` | existing UI endpoints. |

Safety rules:
- `supervised_only=true` → executor always blocked.
- `allow_exec=false` → Alfred only returns code; no execution.
- `dryrun=true` on `/say/alfred` → skip executor even if allowed.
- Filesystem scope: when global access disabled, commands referencing paths outside `workspace_root`, `..`, or `~` return `[policy] Global filesystem access bloqueado...`.

## 7. Workspace helpers
- **Checkpoints:** `workspace/tools/checkpoint.sh` (keeps 30 tags by default)
  ```bash
  cd /opt/falcom/workspace
  ./tools/checkpoint.sh "pre-migration"
  CHECKPOINT_PREFIX=stage- CHECKPOINT_MAX=50 ./tools/checkpoint.sh
  ```
- **Inventory:** `./tools/inventory_workspace.py` writes `reports/workspace_inventory.csv`.
- **Replay MERGED_meta:** `./tools/replay_merged.py --merged-glob "out/dl/MERGED_meta_*.csv" --config configs/trading.json`.
- **Paper executor stub:** `core/execution/paper_executor.py` is where Alfred wires thresholds, stops, fees, ATR, etc.

## 8. Logs & paths
- `arena/logs/*.log` – GPT/Claude/Arena transcripts
- `arena/dialogue/latest.log` – chronological log (UI + `/say` + executor)
- `workspace/logs/`, `workspace/reports/` – generated outputs (ignored by git)
- `ops/` manifests show exactly what is deployed on host

## 9. Day-to-day workflow
1. Populate `.env` with API keys + `ARENA_API_TOKEN`.
2. `systemctl enable --now falcom-arena` (service restarts automatically on failure).
3. Configure nginx + certbot (`https://beta.falcomlabs.com`).
4. Operate via UI or CLI:
   - `arena status`
   - `arena say @ernest ...`
   - Use Mentor panel to switch `Approval Mode`, `Filesystem`, `Supervision`, `Exec` toggles.
5. Alfred works in `/opt/falcom/workspace`, uses checkpoints around every migration / patch.
6. Push updates back here: `git add ... && git commit && git push`.

## 10. .gitignore highlights
- Logs, backups, archives (`*.zip`, `*.tar.gz`), `.venv/`, `.env`, `arena/dialogue/`, `arena/memory/`, `workspace/logs/`, heavy workshop data dirs, node_modules, `venv`, etc., are excluded to avoid shipping sensitive or huge files.

With these scripts, manifests, and endpoints, you can reproduce the full stack (Arena + workspace) on any host, control agents individually via `/say/*`, and keep production state in lockstep with this repository.
