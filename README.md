# Falcom Arena & Workspace

This repository mirrors the production layout at `/opt/falcom`:

- `arena/` – Socket.IO + Eventlet control room (Falcom Arena), including web UI, API endpoints, and mentor tools.
- `workspace/` – BotScalp V3 code tree (selectors, execution, reports, tools).
- `ops/` – Systemd + nginx manifests for deploying the Arena as a supervised service behind TLS.

## Deploy quickstart

```bash
# prerequisites: Python 3.12+, virtualenv, nginx, systemd, certbot
python -m venv /opt/falcom/.venv
source /opt/falcom/.venv/bin/activate
pip install -r arena/requirements.txt  # create if needed

# systemd + nginx
sudo cp ops/systemd/falcom-arena.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now falcom-arena
sudo cp ops/nginx/falcom-arena.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/falcom-arena.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d beta.falcomlabs.com
```

Set `ARENA_API_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` inside `/opt/falcom/env/.env` before starting the service.

The CLI helper lives at `arena/tools/arena_cli.py` (symlinked to `/usr/local/bin/arena` in production).
