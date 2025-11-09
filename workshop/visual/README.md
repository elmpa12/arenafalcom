# botscalpv3 visual replay

Visualização interativa para replays de backtests estilo TradingView, com KPIs ao vivo, comparação A/B e suporte a datasets volumosos.

## Estrutura

```
visual/
  backend/    # FastAPI + loaders + métricas
  frontend/   # lightweight-charts + UI de replay
```

## Pré-requisitos

- Python 3.10+
- Node.js 18+
- Make (opcional, recomendado)

## Backend (Linux)

```bash
cd visual/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ajustar conforme seus diretórios
export VISUAL_DATA_ROOT="$PWD/../data"
export API_PORT=8081
export CORS_ORIGIN="http://localhost:5173"

python app.py
```

Endpoints principais:

- `GET /healthz`
- `GET /api/backtests`
- `GET /api/backtests/{id}/meta`
- `GET /api/backtests/{id}/frames?offset=0&limit=2000`
- `GET /api/backtests/{id}/trades`
- `WS /ws/replay/{id}` (ativar com `USE_WS=true`)

## Frontend

Servindo estático (sem build):

```bash
cd visual/frontend
python -m http.server 5173
```

Opcional com Vite:

```bash
cd visual/frontend
npm install
npm run dev
```

## Make targets

Na raiz do projeto:

```bash
make visual-dev    # backend + frontend em modo desenvolvimento
make visual-build  # build frontend e serve estático pelo backend
```

## Datasets

Organize `VISUAL_DATA_ROOT` com um diretório por execução, contendo pelo menos (o repositório já traz `visual/data/demo_a` e `visual/data/demo_b` como exemplos):

```
run-id/
  meta.json           # metadados + KPIs globais (opcional)
  frames.jsonl        # um Frame por linha
  trades.jsonl        # lista de trades fechados
```

Suportados: `.jsonl`, `.csv`, `.parquet`. Exemplo de linha em `frames.jsonl`:

```json
{"bar":{"t":"2025-01-01T00:00:00Z","o":58200,"h":58210,"l":58190,"c":58205,"v":1234},"signals":[],"trades_open":[],"trades_closed":[],"equity":100000}
```

## Testes e lint

```bash
cd visual/backend
source .venv/bin/activate
pytest --cov=.
ruff check .
black --check .
```

## Configuração via `.env`

Copie `.env.example` para `.env` na pasta `visual/backend` e ajuste:

- `VISUAL_DATA_ROOT`
- `API_PORT`
- `CORS_ORIGIN`
- `USE_WS`

## Hotkeys

- `Space`: play/pause
- `→`: avança timeline
- `←`: retrocede timeline
- `Shift+→`: próximo trade
- `Shift+←`: trade anterior

## Produção

1. Gere export JSONL dos backtests.
2. Execute `make visual-build`.
3. Inicie `python visual/backend/app.py` com `USE_WS=false` para servir frontend estático.
