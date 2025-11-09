## Objetivo
Fornecer instruções curtas e práticas para agentes de codificação (Copilot / AI) que vão trabalhar neste repositório frontend do ScalpTV.

## Contrato rápido (inputs/outputs)
- Input principal: JSON via endpoints HTTP expostos sob `/api` (candles, methods, signals, fx).
- Output esperado pela UI: gráficos de candles + markers no `index.html` usando LightweightCharts.
- Erros: o frontend espera JSON; a função `safeFetch` faz fallback e log quando recebe texto não-JSON.

## Arquitetura (rápido)
- Frontend estático: `index.html` (UI), `app.js` (principal) e arquivos em `static/` (fallbacks/variações).
- Charts: usa LightweightCharts (CDN ou local em `static/vendor/lightweight-charts.standalone.production.js`).
- Fluxo de dados: `app.js` faz fetch em `/api/candles`, `/api/methods`, `/api/signals`, `/api/fx` e atualiza charts/localStorage.
- Avaliação local de trades: quando o backend não retorna exits/pnl, o frontend executa `evaluateOutcomes` para simular SL/TP/TO.

## Padrões e convenções do projeto (importantes)
- Tempo: o frontend lida com timestamps em segundos e em milissegundos; há várias conversões (ver `mapCandleRaw`, `toSec`). Tenha cuidado ao alterar unidades.
- API shapes:
  - `/api/candles` -> { candles: [{ time, open, high, low, close, volume }, ...] }
  - `/api/methods` -> lista ou objeto por timeframes; cada método pode ter `id`, `label`, `config`, `params`, `n_trades_48h`, `score`.
  - `/api/signals` pode retornar `executions` (preferido) ou `signals`; o cliente aceita ambos. Veja `refreshSignalsAndOutcomes`.
- Config central: no topo de `app.js` há `const API = '/api'` — alterar se o backend estiver em outra origem (ex.: `http://localhost:8080/api`).
- Cache/persistência: localStorage keys: `SCALP_EXEC` (exec settings) e `SCALPTV_<symbol>_<tf>_<viewType>_...` (cache de candles).
- Polling: candles = POLL_MS_CANDLES (1.5s), sinais = POLL_MS_SIGNALS (5s). Limites e caps: `MAX_MARKERS = 400`.

## Trechos úteis encontrados (exemplos a citar)
- Mudar origem da API: `app.js` (linha superior) — "const API = \"/api\";" — documentado no próprio arquivo.
- Relevância de `evaluateOutcomes`: lógica cliente para simular outcomes se backend não fornecer `exit_time/exit_price`.
- Como a UI preenche métodos: `fetchMethods()` → `fillMethodsDropdown(list)` — aceita `resp.timeframes[state.tf]` ou lista direta.

## Fluxos de trabalho de desenvolvimento (o que é seguro fazer e como testar)
- Preview rápido (sem backend): sirva a pasta `frontend` com um servidor estático e abra `index.html`.
  - Exemplo:
    ```bash
    cd frontend
    python3 -m http.server 8000
    # abrir http://localhost:8000/index.html
    ```
- Para desenvolvimento integrado com backend: mantenha o frontend e backend na mesma origem ou atualize `const API` no topo de `app.js`.
- Não existem scripts de build ou package.json detectados — alterações JS/CSS podem ser testadas recarregando a página.

## Regras específicas para agentes AI (práticas)
- Leia primeiro `index.html`, `app.js` (raiz) e `static/app.js` — esses contêm o comportamento e as heurísticas principais.
- Preserve as conversões de tempo (ms <-> s). Se adicionar código que manipula timestamps, escreva testes simples ou adicione comentários com a unidade esperada.
- Ao modificar a lógica de trade (SL/TP/TO), verifique `PROG_TP_DEFAULTS`, `ALLOW_OVERLAP`, `HOLD_SCALE` e o uso de `buildATRMap` — pequenas mudanças afetam visualização e cálculo de P/L.
- Ao adicionar chamadas ao backend, respeite os parâmetros já usados: symbol, tf, method/method_id, hours, limit, since.

## Onde procurar exemplos no código
- UI e binding: `index.html` + `/app.js` (root) → criação de charts, botões e controles (ex.: `ensureExecControlsUI`).
- Lógica de dados: `static/app.js` e root `app.js` — contém fetch helpers, `safeFetch`, `fetchCandles`, `fetchSignalsFor`, `evaluateOutcomes`.
- Estilos: `style.css` (root) — tema escuro e classes de pills/controls.

## Perguntas que eu (agente) devo fazer antes de mudanças maiores
- Existe um backend separado? Se sim, qual a origem padrão (mesma origem ou porta diferente)?
- Há testes automatizados ou CI que devemos manter compatíveis ao alterar a API cliente? (Não detectados neste repositório frontend.)

Se algo estiver impreciso ou faltar referência (por exemplo: rota do backend em outro repositório), me diga onde está o backend ou como você normalmente roda o sistema que integra frontend/backend — eu ajusto o arquivo conforme necessário.

---
Por favor, revise estas instruções e me diga se quer que eu inclua: comandos para executar um backend local, exemplos de payloads JSON reais ou testes automatizados básicos para as funções de normalização/avaliação.
