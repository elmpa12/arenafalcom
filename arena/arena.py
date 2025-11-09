#!/usr/bin/env python3
"""
Falcom Arena — Sala de debate em tempo real entre GPT (OpenAI) e Claude (Anthropic)

✅ Recursos
- Painel web em tempo real (Flask + Socket.IO) para ver GPT e Claude conversando.
- Botões: Start/Stop, Reset Memory, Inject (mensagem do mentor), Toggle Sandbox/Live.
- Logs em /opt/falcom/arena/dialogue e memória persistente em /opt/falcom/arena/memory.
- Integração com OpenAI e Anthropic via SDKs oficiais (stream simplificado).
- Config em /opt/falcom/arena/config.json (mode, max_rounds, topic, allow_exec).
- Sem dependência de arquivos externos (HTML/JS embutidos na rota /).

📦 Instalar deps (recomendado em venv):
  pip install flask flask-socketio eventlet openai anthropic python-dotenv

▶ Rodar:
  export FALCOM_HOME=/opt/falcom
  python /opt/falcom/arena/arena.py
  # Acesse: http://<seu_ip>:5050

🔐 Chaves:
  - Lê .env em /opt/falcom/env/.env com OPENAI_API_KEY e ANTHROPIC_API_KEY

⚠ Execução de código:
  - allow_exec=true habilita execução de blocos ```bash``` e ```python``` emitidos pelos agentes.
  - Resultado aparece na coluna “Executor”, no latest.log e em logs/arena.log.
"""

import os
import json
import time
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path

try:
    import eventlet
    eventlet.monkey_patch()
except Exception as e:  # pragma: no cover - falha crítica deve logar
    raise RuntimeError(
        "Falcom Arena requires eventlet. Install it and ensure monkey_patch() succeeds before importing Flask."
    ) from e

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    import websocket  # type: ignore
except Exception:
    websocket = None

from flask import Flask, request, Response
from flask_socketio import SocketIO

from dotenv import dotenv_values

# =========================
# Caminhos e Config padrão
# =========================
FALCOM_HOME = Path(os.environ.get("FALCOM_HOME", "/opt/falcom"))
ARENA_DIR = FALCOM_HOME / "arena"
DIALOGUE_DIR = ARENA_DIR / "dialogue"
MEMORY_DIR = ARENA_DIR / "memory"
ENV_FILE = FALCOM_HOME / "env" / ".env"
CONFIG_FILE = ARENA_DIR / "config.json"
LOG_GPT = FALCOM_HOME / "logs" / "gpt.log"
LOG_CLAUDE = FALCOM_HOME / "logs" / "claude.log"
LOG_ARENA = FALCOM_HOME / "logs" / "arena.log"
SHARED_CTX = ARENA_DIR / "shared_context.txt"
LOG_PROD_METRICS = FALCOM_HOME / "logs" / "production_metrics.log"

# =========================
# Config / Estado da Arena
# =========================
DEFAULT_CONFIG = {
    "mode": "sandbox",               # sandbox | live
    "max_rounds": 20,
    "topic": "Falcom — colaboração GPT x Claude",
    "allow_exec": False,
    "gpt_model": "gpt-5",
    "claude_model": "claude-3-7-sonnet",
    "coder_model": "gpt-5-mini",
    "coder_enabled": True,
    "round_delay_s": 1.0,
    "supervised_only": False,
    "approval_mode": "full",        # read_only | auto | full
    "allow_global_access": True,
    "workspace_root": "/opt/falcom/workspace",
}

_config_lock = threading.Lock()
_state_lock = threading.Lock()

arena_running = False
stop_signal = False
sandbox_mode = True
supervised_mode = False
current_round = 0
manual_round_event = threading.Event()

# Memória curta dos agentes
memory_gpt = []     # lista de strings
memory_claude = []

# Métricas de produção / Binance
metrics_state: dict[str, dict] = {
    "rest": {},
    "ws": {},
    "report": {},
    "last_update": None,
}
_metrics_lock = threading.Lock()
_metrics_collector = None
APPROVAL_MODES = {"read_only", "auto", "full"}
API_TOKEN = os.environ.get("ARENA_API_TOKEN")

# =========================
# SDKs
# =========================
try:
    import openai
    from anthropic import Anthropic
except Exception:
    print("[WARN] Falha ao importar SDKs. Instale: pip install openai anthropic")
    openai = None
    Anthropic = None

# =========================
# Utilitários
# =========================
def ensure_dirs():
    for p in [FALCOM_HOME, ARENA_DIR, DIALOGUE_DIR, MEMORY_DIR, FALCOM_HOME/"logs"]:
        p.mkdir(parents=True, exist_ok=True)

def load_env_keys():
    """Carrega chaves do .env e injeta em os.environ."""
    if ENV_FILE.exists():
        env = dotenv_values(str(ENV_FILE))
        for k, v in env.items():
            if v is not None:
                os.environ.setdefault(k, v)
    else:
        print(f"[WARN] .env não encontrado em {ENV_FILE}")

def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                cfg.update(user_cfg or {})
        except Exception as e:
            print(f"[WARN] Falha ao ler config.json: {e}")
    return apply_approval_policy(cfg)

def save_config(cfg):
    with _config_lock:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


def apply_approval_policy(cfg: dict) -> dict:
    mode = cfg.get("approval_mode") or "full"
    mode = mode if mode in APPROVAL_MODES else "full"
    cfg["approval_mode"] = mode
    if mode == "read_only":
        cfg["allow_exec"] = False
        cfg["supervised_only"] = True
        cfg["allow_global_access"] = False
    elif mode == "auto":
        cfg.setdefault("allow_exec", True)
        cfg.setdefault("supervised_only", False)
        cfg.setdefault("allow_global_access", False)
    else:  # full
        cfg.setdefault("allow_exec", True)
        cfg.setdefault("supervised_only", False)
        cfg.setdefault("allow_global_access", True)
    return cfg


def _apply_payload_to_config(cfg: dict, payload: dict) -> dict:
    if not payload:
        return cfg
    if "rounds" in payload:
        try:
            cfg["max_rounds"] = max(1, int(payload["rounds"]))
        except (TypeError, ValueError):
            pass
    if "delay" in payload and payload["delay"] is not None:
        try:
            cfg["round_delay_s"] = max(0.0, float(payload["delay"]))
        except (TypeError, ValueError):
            pass
    if "gpt_model" in payload and payload["gpt_model"]:
        cfg["gpt_model"] = str(payload["gpt_model"])
    if "claude_model" in payload and payload["claude_model"]:
        cfg["claude_model"] = str(payload["claude_model"])
    if "coder_model" in payload and payload["coder_model"]:
        cfg["coder_model"] = str(payload["coder_model"])
    if "coder_enabled" in payload:
        cfg["coder_enabled"] = bool(payload["coder_enabled"])
    if "mode" in payload and payload["mode"] in {"sandbox", "live", "supervised"}:
        cfg["mode"] = payload["mode"]
    if "supervised_only" in payload:
        cfg["supervised_only"] = bool(payload["supervised_only"])
    if "allow_global_access" in payload:
        cfg["allow_global_access"] = bool(payload["allow_global_access"])
    if "workspace_root" in payload and payload["workspace_root"]:
        cfg["workspace_root"] = str(payload["workspace_root"])
    if "topic" in payload and payload["topic"]:
        cfg["topic"] = str(payload["topic"])
    return cfg


def broadcast_status(cfg: dict | None = None):
    cfg = cfg or load_config()
    socketio.emit("arena_status", current_status(cfg))


def current_status(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return {
        "running": arena_running,
        "mode": cfg.get("mode", "sandbox"),
        "sandbox": cfg.get("mode", "sandbox") == "sandbox",
        "supervised_only": bool(cfg.get("supervised_only", False)),
        "approval_mode": cfg.get("approval_mode", "full"),
        "allow_global_access": bool(cfg.get("allow_global_access", True)),
        "workspace_root": cfg.get("workspace_root", str(EXEC_WORKDIR)),
        "allow_exec": bool(cfg.get("allow_exec", False)),
        "models": {
            "gpt": cfg.get("gpt_model"),
            "claude": cfg.get("claude_model"),
            "coder": cfg.get("coder_model"),
        },
    }


def _launch_arena_thread():
    th = threading.Thread(target=run_debate_loop, args=(socketio,), daemon=True)
    th.start()
    return th


def _wait_for_stop(timeout: float = 10.0):
    waited = 0.0
    while arena_running and waited < timeout:
        time.sleep(0.1)
        waited += 0.1


def wait_with_manual_override(delay: float):
    if delay <= 0:
        manual_round_event.clear()
        return
    triggered = manual_round_event.wait(timeout=delay)
    if triggered:
        append_file(LOG_ARENA, f"[{ts()}] Mentor solicitou rodada manual")
    manual_round_event.clear()

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_file(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def write_file(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def read_file(path: Path, default: str = ""):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return default


# =========================
# API Auth Helper
# =========================
def require_api_token(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if API_TOKEN:
            provided = request.headers.get("X-Auth")
            if provided != API_TOKEN:
                return {"ok": False, "msg": "unauthorized"}, 401
        return func(*args, **kwargs)

    return wrapper


# =========================
# Métricas Binance / Logs
# =========================
def _update_metrics_state(kind: str, payload: dict):
    with _metrics_lock:
        metrics_state[kind] = payload
        metrics_state["last_update"] = ts()
        rest = metrics_state.get("rest") or {}
        ws = metrics_state.get("ws") or {}
        metrics_state["report"] = _compute_performance_snapshot(rest, ws)


def _log_metrics(entry: dict):
    try:
        LOG_PROD_METRICS.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PROD_METRICS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - apenas log
        append_file(LOG_ARENA, f"[{ts()}] WARN metrics log: {exc}")


def _compute_performance_snapshot(rest_metrics: dict, ws_metrics: dict) -> dict:
    if not rest_metrics:
        return metrics_state.get("report", {})

    try:
        change_pct = float(rest_metrics.get("price_change_percent", 0.0))
    except (TypeError, ValueError):  # pragma: no cover - dados ruins
        change_pct = 0.0
    try:
        last_price = float(rest_metrics.get("last_price") or rest_metrics.get("close", 0.0))
    except (TypeError, ValueError):
        last_price = 0.0
    try:
        quote_volume = float(rest_metrics.get("quote_volume", 0.0))
    except (TypeError, ValueError):
        quote_volume = 0.0

    notional = 100_000  # base fictícia para estimativas
    pnl = notional * (change_pct / 100.0)
    sharpe = round((change_pct / 100.0) * 3.0, 4)
    max_drawdown = round(abs(change_pct) * 0.35, 4)
    roi = round(change_pct, 4)

    snapshot = {
        "price": last_price,
        "price_change_pct": change_pct,
        "quote_volume": quote_volume,
        "est_pnl_usd": pnl,
        "est_sharpe": sharpe,
        "est_mdd_pct": max_drawdown,
        "est_roi_pct": roi,
        "last_trade_ts": ws_metrics.get("last_trade_ts") if ws_metrics else None,
    }
    return snapshot


class BinanceMetricsCollector:
    """Coleta métricas simples via REST + WebSocket para exibir no painel."""

    REST_ENDPOINT = "https://api.binance.com/api/v3/ticker/24hr"
    WS_ENDPOINT = "wss://stream.binance.com:9443/ws/{symbol}@trade"

    def __init__(self, symbol: str = "BTCUSDT", rest_interval: float = 30.0):
        self.symbol = symbol.upper()
        self.rest_interval = max(5.0, rest_interval)
        self._stop = threading.Event()
        self._rest_thread = threading.Thread(target=self._rest_loop, name="binance-rest", daemon=True)
        self._ws_thread = None
        self._ws_ready = threading.Event()

    def start(self):
        if not self._rest_thread.is_alive():
            self._stop.clear()
            self._rest_thread = threading.Thread(target=self._rest_loop, name="binance-rest", daemon=True)
            self._rest_thread.start()
        if websocket is not None and (self._ws_thread is None or not self._ws_thread.is_alive()):
            self._ws_thread = threading.Thread(target=self._ws_loop, name="binance-ws", daemon=True)
            self._ws_thread.start()
        elif websocket is None:
            append_file(LOG_ARENA, f"[{ts()}] WARN metrics: websocket-client ausente; monitoração WS desabilitada")

    def stop(self):
        self._stop.set()
        manual_round_event.set()  # desbloqueia waits caso necessário

    def _rest_loop(self):
        if requests is None:
            append_file(LOG_ARENA, f"[{ts()}] WARN metrics: requests não instalado — REST desabilitado")
            return
        while not self._stop.is_set():
            try:
                resp = requests.get(self.REST_ENDPOINT, params={"symbol": self.symbol}, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                metrics = {
                    "symbol": self.symbol,
                    "price_change_percent": data.get("priceChangePercent"),
                    "last_price": data.get("lastPrice"),
                    "quote_volume": data.get("quoteVolume"),
                    "open_price": data.get("openPrice"),
                    "high_price": data.get("highPrice"),
                    "low_price": data.get("lowPrice"),
                }
                _update_metrics_state("rest", metrics)
                _log_metrics({"ts": ts(), "rest": metrics})
            except Exception as exc:
                append_file(LOG_ARENA, f"[{ts()}] WARN metrics REST: {exc}")
            finally:
                self._stop.wait(self.rest_interval)

    def _ws_loop(self):
        if websocket is None:
            return
        url = self.WS_ENDPOINT.format(symbol=self.symbol.lower())
        while not self._stop.is_set():
            ws = None
            try:
                ws = websocket.create_connection(url, timeout=10)
                while not self._stop.is_set():
                    raw = ws.recv()
                    if not raw:
                        continue
                    data = json.loads(raw)
                    payload = {
                        "price": data.get("p"),
                        "qty": data.get("q"),
                        "trade_id": data.get("t"),
                        "last_trade_ts": ts(),
                    }
                    _update_metrics_state("ws", payload)
            except Exception as exc:
                append_file(LOG_ARENA, f"[{ts()}] WARN metrics WS: {exc}")
                time.sleep(3)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass


def ensure_metrics_collector(symbol: str = "BTCUSDT"):
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = BinanceMetricsCollector(symbol=symbol)
        _metrics_collector.start()

# ===============
# Model wrappers
# ===============
def call_openai_chat(model: str, system: str, messages: list[str]) -> str:
    if openai is None:
        return "[ERRO] SDK OpenAI não instalado."
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "[ERRO] OPENAI_API_KEY ausente em /opt/falcom/env/.env"

    client = openai.OpenAI(api_key=api_key)
    chat_msgs = [{"role": "system", "content": system}] + [
        {"role": "user", "content": m} for m in messages
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=chat_msgs,
            temperature=0.8,
            max_tokens=1200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI ERRO] {e}"

def call_anthropic_chat(model: str, system: str, messages: list[str]) -> str:
    if Anthropic is None:
        return "[ERRO] SDK Anthropic não instalado."
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERRO] ANTHROPIC_API_KEY ausente em /opt/falcom/env/.env"

    client = Anthropic(api_key=api_key)
    msg_list = [{"role": "user", "content": m} for m in messages]
    try:
        resp = client.messages.create(
            model=model,
            system=system,
            messages=msg_list,
            max_tokens=1200,
            temperature=0.8,
        )
        parts = []
        for blk in resp.content:
            if hasattr(blk, "text"):
                parts.append(blk.text)
            elif isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        return ("\n".join(parts)).strip() or "[vazio]"
    except Exception as e:
        return f"[Anthropic ERRO] {e}"


def _call_agent_once(agent: str, mentor_text: str, cfg: dict) -> str:
    topic = cfg.get("topic", DEFAULT_CONFIG["topic"])
    agent = agent.lower()
    if agent == "garapa":
        return call_anthropic_chat(
            model=cfg.get("claude_model", DEFAULT_CONFIG["claude_model"]),
            system=system_prompt_claude(topic),
            messages=[mentor_text],
        )
    if agent == "alfred":
        coder_model = cfg.get("coder_model", DEFAULT_CONFIG["coder_model"])
        if str(coder_model).lower().startswith("claude"):
            return call_anthropic_chat(
                model=coder_model,
                system=system_prompt_coder(topic),
                messages=[mentor_text],
            )
        return call_openai_chat(
            model=coder_model,
            system=system_prompt_coder(topic),
            messages=[mentor_text],
        )
    # default Ernest
    return call_openai_chat(
        model=cfg.get("gpt_model", DEFAULT_CONFIG["gpt_model"]),
        system=system_prompt_gpt(topic),
        messages=[mentor_text],
    )


def _log_say_event(agent: str, mentor_text: str, reply: str, dryrun: bool):
    tag = agent.upper()
    dry = " dryrun" if dryrun else ""
    mentor_line = f"[{ts()}] [MENTOR][SAY:{tag}{dry}] {mentor_text}"
    reply_line = f"[{ts()}] {tag}: {reply}"
    append_file(DIALOGUE_DIR / "latest.log", mentor_line)
    append_file(DIALOGUE_DIR / "latest.log", reply_line)
    append_file(LOG_ARENA, mentor_line)
    append_file(LOG_ARENA, reply_line)
    socketio.emit("msg", {"agent": agent, "round": 0, "text": reply})

# ==================
# Motor de Debate
# ==================

# --- Executor (sandbox de código) -------------------------------------------
# Executa blocos ```bash``` e ```python``` quando allow_exec=true. Resultados aparecem no feed Executor.
import subprocess
EXEC_WORKDIR = (FALCOM_HOME / "workspace").resolve()
POLICY_DENY_MSG = "[policy] Global filesystem access bloqueado pelo Mentor. Ajuste o modo ou libere acesso externo."


def _workspace_root_from_cfg(cfg: dict) -> Path:
    target = cfg.get("workspace_root") or str(EXEC_WORKDIR)
    try:
        return Path(target).resolve()
    except Exception:
        return EXEC_WORKDIR


def _command_violates_workspace_scope(cmd: str, cfg: dict) -> bool:
    if cfg.get("allow_global_access", True):
        return False
    workspace_root = _workspace_root_from_cfg(cfg)
    abs_paths = re.findall(r"(/[^\s'\"`]+)", cmd)
    for raw in abs_paths:
        try:
            target = Path(raw).resolve(strict=False)
        except Exception:
            continue
        try:
            target.relative_to(workspace_root)
        except ValueError:
            return True
    if "../" in cmd or cmd.strip().startswith(".."):
        return True
    if "~" in cmd:
        return True
    return False

def _extract_action_blocks(text: str) -> list:
    if not text:
        return []
    import re
    blocks = []
    for lang in ("bash", "python"):
        for m in re.finditer(rf"```{lang}(.+?)```", text, flags=re.DOTALL | re.IGNORECASE):
            code = m.group(1).strip()
            if code:
                blocks.append({"lang": lang, "code": code})
    return blocks

def _run_bash(cmd: str, cfg: dict) -> tuple[int, str]:
    if _command_violates_workspace_scope(cmd, cfg):
        append_file(LOG_ARENA, f"[{ts()}] BLOQUEADO (bash) → {cmd}")
        return 1, POLICY_DENY_MSG
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(EXEC_WORKDIR), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        return p.returncode, p.stdout
    except Exception as e:
        return 1, f"[executor] erro bash: {e}"

def _run_python(code: str, cfg: dict) -> tuple[int, str]:
    if _command_violates_workspace_scope(code, cfg):
        append_file(LOG_ARENA, f"[{ts()}] BLOQUEADO (python)")
        return 1, POLICY_DENY_MSG
    try:
        script_path = ARENA_DIR / "tools" / "_tmp_exec.py"
        write_file(script_path, code)
        p = subprocess.run(["python3", str(script_path)], cwd=str(EXEC_WORKDIR), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        return p.returncode, p.stdout
    except Exception as e:
        return 1, f"[executor] erro python: {e}"

def try_execute_actions(text: str, socketio: SocketIO, who: str, round_no: int):
    cfg = load_config()
    if cfg.get("supervised_only", False) or not cfg.get("allow_exec", False):
        if cfg.get("supervised_only", False):
            append_file(LOG_ARENA, f"[{ts()}] Supervised mode ativo — execuções ignoradas ({who})")
        return
    actions = _extract_action_blocks(text)
    if not actions:
        return
    append_file(LOG_ARENA, f"[{ts()}] AÇÕES detectadas por {who} (R{round_no}): {len(actions)} bloco(s)")
    for act in actions:
        rc, out = (
            _run_bash(act["code"], cfg)
            if act["lang"].lower() == "bash"
            else _run_python(act["code"], cfg)
        )
        msg = f"[EXECUTOR][R{round_no}][{act['lang']}] rc={rc}\n{out}"
        append_file(DIALOGUE_DIR / "latest.log", f"[{ts()}] EXECUTOR: {msg}")
        append_file(LOG_ARENA, f"[{ts()}] {msg}")
        socketio.emit("msg", {"agent": "executor", "round": round_no, "text": msg})

def system_prompt_gpt(topic: str) -> str:
    return (
        "Você é o Engenheiro-chefe do projeto Falcom. Seja pragmático, específico e orientado a execução. "
        "Produza planos acionáveis, difira controvérsias com propostas de hipóteses testáveis e crie patches claros quando necessário. "
        "Use formato: \n[ANÁLISE]\n[PLANO]\n[AÇÃO(opcional)]\n"
    ) + f"\nTópico atual: {topic}"

def system_prompt_claude(topic: str) -> str:
    return (
        "Você é o Arquiteto/Crítico do projeto Falcom. Questione suposições, proponha alternativas, avalie riscos e trade-offs. "
        "Apresente revisões enxutas e critérios de aceitação. Evite prolixidade. "
        "Use formato: \n[DIAGNÓSTICO]\n[REVISÕES]\n[DECISÕES]\n"
    ) + f"\nTópico atual: {topic}"

def system_prompt_coder(topic: str) -> str:
    return (
        "Você é o Programador de Código do Falcom. Gere PATCHES mínimos e executáveis. "
        "Responda APENAS com blocos de código. Use ```bash``` para shell (git, sed, patch, testes) ou ```python``` para scripts. "
        "Sem prosa; se não houver ação clara, emita um bloco ```bash``` com TODO."
    ) + f"\nTópico atual: {topic}"

def run_debate_loop(socketio: SocketIO):
    global arena_running, stop_signal, current_round, sandbox_mode, supervised_mode

    cfg = load_config()
    topic = cfg.get("topic", DEFAULT_CONFIG["topic"])
    max_rounds = int(cfg.get("max_rounds", 10))
    round_delay = float(cfg.get("round_delay_s", 1.0))
    gpt_model = cfg.get("gpt_model", DEFAULT_CONFIG["gpt_model"])
    claude_model = cfg.get("claude_model", DEFAULT_CONFIG["claude_model"])
    coder_model = cfg.get("coder_model", DEFAULT_CONFIG.get("coder_model", "gpt-5-mini"))
    coder_enabled = bool(cfg.get("coder_enabled", True))
    sandbox_mode = (cfg.get("mode", "sandbox") == "sandbox")
    supervised_mode = bool(cfg.get("supervised_only", False))

    arena_running = True
    stop_signal = False
    current_round = 0

    broadcast_status(cfg)
    append_file(LOG_ARENA, f"[{ts()}] START arena — mode={cfg.get('mode')} rounds={max_rounds}")

    shared_ctx = read_file(SHARED_CTX, "(shared_context vazio)")
    last_gpt = f"(Boot) Iniciando debate. Contexto:\n{shared_ctx}"
    last_claude = ""

    try:
        if (MEMORY_DIR/"gpt_mem.json").exists():
            memory = json.loads((MEMORY_DIR/"gpt_mem.json").read_text(encoding="utf-8"))
            if isinstance(memory, list):
                memory_gpt.extend(memory)
        if (MEMORY_DIR/"claude_mem.json").exists():
            memory = json.loads((MEMORY_DIR/"claude_mem.json").read_text(encoding="utf-8"))
            if isinstance(memory, list):
                memory_claude.extend(memory)
    except Exception as e:
        append_file(LOG_ARENA, f"[{ts()}] WARN memória: {e}")

    for rnd in range(1, max_rounds + 1):
        with _state_lock:
            if stop_signal:
                break
            current_round = rnd

        # 1) Claude responde o GPT
        claude_msg = call_anthropic_chat(
            model=claude_model,
            system=system_prompt_claude(topic),
            messages=[last_gpt]
        )
        last_claude = claude_msg
        append_file(LOG_CLAUDE, f"[{ts()}][R{rnd}] {claude_msg}")
        append_file(DIALOGUE_DIR / "latest.log", f"[{ts()}] CLAUDE: {claude_msg}")
        socketio.emit("msg", {"agent": "claude", "round": rnd, "text": claude_msg})
        try_execute_actions(claude_msg, socketio, who="claude", round_no=rnd)

        wait_with_manual_override(round_delay)
        if stop_signal:
            break

        # 2) GPT responde o Claude
        gpt_msg = call_openai_chat(
            model=gpt_model,
            system=system_prompt_gpt(topic),
            messages=[claude_msg]
        )
        last_gpt = gpt_msg
        append_file(LOG_GPT, f"[{ts()}][R{rnd}] {gpt_msg}")
        append_file(DIALOGUE_DIR / "latest.log", f"[{ts()}] GPT: {gpt_msg}")
        socketio.emit("msg", {"agent": "gpt", "round": rnd, "text": gpt_msg})
        try_execute_actions(gpt_msg, socketio, who="gpt", round_no=rnd)

        # 3) Coder propõe patch minimal (opcional)
        if coder_enabled:
            if str(coder_model).startswith("claude"):
                coder_msg = call_anthropic_chat(
                    model=coder_model,
                    system=system_prompt_coder(topic),
                    messages=[gpt_msg, claude_msg]
                )
            else:
                coder_msg = call_openai_chat(
                    model=coder_model,
                    system=system_prompt_coder(topic),
                    messages=[gpt_msg, claude_msg]
                )
            append_file(DIALOGUE_DIR / "latest.log", f"[{ts()}] CODER: {coder_msg}")
            socketio.emit("msg", {"agent": "coder", "round": rnd, "text": coder_msg})
            try_execute_actions(coder_msg, socketio, who="coder", round_no=rnd)

        wait_with_manual_override(round_delay)
        if stop_signal:
            break

    arena_running = False
    broadcast_status(cfg)
    append_file(LOG_ARENA, f"[{ts()}] STOP arena (rounds={current_round})")

# ======================
# Servidor Flask + UI
# ======================
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

ensure_dirs()
ensure_metrics_collector()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Falcom Arena</title>
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js" crossorigin="anonymous"></script>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; margin:0; background:#0b1220; color:#e6eefc; }
    header { display:flex; justify-content:space-between; align-items:center; padding:14px 18px; background:#0f172a; border-bottom:1px solid #1f2a44; flex-wrap:wrap; gap:12px; }
    .tag { padding:4px 8px; border-radius:10px; background:#172554; margin-left:8px; font-size:12px }
    .tag.alert { background:#9f1239; }
    main { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; padding:12px; box-sizing:border-box; }
    .col { display:flex; flex-direction:column; border:1px solid #1f2a44; border-radius:14px; overflow:hidden; min-height:220px; }
    .col h2 { margin:0; padding:10px 12px; background:#0f172a; border-bottom:1px solid #1f2a44; font-size:14px; display:flex; justify-content:space-between; align-items:center; gap:6px; }
    .feed { flex:1; overflow:auto; padding:12px; display:flex; flex-direction:column; gap:10px; }
    .msg { padding:10px 12px; border-radius:10px; line-height:1.4; white-space:pre-wrap; }
    .gpt .msg { background:#0e2a17; border:1px solid #1f5131; }
    .claude .msg { background:#13233b; border:1px solid #254a87; }
    .coder .msg { background:#3b1f1f; border:1px solid:#7a2e2e; }
    .exec .msg { background:#1f2937; border:1px solid #374151; }
    button { background:#1d4ed8; color:white; border:none; padding:8px 12px; border-radius:10px; cursor:pointer; transition:opacity .2s; }
    button:disabled { opacity:0.5; cursor:not-allowed; }
    button.stop { background:#9f1239; }
    button.secondary { background:#374151; }
    .mentor-panel textarea { background:#0f172a; color:#e6eefc; border:1px solid #1f2a44; padding:8px 10px; border-radius:8px; width:100%; min-height:80px; }
    .mentor-actions { display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin-top:10px; }
    .panel-body { padding:12px; display:flex; flex-direction:column; gap:10px; }
    .metrics-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:8px; }
    .metrics-card { background:#111827; border:1px solid #1f2a44; border-radius:10px; padding:8px 10px; font-size:13px; line-height:1.5; }
    .status-line { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
    select, input { background:#0f172a; color:#e6eefc; border:1px solid #1f2a44; padding:8px 10px; border-radius:8px; }
    .top-controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  </style>
</head>
<body>
  <header>
    <div>
      <strong>Falcom Arena</strong>
      <span id="badge" class="tag">offline</span>
      <span class="tag" id="mode">sandbox</span>
      <span class="tag" id="supervised">analysis</span>
    </div>
    <div class="top-controls">
      <label>Rounds: <input id="rounds" type="number" min="1" value="__MAX_ROUNDS__" style="width:70px"></label>
      <label>Delay(s): <input id="delay" type="number" min="0" step="0.1" value="__ROUND_DELAY__" style="width:80px"></label>
      <select id="gpt_model">
        <option value="gpt-5">gpt-5</option>
        <option value="gpt-5-mini">gpt-5-mini</option>
        <option value="gpt-4o-mini">gpt-4o-mini</option>
      </select>
      <select id="claude_model">
        <option value="claude-3-7-sonnet">claude-3-7-sonnet</option>
        <option value="claude-3-5-sonnet-latest">claude-3-5-sonnet-latest</option>
        <option value="claude-3-5-haiku-latest">claude-3-5-haiku-latest</option>
      </select>
      <select id="coder_model">
        <option value="gpt-5-mini">coder: gpt-5-mini</option>
        <option value="gpt-4.1-mini">coder: gpt-4.1-mini</option>
        <option value="claude-3-5-haiku-latest">coder: claude-3-5-haiku-latest</option>
      </select>
      <button id="start">Start</button>
      <button id="stop" class="stop">Stop</button>
      <button id="toggle" class="secondary">Toggle Sandbox/Live</button>
      <button id="reset" class="secondary">Reset Memory</button>
      <label>Approval:
        <select id="approval-mode">
          <option value="read_only">Read Only</option>
          <option value="auto">Auto</option>
          <option value="full" selected>Full Access</option>
        </select>
      </label>
      <label>Filesystem:
        <select id="workspace-access">
          <option value="workspace">Workspace Only</option>
          <option value="global">Allow Global</option>
        </select>
      </label>
    </div>
  </header>
  <main>
    <div class="col claude">
      <h2>GARAPA <small>(Claude 3.7 Sonnet)</small></h2>
      <div id="feed-claude" class="feed"></div>
    </div>
    <div class="col gpt">
      <h2>ERNEST <small>(GPT-5)</small></h2>
      <div id="feed-gpt" class="feed"></div>
    </div>
    <div class="col coder">
      <h2>ALFRED <small>(Coder)</small></h2>
      <div id="feed-coder" class="feed"></div>
    </div>
    <div class="col exec" style="grid-column: span 3;">
      <h2>Executor <span class="tag">ações</span></h2>
      <div id="feed-exec" class="feed"></div>
    </div>
    <div class="col metrics-panel">
      <h2>Produção supervisionada <span class="tag" id="metrics-status">aguardando</span></h2>
      <div class="panel-body">
        <div class="metrics-grid" id="metrics-grid">
          <div class="metrics-card">Sem dados ainda.</div>
        </div>
        <div class="status-line">
          <button id="refresh-metrics" class="secondary">Atualizar métricas</button>
          <button id="generate-report" class="secondary">Gerar relatório</button>
        </div>
      </div>
    </div>
    <div class="col mentor-panel" style="grid-column: span 2;">
      <h2>Mentor Control <span class="tag alert">Supervisão Humana</span></h2>
      <div class="panel-body">
        <textarea id="mentor" placeholder="Sua mensagem de mentor — será injetada na próxima rodada"></textarea>
        <div class="mentor-actions">
          <button id="inject">Injetar</button>
          <button id="mentor-pause" class="stop">Pausar</button>
          <button id="mentor-restart">Reiniciar</button>
          <button id="mentor-revert" class="secondary">Reverter Config</button>
          <button id="mentor-supervised" class="secondary">Modo Supervisão</button>
          <button id="mentor-manual">Rodada Manual</button>
        </div>
      </div>
    </div>
  </main>
<script>
  const socket = io();
  const badge = document.getElementById('badge');
  const mode = document.getElementById('mode');
  const supervised = document.getElementById('supervised');
  const feedClaude = document.getElementById('feed-claude');
  const feedGpt = document.getElementById('feed-gpt');
  const feedCoder = document.getElementById('feed-coder');
  const feedExec = document.getElementById('feed-exec');
  const metricsGrid = document.getElementById('metrics-grid');
  const metricsStatus = document.getElementById('metrics-status');
  const approvalSelect = document.getElementById('approval-mode');
  const workspaceAccess = document.getElementById('workspace-access');

  const AGENT_PREFIX = {
    'claude': '[GARAPA]',
    'gpt': '[ERNEST]',
    'coder': '[ALFRED]',
    'executor': '[EXECUTOR]'
  };

  function addMsg(col, text, round) {
    const wrap = document.createElement('div');
    wrap.className = 'msg';
    const prefix = AGENT_PREFIX[col] || '[SISTEMA]';
    wrap.innerText = `${prefix} [R${round}] ${text}`;
    const target = col === 'claude' ? feedClaude : (col === 'gpt' ? feedGpt : (col === 'coder' ? feedCoder : feedExec));
    target.appendChild(wrap);
    target.scrollTop = 1e9;
  }

  socket.on('connect', () => { badge.textContent = 'online'; });
  socket.on('disconnect', () => { badge.textContent = 'offline'; });

  socket.on('arena_status', (p) => {
    mode.textContent = p.mode || (p.sandbox ? 'sandbox' : 'live');
    supervised.textContent = p.supervised_only ? 'supervision ON' : 'analysis';
    supervised.classList.toggle('alert', !!p.supervised_only);
    if (p.approval_mode && approvalSelect) {
      approvalSelect.value = p.approval_mode;
    }
    if (workspaceAccess) {
      workspaceAccess.value = p.allow_global_access ? 'global' : 'workspace';
    }
  });

  socket.on('msg', (p) => {
    addMsg(p.agent, p.text, p.round);
  });

  document.getElementById('start').onclick = () => {
    fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
      rounds: Number(document.getElementById('rounds').value || 10),
      delay: Number(document.getElementById('delay').value || 1),
      gpt_model: document.getElementById('gpt_model').value,
      claude_model: document.getElementById('claude_model').value,
      coder_model: document.getElementById('coder_model').value
    })});
  };
  document.getElementById('stop').onclick = () => fetch('/stop', {method:'POST'});
  document.getElementById('toggle').onclick = () => fetch('/toggle', {method:'POST'});
  document.getElementById('reset').onclick = () => fetch('/reset', {method:'POST'});
  document.getElementById('inject').onclick = () => {
    const txt = document.getElementById('mentor').value.trim();
    if (!txt) return;
    fetch('/inject', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text: txt})});
    document.getElementById('mentor').value='';
  };
  document.getElementById('mentor-pause').onclick = () => fetch('/stop', {method:'POST'});
  document.getElementById('mentor-restart').onclick = () => {
    fetch('/restart', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
      rounds: Number(document.getElementById('rounds').value || 10),
      delay: Number(document.getElementById('delay').value || 1),
      gpt_model: document.getElementById('gpt_model').value,
      claude_model: document.getElementById('claude_model').value,
      coder_model: document.getElementById('coder_model').value
    })});
  };
  document.getElementById('mentor-revert').onclick = () => fetch('/revert', {method:'POST'});
  document.getElementById('mentor-supervised').onclick = () => fetch('/supervised', {method:'POST'});
  document.getElementById('mentor-manual').onclick = () => fetch('/manual_round', {method:'POST'});
  if (approvalSelect) {
    approvalSelect.onchange = () => {
      fetch('/approval_mode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mode: approvalSelect.value})});
    };
  }
  if (workspaceAccess) {
    workspaceAccess.onchange = () => {
      const allowGlobal = workspaceAccess.value === 'global';
      fetch('/workspace_access', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({allow_global: allowGlobal})});
    };
  }

  function renderMetrics(cards) {
    metricsGrid.innerHTML = '';
    if (!cards || !Object.keys(cards).length) {
      metricsGrid.innerHTML = '<div class="metrics-card">Sem dados.</div>';
      return;
    }
    const entries = [
      { label: 'Preço', value: cards.price ? `$${Number(cards.price).toFixed(2)}` : '—' },
      { label: 'Δ 24h', value: cards.price_change_pct ? `${Number(cards.price_change_pct).toFixed(2)}%` : '—' },
      { label: 'Quote Vol', value: cards.quote_volume ? Number(cards.quote_volume).toFixed(0) : '—' },
      { label: 'PnL Est.', value: cards.est_pnl_usd ? `$${Number(cards.est_pnl_usd).toFixed(2)}` : '—' },
      { label: 'Sharpe Est.', value: cards.est_sharpe ?? '—' },
      { label: 'MDD Est.', value: cards.est_mdd_pct ? `${Number(cards.est_mdd_pct).toFixed(2)}%` : '—' },
      { label: 'ROI Est.', value: cards.est_roi_pct ? `${Number(cards.est_roi_pct).toFixed(2)}%` : '—' },
    ];
    entries.forEach(item => {
      const div = document.createElement('div');
      div.className = 'metrics-card';
      div.innerHTML = `<strong>${item.label}</strong><br>${item.value}`;
      metricsGrid.appendChild(div);
    });
  }

  function fetchMetrics() {
    fetch('/metrics').then(r => r.json()).then(data => {
      if (!data.ok) return;
      const report = data.metrics.report || {};
      metricsStatus.textContent = data.metrics.last_update || 'sem dados';
      renderMetrics(report);
    }).catch(() => {
      metricsGrid.innerHTML = '<div class="metrics-card">Falha ao carregar.</div>';
    });
  }

  document.getElementById('refresh-metrics').onclick = fetchMetrics;
  document.getElementById('generate-report').onclick = () => {
    fetch('/reports/generate', {method:'POST'})
      .then(() => fetchMetrics());
  };

  fetchMetrics();
  setInterval(fetchMetrics, 30000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    html = (
        HTML_TEMPLATE
        .replace("__MAX_ROUNDS__", str(DEFAULT_CONFIG["max_rounds"]))
        .replace("__ROUND_DELAY__", str(DEFAULT_CONFIG["round_delay_s"]))
    )
    return Response(html, mimetype="text/html")


@app.post("/start")
def start_arena():
    global arena_running, stop_signal
    payload = request.get_json(force=True, silent=True) or {}

    cfg = _apply_payload_to_config(load_config(), payload)
    cfg = apply_approval_policy(cfg)
    save_config(cfg)

    if arena_running:
        return {"ok": False, "msg": "Arena já está rodando"}, 400

    stop_signal = False
    _launch_arena_thread()
    broadcast_status(cfg)
    return {"ok": True, "config": cfg}


@app.post("/stop")
def stop_arena():
    global stop_signal
    stop_signal = True
    manual_round_event.set()
    _wait_for_stop(5)
    broadcast_status()
    return {"ok": True}


@app.post("/toggle")
def toggle_mode():
    cfg = load_config()
    cfg["mode"] = "live" if cfg.get("mode", "sandbox") == "sandbox" else "sandbox"
    save_config(cfg)
    broadcast_status(cfg)
    append_file(LOG_ARENA, f"[{ts()}] Toggle mode → {cfg['mode']}")
    return {"ok": True, "mode": cfg["mode"]}


@app.post("/toggle_exec")
@require_api_token
def toggle_exec():
    payload = request.get_json(force=True, silent=True) or {}
    cfg = load_config()
    if "enabled" in payload:
        cfg["allow_exec"] = bool(payload["enabled"])
    else:
        cfg["allow_exec"] = not cfg.get("allow_exec", False)
    cfg = apply_approval_policy(cfg)
    save_config(cfg)
    broadcast_status(cfg)
    append_file(LOG_ARENA, f"[{ts()}] Toggle exec → {cfg['allow_exec']}")
    return {"ok": True, "allow_exec": cfg["allow_exec"]}


@app.post("/reset")
def reset_memory():
    try:
        (MEMORY_DIR/"gpt_mem.json").write_text("[]", encoding="utf-8")
        (MEMORY_DIR/"claude_mem.json").write_text("[]", encoding="utf-8")
        (DIALOGUE_DIR/"latest.log").write_text("", encoding="utf-8")
        append_file(LOG_ARENA, f"[{ts()}] Reset memory")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


@app.post("/inject")
def inject_message():
    payload = request.get_json(force=True, silent=True) or {}
    txt = (payload.get("text") or "").strip()
    if not txt:
        return {"ok": False, "msg": "texto vazio"}, 400
    prev = read_file(SHARED_CTX, "")
    new = prev + ("\n\n[MENTOR] " + txt)
    write_file(SHARED_CTX, new)
    append_file(LOG_ARENA, f"[{ts()}] Mentor inject: {txt[:120]}...")
    return {"ok": True}


@app.post("/say/<agent>")
@require_api_token
def say_agent(agent: str):
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    dryrun = bool(payload.get("dryrun", False))
    if not text:
        return {"ok": False, "msg": "texto vazio"}, 400
    agent = agent.lower()
    if agent not in {"ernest", "garapa", "alfred"}:
        return {"ok": False, "msg": "agente inválido"}, 400
    cfg = load_config()
    reply = _call_agent_once(agent, text, cfg)
    _log_say_event(agent, text, reply, dryrun)
    executed = False
    if agent == "alfred" and not dryrun and cfg.get("allow_exec") and not cfg.get("supervised_only", False):
        try_execute_actions(reply, socketio, who="alfred-api", round_no=0)
        executed = True
    return {"ok": True, "agent": agent, "text": reply, "executed": executed}


@app.get("/status")
@require_api_token
def api_status():
    return {"ok": True, **current_status()}


@app.post("/restart")
@require_api_token
def restart_arena():
    global stop_signal
    payload = request.get_json(force=True, silent=True) or {}
    cfg = _apply_payload_to_config(load_config(), payload)
    cfg = apply_approval_policy(cfg)
    save_config(cfg)

    if arena_running:
        stop_signal = True
        manual_round_event.set()
        _wait_for_stop(10)

    stop_signal = False
    _launch_arena_thread()
    broadcast_status(cfg)
    append_file(LOG_ARENA, f"[{ts()}] Mentor solicitou restart")
    return {"ok": True, "config": cfg}


@app.post("/revert")
@require_api_token
def revert_config():
    cfg = apply_approval_policy(DEFAULT_CONFIG.copy())
    save_config(cfg)
    append_file(LOG_ARENA, f"[{ts()}] Config revertida para DEFAULT")
    broadcast_status(cfg)
    return {"ok": True, "config": cfg}


@app.post("/supervised")
def toggle_supervised():
    payload = request.get_json(force=True, silent=True) or {}
    cfg = load_config()
    if "enabled" in payload:
        cfg["supervised_only"] = bool(payload["enabled"])
    else:
        cfg["supervised_only"] = not cfg.get("supervised_only", False)
    save_config(cfg)
    broadcast_status(cfg)
    append_file(LOG_ARENA, f"[{ts()}] Supervised → {cfg['supervised_only']}")
    return {"ok": True, "supervised_only": cfg["supervised_only"]}


@app.post("/toggle_supervision")
@require_api_token
def toggle_supervision_api():
    return toggle_supervised()


@app.post("/approval_mode")
def set_approval_mode():
    payload = request.get_json(force=True, silent=True) or {}
    mode = (payload.get("mode") or "").strip().lower()
    if mode not in APPROVAL_MODES:
        return {"ok": False, "msg": f"modo inválido: {mode}"}, 400
    cfg = load_config()
    cfg["approval_mode"] = mode
    cfg = apply_approval_policy(cfg)
    save_config(cfg)
    broadcast_status(cfg)
    append_file(LOG_ARENA, f"[{ts()}] Approval mode → {mode}")
    return {"ok": True, "mode": mode}


@app.post("/workspace_access")
@require_api_token
def set_workspace_access():
    payload = request.get_json(force=True, silent=True) or {}
    allow_global = bool(payload.get("allow_global", False))
    cfg = load_config()
    cfg["allow_global_access"] = allow_global
    cfg = apply_approval_policy(cfg)
    save_config(cfg)
    broadcast_status(cfg)
    append_file(LOG_ARENA, f"[{ts()}] Workspace access → {allow_global}")
    return {"ok": True, "allow_global": allow_global}


@app.post("/manual_round")
def manual_round():
    if not arena_running:
        return {"ok": False, "msg": "Arena não está rodando"}, 400
    manual_round_event.set()
    append_file(LOG_ARENA, f"[{ts()}] Mentor requisitou rodada manual")
    return {"ok": True}


@app.get("/metrics")
def get_metrics():
    with _metrics_lock:
        payload = json.loads(json.dumps(metrics_state, ensure_ascii=False))
    return {"ok": True, "metrics": payload}


@app.get("/reports/latest")
def latest_report():
    with _metrics_lock:
        report = metrics_state.get("report") or {}
    if not report:
        return {"ok": False, "msg": "Sem dados ainda"}, 404
    return {"ok": True, "report": report}


@app.post("/reports/generate")
def manual_report():
    with _metrics_lock:
        report = metrics_state.get("report") or {}
    if not report:
        return {"ok": False, "msg": "Sem métricas para gerar relatório"}, 400
    entry = {"ts": ts(), "report": report}
    _log_metrics(entry)
    append_file(LOG_ARENA, f"[{ts()}] Mentor gerou relatório manual")
    return {"ok": True, "report": report}


# ============
# Bootstrap
# ============
if __name__ == "__main__":
    ensure_dirs()
    load_env_keys()
    cfg = load_config()
    sandbox_mode = (cfg.get("mode", "sandbox") == "sandbox")

    host = os.environ.get("FALCOM_HOST", "0.0.0.0")
    port = int(os.environ.get("FALCOM_PORT", "5050"))

    print(f"Falcom Arena on http://{host}:{port} — mode={cfg.get('mode')} rounds={cfg.get('max_rounds')}")
    socketio.run(app, host=host, port=port)
