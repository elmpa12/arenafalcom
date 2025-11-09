#!/usr/bin/env python3
"""CLI para interagir com a Falcom Arena via endpoints /say e /status."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


def build_base_url() -> str:
    env_url = os.environ.get("ARENA_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    host = os.environ.get("ARENA_HOST")
    port = os.environ.get("FALCOM_PORT", "5051")
    if host:
        return f"http://{host}:{port}"
    return f"http://127.0.0.1:{port}"


def get_token() -> str:
    token = os.environ.get("ARENA_API_TOKEN")
    if not token:
        print("[arena] ARENA_API_TOKEN não definido — acesso negado", file=sys.stderr)
        sys.exit(1)
    return token


def http_post(base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        f"{base}{path}",
        json=payload,
        headers={"X-Auth": get_token()},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise SystemExit(f"[arena] erro {resp.status_code}: {resp.text}")
    return resp.json()


def http_get(base: str, path: str) -> dict[str, Any]:
    resp = requests.get(
        f"{base}{path}",
        headers={"X-Auth": get_token()},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise SystemExit(f"[arena] erro {resp.status_code}: {resp.text}")
    return resp.json()


def cmd_say(args: argparse.Namespace) -> None:
    base = build_base_url()
    agent = args.agent.lower().lstrip("@")
    if agent not in {"ernest", "garapa", "alfred"}:
        raise SystemExit("[arena] agente deve ser @ernest, @garapa ou @alfred")
    text = " ".join(args.text).strip()
    if not text:
        raise SystemExit("[arena] texto vazio")
    payload = {"text": text, "dryrun": args.dryrun}
    data = http_post(base, f"/say/{agent}", payload)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_status(_: argparse.Namespace) -> None:
    base = build_base_url()
    data = http_get(base, "/status")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="arena", description="Falcom Arena CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    say_p = sub.add_parser("say", help="enviar prompt para um agente")
    say_p.add_argument("agent", help="@ernest | @garapa | @alfred")
    say_p.add_argument("text", nargs=argparse.REMAINDER, help="mensagem")
    say_p.add_argument("--dryrun", action="store_true", help="não executar ações")
    say_p.set_defaults(func=cmd_say)

    status_p = sub.add_parser("status", help="status atual da Arena")
    status_p.set_defaults(func=cmd_status)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
