"""High-level pipeline orchestrator (selector → seq → GPU → merge)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable

from core.orchestration.dl_ssh import RemoteSession, SSHSettings


def _log(stage: str, message: str) -> None:
    print(f"[pipeline:{stage}] {message}", flush=True)


def _normalize_cmd(cmd: Any) -> list[str]:
    if isinstance(cmd, str):
        return ["bash", "-lc", cmd]
    if isinstance(cmd, Iterable):
        return [str(part) for part in cmd]
    raise TypeError(f"Unsupported command type: {type(cmd)}")


def _run_local(section: dict, label: str) -> None:
    if not section:
        return
    cmd = _normalize_cmd(section.get("cmd"))
    cwd = section.get("cwd")
    env = os.environ.copy()
    env.update(section.get("env", {}))

    _log(label, " ".join(shlex.quote(part) for part in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _log(label, line.rstrip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{label} falhou com exit={ret}")


def _run_remote(remote_cfg: dict) -> None:
    if not remote_cfg:
        return
    ssh_cfg = SSHSettings(**remote_cfg["ssh"])
    if remote_cfg.get("dry_run"):
        ssh_cfg.dry_run = True
    workdir = remote_cfg.get("workdir")

    prepares = remote_cfg.get("prepare", [])
    uploads = remote_cfg.get("upload", [])
    downloads = remote_cfg.get("download", [])
    commands = remote_cfg.get("commands", [])

    with RemoteSession(ssh_cfg) as session:
        # Prepare remote (mkdirs, etc.) before rsync
        for entry in prepares:
            if isinstance(entry, dict):
                session.run(entry["cmd"], cwd=entry.get("cwd", workdir), env=entry.get("env"))
            else:
                session.run(entry, cwd=workdir)

        for item in uploads:
            session.sync_to_remote(item["local"], item["remote"], delete=item.get("delete", False))

        for entry in commands:
            if isinstance(entry, dict):
                session.run(
                    entry["cmd"],
                    cwd=entry.get("cwd", workdir),
                    env=entry.get("env"),
                )
            else:
                session.run(entry, cwd=workdir)

        for item in downloads:
            session.sync_from_remote(item["remote"], item["local"], delete=item.get("delete", False))


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit("Instale PyYAML para usar arquivos .yaml") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def run_pipeline(config_path: str) -> None:
    cfg = load_config(Path(config_path))
    _log("config", f"Carregando pipeline de {config_path}")

    if cfg.get("selector"):
        _run_local(cfg["selector"], "selector")
    if cfg.get("make_seq"):
        _run_local(cfg["make_seq"], "make_seq")
    if cfg.get("dl"):
        _run_local(cfg["dl"], "dl")
    if cfg.get("dl_local"):
        _run_local(cfg["dl_local"], "dl")
    if cfg.get("remote"):
        _run_remote(cfg["remote"])
    if cfg.get("merge"):
        _run_local(cfg["merge"], "merge")

    _log("done", "Pipeline concluído com sucesso.")


__all__ = ["run_pipeline"]
