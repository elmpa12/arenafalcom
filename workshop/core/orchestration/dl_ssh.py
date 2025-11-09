"""Lightweight SSH/Rsync helpers used by the DL orchestration pipeline."""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import paramiko


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _default_logger(level: str, message: str) -> None:
    print(f"[dl_ssh][{_ts()}][{level}] {message}", flush=True)


@dataclass
class SSHSettings:
    host: str
    user: str
    port: int = 22
    key_path: Optional[str] = None
    password: Optional[str] = None
    known_hosts: Optional[str] = None
    dry_run: bool = False


class RemoteSession:
    """Context manager around paramiko with rsync helpers."""

    def __init__(self, settings: SSHSettings, logger=_default_logger) -> None:
        self.settings = settings
        self._client: paramiko.SSHClient | None = None
        self.log = logger
        self._dry = bool(settings.dry_run)

    # ------------------------------------------------------------------ context
    def __enter__(self) -> "RemoteSession":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()

    # ----------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        if self._client is not None or self._dry:
            if self._dry:
                self.log("INFO", f"[DRY-RUN] SSH {self.settings.user}@{self.settings.host}:{self.settings.port}")
            return
        cli = paramiko.SSHClient()
        if self.settings.known_hosts:
            cli.load_host_keys(self.settings.known_hosts)
        else:
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        from pathlib import Path as _Path
        key_filename = self.settings.key_path
        if key_filename:
            key_filename = str(_Path(key_filename).expanduser())
        cli.connect(
            hostname=self.settings.host,
            port=self.settings.port,
            username=self.settings.user,
            key_filename=key_filename,
            password=self.settings.password,
            timeout=30,
        )
        cli.get_transport().set_keepalive(15)
        self._client = cli
        self.log("INFO", f"Conectado a {self.settings.user}@{self.settings.host}:{self.settings.port}")

    def close(self) -> None:
        if self._dry:
            self.log("INFO", "[DRY-RUN] Sessão SSH encerrada")
            return
        if self._client:
            self._client.close()
            self._client = None
            self.log("INFO", "Sessão SSH encerrada")

    # --------------------------------------------------------------------- utils
    def run(self, command: str | Iterable[str], *, cwd: Optional[str] = None, env: Optional[dict] = None, timeout: int | None = None) -> int:
        """
        Execute a remote command and stream stdout/stderr to the logger.

        ``command`` can be a shell string or an iterable of tokens.
        """

        if self._dry:
            if isinstance(command, (list, tuple)):
                cmd = " ".join(shlex.quote(str(part)) for part in command)
            else:
                cmd = command
            if env:
                exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
                cmd = f"export {exports} && {cmd}"
            if cwd:
                cmd = f"cd {shlex.quote(cwd)} && {cmd}"
            self.log("RUN", f"[DRY-RUN] {cmd}")
            return 0

        self.connect()
        assert self._client
        if isinstance(command, (list, tuple)):
            cmd = " ".join(shlex.quote(str(part)) for part in command)
        else:
            cmd = command
        if env:
            exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
            cmd = f"export {exports} && {cmd}"
        if cwd:
            cmd = f"cd {shlex.quote(cwd)} && {cmd}"

        self.log("RUN", cmd)
        channel = self._client.get_transport().open_session()
        if timeout:
            channel.settimeout(timeout)
        channel.exec_command(cmd)

        def _drain(stream, label: str) -> None:
            while not channel.exit_status_ready() or channel.recv_ready():
                data = stream.readline()
                if not data:
                    break
                if isinstance(data, bytes):
                    txt = data.decode("utf-8", "ignore").rstrip()
                else:
                    txt = str(data).rstrip()
                self.log(label, txt)

        stdout = channel.makefile("r")
        stderr = channel.makefile_stderr("r")

        _drain(stdout, "STDOUT")
        _drain(stderr, "STDERR")
        channel.shutdown_read()
        exit_status = channel.recv_exit_status()
        level = "OK" if exit_status == 0 else "ERR"
        self.log(level, f"Comando finalizado com exit={exit_status}")
        return exit_status

    # -------------------------------------------------------------- sync helpers
    def _ssh_transport(self) -> str:
        parts = ["ssh", "-p", str(self.settings.port)]
        if self.settings.key_path:
            parts.extend(["-i", self.settings.key_path])
        return " ".join(parts)

    def sync_to_remote(self, local: str | Path, remote: str, *, delete: bool = False) -> None:
        """rsync local→remote preserving permissões."""
        if self._dry:
            self.log("SYNC-UP", f"[DRY-RUN] {local} -> {self.settings.user}@{self.settings.host}:{remote}")
            return
        local_path = str(local)
        if Path(local_path).is_dir() and not local_path.endswith("/"):
            local_path = f"{local_path}/"
        remote_target = f"{self.settings.user}@{self.settings.host}:{remote}"
        cmd = [
            "rsync",
            "-az",
            "--info=progress2",
            "-e",
            self._ssh_transport(),
        ]
        if delete:
            cmd.append("--delete")
        cmd.extend([local_path, remote_target])
        self._run_local(cmd, "SYNC-UP")

    def sync_from_remote(self, remote: str, local: str | Path, *, delete: bool = False) -> None:
        if self._dry:
            self.log("SYNC-DOWN", f"[DRY-RUN] {self.settings.user}@{self.settings.host}:{remote} -> {local}")
            return
        remote_src = f"{self.settings.user}@{self.settings.host}:{remote}"
        local_path = str(local)
        Path(local_path).mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-az",
            "--info=progress2",
            "-e",
            self._ssh_transport(),
        ]
        if delete:
            cmd.append("--delete")
        cmd.extend([remote_src, local_path])
        self._run_local(cmd, "SYNC-DOWN")

    def _run_local(self, cmd: list[str], label: str) -> None:
        self.log(label, " ".join(shlex.quote(part) for part in cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout
        for line in proc.stdout:
            self.log(label, line.rstrip())
        exit_code = proc.wait()
        level = "OK" if exit_code == 0 else "ERR"
        self.log(level, f"{label} finalizado com exit={exit_code}")
        if exit_code != 0:
            raise RuntimeError(f"{label} falhou com exit={exit_code}")


__all__ = ["RemoteSession", "SSHSettings"]
