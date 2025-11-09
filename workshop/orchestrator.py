
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_all_v12_orchestrator_password.py — orquestrador (selector local + DL remoto)
• SSH por senha (default host 100.88.219.118, user gpuadmin, pass coco123)
• Criação de pastas no Windows via PowerShell (evita bugs de SFTP mkdir)
• Execução remota em background (Windows: Start-Process | Linux: nohup)
• Tail de log remoto (status|full|none)
• Espera curta por ZIP com fallback: tenta gerar o ZIP se houver arquivos; se não houver, segue em frente
• Download do ZIP se existir; descompacta localmente
"""

from __future__ import annotations
import argparse
import json
import os, sys, time, subprocess, traceback, re, threading, zipfile
import posixpath
import stat
import shlex
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.providers import (
    DEFAULT_METADATA_PATH,
    InstanceMeta,
    load_metadata,
    provider_factory,
    save_metadata,
)

# ====================== Logging ======================
class C:
    H="\033[95m"; B="\033[94m"; G="\033[92m"; Y="\033[93m"; R="\033[91m"; E="\033[0m"; DIM="\033[2m"

def _ts(): return datetime.now().strftime("%H:%M:%S")
def info(m): print(f"{C.G}[{_ts()}] [OK]{C.E} {m}", flush=True)
def warn(m): print(f"{C.Y}[{_ts()}] [WARN]{C.E} {m}", flush=True)
def err(m):  print(f"{C.R}[{_ts()}] [ERRO]{C.E} {m}", flush=True)
def dbg(m, enabled:bool):
    if enabled: print(f"{C.B}[{_ts()}] [DEBUG]{C.E} {m}", flush=True)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# ====================== SSH Helpers ======================
def ssh_connect(host, port, user, password=None, key_filename=None, attempts=3, debug=False):
    """Conecta via SSH com senha ou chave SSH e abre SFTP."""
    import paramiko, time as _t
    from pathlib import Path

    last = None
    auth_method = "chave SSH" if key_filename else "senha"

    for i in range(1, attempts+1):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            info(f"[GPU] SSH {user}@{host}:{port} (tentativa {i}/{attempts}) - {auth_method}")

            if key_filename:
                # Autenticação por chave SSH
                key_path = str(Path(key_filename).expanduser())
                cli.connect(
                    hostname=host, port=int(port), username=user,
                    key_filename=key_path, timeout=45
                )
            else:
                # Autenticação por senha
                cli.connect(
                    hostname=host, port=int(port), username=user, password=password,
                    timeout=45, allow_agent=False, look_for_keys=False
                )

            try:
                cli.get_transport().set_keepalive(15)
            except Exception:
                pass
            sftp = cli.open_sftp()
            info(f"[GPU] SFTP aberto com sucesso (autenticação por {auth_method})")
            return cli, sftp
        except Exception as e:
            last = e
            warn(f"Conexão falhou: {e}")
            _t.sleep(4*i)
    raise RuntimeError(f"Falha para conectar ao remoto: {last}")

def win_to_sftp(path:str)->str:
    r"""Converte C:\x\y\z -> /C:/x/y/z para uso em SFTP/stat."""
    p=path.replace("\\","/")
    if ":" in p and not p.startswith("/"):
        p="/"+p
    return p


def sftp_exists(ssh, path_sftp: str) -> bool:
    try:
        s = ssh.open_sftp()
        try:
            s.stat(path_sftp)
            return True
        finally:
            s.close()
    except Exception:
        return False

def sftp_dir_nonempty(ssh, dir_sftp: str) -> bool:
    try:
        s = ssh.open_sftp()
        try:
            return len(s.listdir(dir_sftp)) > 0
        finally:
            s.close()
    except Exception:
        return False

# ====================== Remote runners ======================
def start_remote_windows(ssh, sftp, args, remote_out, remote_zip, parquet_guess, remote_log, debug=False):
    """Executa o DL remoto diretamente (com todos os args do comando inicial)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Garante diretórios remotos
    for d in [args.gpu_root, f"{args.gpu_root}\\out", remote_out]:
        mkdir_cmd = f'powershell -Command "if (!(Test-Path -Path \\"{d}\\")) {{ New-Item -ItemType Directory -Path \\"{d}\\" -Force | Out-Null }}"'
        ssh.exec_command(mkdir_cmd)

    # Monta o comando de uma linha (com os args passados ao orquestrador)
    cmd = (
        f'powershell -ExecutionPolicy Bypass -NoProfile '
        f'-Command "cd {args.gpu_root}; '
        f'{args.gpu_python} {args.gpu_root}\\{args.dl_script} '
        f'--v2_path {args.gpu_root}\\selector21.py '
        f'--data_dir {args.gpu_root}\\datafull '
        f'--symbol {args.symbol} '
        f'--tf {args.dl_tf} '
        f'--start {args.start} '
        f'--end {args.end} '
        f'--out {remote_out} '
        f'--models {args.dl_models} '
        f'--horizon {args.dl_horizon} --lags {args.dl_lags} '
        f'--epochs {args.dl_epochs} --batch {args.dl_batch} '
        f'--device {args.dl_device} '
        f'> {remote_log} 2>&1; '
        f'if (Test-Path {remote_out}) {{ '
        f'if ((Get-ChildItem {remote_out}).Count -gt 0) {{ '
        f'Compress-Archive -Path {remote_out}\\* -DestinationPath {remote_zip} -Force; '
        f'Write-Host [ZIP_READY] {remote_zip}; '
        f'}} else {{ Write-Host [ZIP_SKIP] diretório vazio; }} '
        f'}} else {{ Write-Host [ZIP_SKIP] sem diretório remoto; }}"'
    )

    info(f"[GPU] Executando DL remoto com parâmetros do comando inicial...")
    dbg(cmd, debug)
    ssh.exec_command(cmd)
    info(f"[GPU] DL remoto disparado. Log: {remote_log}")
    return True




def start_remote_linux(ssh, sftp, args, remote_out, remote_zip, parquet_guess, remote_log, debug=False):
    """Cria .sh e inicia com nohup em background (Linux)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_sh  = f"{args.gpu_root}/out/dl_remote_{stamp}.sh"

    # cria pastas via shell remoto
    ssh.exec_command(f'mkdir -p "{args.gpu_root}/out" "{remote_out}"')

    sh = f"""#!/usr/bin/env bash
set -euo pipefail
cd "{args.gpu_root}"
PY="{args.gpu_python}"
if [ -f "{args.gpu_venv_activate}" ]; then
  source "{args.gpu_venv_activate}"; PY="python"
fi

if [ -f "{parquet_guess}" ]; then
  echo "[PARQUET] usando arquivo {parquet_guess}"
  $PY {args.dl_script} \
    --data_file "{parquet_guess}" \
    --tf {args.dl_tf} \
    --out "{remote_out}" \
    --models {args.dl_models} \
    --horizon {args.dl_horizon} --lags {args.dl_lags} \
    --wfo_mode {args.dl_window} \
    --wfo_train_months {args.dl_train_months} --wfo_val_months {args.dl_val_months} --wfo_step_months {args.dl_step_months} \
    --epochs {args.dl_epochs} --batch {args.dl_batch} \
    --device {args.dl_device} \
    --selector_glob '{args.dl_selector_glob}' \
    --extra_feats_csv '{args.dl_extra_feats_csv}'
else
  echo "[PARQUET] parquet não encontrado, fará enrich local"
  $PY {args.dl_script} \
    --v2_path "{args.gpu_root}/selector21.py" \
    --data_dir "{args.gpu_root}/datafull" \
    --symbol {args.symbol} \
    --tf {args.dl_tf} \
    --start {args.start} --end {args.end} \
    --out "{remote_out}" \
    --models {args.dl_models} \
    --horizon {args.dl_horizon} --lags {args.dl_lags} \
    --wfo_mode {args.dl_window} \
    --wfo_train_months {args.dl_train_months} --wfo_val_months {args.dl_val_months} --wfo_step_months {args.dl_step_months} \
    --epochs {args.dl_epochs} --batch {args.dl_batch} \
    --device {args.dl_device} \
    --selector_glob '{args.dl_selector_glob}' \
    --extra_feats_csv '{args.dl_extra_feats_csv}'
fi
if [ -d "{remote_out}" ] && [ "$(ls -A '{remote_out}')" ]; then
  zip -r "{remote_zip}" "{remote_out}" && echo "[ZIP_READY] {remote_zip}"
else
  echo "[ZIP_SKIP] nada para zipar"
fi
"""
    with sftp.open(remote_sh, "w") as f:
        f.write(sh)
    sftp.chmod(remote_sh, 0o755)
    ssh.exec_command(f'nohup bash "{remote_sh}" > "{remote_log}" 2>&1 < /dev/null &')
    info(f"[GPU] Job remoto (Linux) iniciado em background. Log: {remote_log}")
    return remote_sh

def tail_remote_log(ssh, remote_log: str, stop_event: threading.Event, mode: str = "status"):
    """Tail do log remoto via SFTP; mode=status|full|none."""
    if mode == "none":
        return
    def want(line: str) -> bool:
        if mode == "full":
            return True
        return bool(re.search(r"\[(EPOCH|SUMMARY|ZIP_READY|ZIP_SKIP|ERR|OOS|DL|PARQUET|REMOTE|TRAIN)\]", line))
    pos = 0
    while not stop_event.is_set():
        try:
            s = ssh.open_sftp()
            try:
                with s.open(remote_log, "r") as lf:
                    lf.seek(pos)
                    new = lf.read().decode("utf-8", "ignore")
                    if new:
                        for line in new.splitlines():
                            if line.strip() and want(line):
                                print(f"{C.DIM}[{_ts()}]{C.E} {C.B}[GPU-OUT]{C.E} {line}")
                        pos = lf.tell()
            finally:
                s.close()
        except Exception:
            time.sleep(3)
            continue
        time.sleep(2)

def wait_for_zip_or_skip(ssh, zip_path_sftp: str, remote_out_sftp: str, remote_log: str,
                         timeout_sec: int = 10) -> dict:
    """
    Espera até timeout_sec por [ZIP_READY] ou pelo arquivo ZIP.
    Se não aparecer, verifica se o diretório de saída existe e está vazio.
    Retorna dict com diagnóstico e não levanta exceção.
    """
    start = time.time()
    # loop curto
    while time.time() - start < timeout_sec:
        # 1) procura marker no log
        try:
            s = ssh.open_sftp()
            try:
                with s.open(remote_log, "r") as lf:
                    st = lf.stat()
                    lf.seek(max(0, st.st_size - 20000))
                    buf = lf.read().decode("utf-8", "ignore")
                    if "[ZIP_READY]" in buf:
                        return {"zip": True, "reason": "marker"}
            finally:
                s.close()
        except Exception:
            pass
        # 2) verifica existência do zip
        try:
            s = ssh.open_sftp()
            try:
                s.stat(zip_path_sftp)
                return {"zip": True, "reason": "stat"}
            finally:
                s.close()
        except Exception:
            pass
        time.sleep(2)

    # timeout — checa diretório
    out_exists = sftp_exists(ssh, remote_out_sftp)
    out_nonempty = sftp_dir_nonempty(ssh, remote_out_sftp) if out_exists else False
    return {"zip": False, "reason": "timeout", "out_exists": out_exists, "out_nonempty": out_nonempty}

# ====================== Main ======================
def legacy_main(argv=None):
    import argparse
    ap = argparse.ArgumentParser("orquestrador v12 (selector local + DL remoto, password)")
    # debug
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--live_tail", default="status", choices=["status","full","none"])
    # local/server
    ap.add_argument("--server_root", default="/opt/botscalp")
    ap.add_argument("--data_dir", default="/opt/botscalp/datafull")
    ap.add_argument("--selector_py", default="", help="Caminho do selector*.py (se vazio, tenta descobrir)")
    ap.add_argument("--v2_path", default="/opt/botscalp/selector21.py")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-09-30")
    ap.add_argument("--out_root", default="")
    # TFs
    ap.add_argument("--tfs", default="5m,15m")
    # selector extras
    ap.add_argument("--fee_perc", type=float, default=0.00018)
    ap.add_argument("--slippage", type=int, default=1)
    ap.add_argument("--selector_extra", default="", help="Pass-through adicional para o selector")
    # Walk-Forward (selector)
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--wf_train_months", type=float, default=1.0)
    ap.add_argument("--wf_val_months", type=float, default=0.25)
    ap.add_argument("--wf_step_months", type=float, default=0.25)
    ap.add_argument("--wf_grid_mode", default="light")
    ap.add_argument("--wf_top_n", type=int, default=5)
    ap.add_argument("--wf_expand", action="store_true")
    ap.add_argument("--wf_no_expand", action="store_true")
    # remoto/GPU (password-based or key-based)
    ap.add_argument("--gpu_host", default="100.88.219.118")
    ap.add_argument("--gpu_port", type=int, default=22)
    ap.add_argument("--gpu_user", default="gpuadmin")
    ap.add_argument("--gpu_pass", default="coco123")
    ap.add_argument("--gpu_key", default="", help="Chave SSH (se vazio, usa senha)")
    ap.add_argument("--gpu_root", default=r"C:\botscalp")
    ap.add_argument("--gpu_os", default="windows", choices=["windows","linux","auto"])
    ap.add_argument("--gpu_python", default=r"C:\botscalp\.venv\Scripts\python.exe")
    ap.add_argument("--gpu_venv_activate", default=r"C:\botscalp\.venv\Scripts\Activate.ps1")
    # DL
    ap.add_argument("--dl_script", default="dl_heads_v8.py")
    ap.add_argument("--dl_models", default="gru,lstm,cnn,transformer,dense")
    ap.add_argument("--dl_tf", default="5m")
    ap.add_argument("--dl_horizon", type=int, default=3)
    ap.add_argument("--dl_lags", type=int, default=128)
    ap.add_argument("--dl_epochs", type=int, default=16)
    ap.add_argument("--dl_batch", type=int, default=4096)
    ap.add_argument("--dl_window", default="monthly", choices=["monthly","weekly","none"])
    ap.add_argument("--dl_train_months", type=float, default=1.0)
    ap.add_argument("--dl_val_months", type=float, default=0.25)
    ap.add_argument("--dl_step_months", type=float, default=0.25)
    ap.add_argument("--dl_device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--dl_selector_glob", default="", help="Glob dos CSVs de probs do selector para merger")
    ap.add_argument("--dl_extra_feats_csv", default="", help="CSV de features extras para merger")
    # Espera do ZIP (curta)
    ap.add_argument("--zip_timeout_sec", type=int, default=10, help="Tempo máx para esperar o ZIP (padrão 10s)")
    args = ap.parse_args(argv)

    debug = bool(args.debug)
    info("== INÍCIO v12 ==")

    server_root = Path(args.server_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        err(f"--data_dir inexistente: {data_dir}"); sys.exit(2)

    # detectar selector
    cand = [args.selector_py, str(server_root/"selector7.py"), str(server_root/"selector21.py")]
    selector_py = None
    for c in cand:
        if c and Path(c).exists():
            selector_py = Path(c).resolve(); break
    if not selector_py:
        err("selector*.py não encontrado (tente --selector_py ou coloque no server_root)")
        sys.exit(2)
    v2_path = Path(args.v2_path).resolve()

    out_root = Path(args.out_root or (server_root/f"out/exp_v12_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")).resolve()
    ensure_dir(out_root); (out_root/"logs").mkdir(exist_ok=True, parents=True)

    # TFs
    tfs = [t.strip() for t in str(args.tfs).split(",") if t.strip()]
    info(f"TFs: {','.join(tfs)}")

    # ---------------- LOCAL: SELECTOR ----------------
    def job_selector_local():
        for tf in tfs:
            horizon = 3 if tf=="5m" else 5
            max_hold = 20 if tf=="5m" else 40
            out_dir = out_root/f"selector_{tf}"; ensure_dir(out_dir)

            wf_flags = ""
            if args.walkforward:
                wf_flags = (
                    f'--walkforward '
                    f'--wf_train_months {args.wf_train_months} '
                    f'--wf_val_months {args.wf_val_months} '
                    f'--wf_step_months {args.wf_step_months} '
                    f'--wf_grid_mode {args.wf_grid_mode} '
                    f'--wf_top_n {args.wf_top_n} '
                )
                if args.wf_expand:
                    wf_flags += "--wf_expand "
                elif args.wf_no_expand:
                    wf_flags += "--wf_no_expand "

            cmd = (
              f'{sys.executable} "{selector_py}" --v2_path "{v2_path}" --symbol {args.symbol} '
              f'--data_dir "{data_dir}" '
              f'--start {args.start} --end {args.end} --exec_rules "{tf}" '
              f'{wf_flags}'
              f'--run_ml --ml_model_kind ensemble_reg --ml_use_agg --ml_use_depth --ml_add_base_feats --ml_add_combo_feats '
              f'--ml_combo_ops MAJ --ml_combo_top_n 20 --ml_horizon {horizon} --ml_lags {10 if tf=="5m" else 12} '
              f'--ml_recency_mode exp --ml_recency_half_life 60 --ml_calibrate isotonic '
              f'--ml_opt_thr --ml_thr_grid 0.35,0.65,0.02 '
              f'--futures --fee_perc {args.fee_perc} --slippage {args.slippage} '
              f'--use_atr_stop --atr_stop_len 14 --atr_stop_mult 1.8 --use_atr_tp --atr_tp_len 14 --atr_tp_mult 1.0 '
              f'--timeout_mode bars --max_hold {max_hold} --trailing '
              f'--par_backend thread --n_jobs 8 --loader_verbose --print_top10 --out_root "{out_dir}" '
              f'--s3_spec_feats 1 --s3_meta_enable 1 --s3_tb_enable 1 --s3_l2_enable 1 --s3_l2_mode assist '
              f'{args.selector_extra}'
            )
            if args.dry_run:
                info(f"[SERVER] selector {tf} — (dry-run)")
                continue
            log_file = out_root/"logs"/f"selector_{tf}.log"
            info(f"[SERVER] selector {tf} — start (log: {log_file})")
            rc = subprocess.call(cmd, shell=True, cwd=str(server_root))
            if rc != 0:
                raise RuntimeError(f"selector {tf} falhou (rc={rc})")
            info(f"[SERVER] selector {tf} — done")
        return True

    # ---------------- REMOTO: DL HEADS ----------------
    def job_dl_remote():
        # Usar chave SSH se fornecida, senão usar senha
        key_file = args.gpu_key if args.gpu_key else None
        password = None if key_file else args.gpu_pass
        cli, sftp = ssh_connect(args.gpu_host, args.gpu_port, args.gpu_user, password=password, key_filename=key_file, attempts=4, debug=debug)
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            remote_out = args.gpu_root + ("\\out\\dl_v7_" + stamp if args.gpu_os=="windows" else f"/out/dl_v7_{stamp}")
            remote_zip = args.gpu_root + ("\\out\\dl_v7_" + stamp + ".zip" if args.gpu_os=="windows" else f"/out/dl_v7_{stamp}.zip")
            parquet_guess = args.gpu_root + (f"\\datafull\\merged_{args.symbol}_{args.dl_tf}_{args.start.replace('-','')}_{args.end.replace('-','')}.parquet" if args.gpu_os=="windows" else f"/datafull/merged_{args.symbol}_{args.dl_tf}_{args.start.replace('-','')}_{args.end.replace('-','')}.parquet")
            remote_log = args.gpu_root + ("\\out\\dl_remote_" + stamp + ".log" if args.gpu_os=="windows" else f"/out/dl_remote_{stamp}.log")

            # inicia job remoto
            if args.gpu_os == "linux":
                start_remote_linux(cli, sftp, args, remote_out, remote_zip, parquet_guess, remote_log, debug=debug)
            else:
                start_remote_windows(cli, sftp, args, remote_out, remote_zip, parquet_guess, remote_log, debug=debug)

            # tail do log remoto (thread)
            stop = threading.Event()
            t = threading.Thread(target=tail_remote_log, args=(cli, remote_log, stop, args.live_tail), daemon=True)
            t.start()

            # aguarda curto por ZIP ou diagnóstico
            zip_sftp = win_to_sftp(remote_zip) if args.gpu_os=="windows" else remote_zip
            out_sftp = win_to_sftp(remote_out) if args.gpu_os=="windows" else remote_out
            info("[GPU] aguardando término (ZIP_READY/arquivo)...")
            res = wait_for_zip_or_skip(cli, zip_sftp, out_sftp, remote_log, timeout_sec=int(args.zip_timeout_sec))
            stop.set()

            # download se existir
            if res.get("zip"):
                local_zip = Path(server_root/"out")/Path(zip_sftp).name
                ensure_dir(local_zip.parent)
                s = cli.open_sftp()
                try:
                    s.get(zip_sftp, str(local_zip))
                    info(f"[GPU] ZIP baixado: {local_zip}")
                    # extrai
                    with zipfile.ZipFile(local_zip, "r") as zf:
                        zf.extractall(server_root/"out"/"dl_heads_remote")
                    info("[GPU] ZIP extraído em out/dl_heads_remote")
                finally:
                    s.close()
            else:
                # Sem ZIP: loga diagnóstico e segue em frente
                warn(f"[GPU] Sem ZIP — reason={res.get('reason')} out_exists={res.get('out_exists')} out_nonempty={res.get('out_nonempty')} (seguindo em frente)")

            return True
        finally:
            try: cli.close()
            except Exception: pass

    # ------------- Execução paralela -------------
    t0 = time.time()
    had_error = False; reasons = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_srv = ex.submit(job_selector_local)
        fut_gpu = ex.submit(job_dl_remote)
        for fut in as_completed([fut_srv, fut_gpu]):
            try:
                fut.result()
            except Exception as e:
                had_error = True
                reasons.append(str(e))
                err(f"Job falhou: {e}\n{traceback.format_exc()}")

    if had_error:
        err("== FIM COM ERROS ==")
        for i, r in enumerate(reasons, 1):
            err(f"Motivo {i}: {r}")
        sys.exit(3)
    else:
        info("== FIM ==")

    print(f"{C.H}✅ Pipeline paralelo finalizado{C.E}")
    print(f"{C.B}Saída raiz:{C.E} {server_root/'out'}")
    print(f"{C.B}Tempo total:{C.E} {int(time.time()-t0)}s")



# ====================== Nova CLI (providers) ======================
NEW_COMMANDS = {"launch", "run", "terminate", "fullrun", "config"}

CLI_CONFIG_PATH = Path("tools/orchestrator_cli_config.json")
CONFIG_SECTIONS = ("launch", "run", "fullrun")

YES_INPUTS = {"y", "yes", "s", "sim", "1", "true"}
NO_INPUTS = {"n", "no", "nao", "não", "0", "false"}
NULL_INPUTS = {"none", "null", "nil", "vazio", "empty", "-"}


@dataclass
class ConfigField:
    section: str
    key: str
    prompt: str
    kind: str = "str"
    default: Any = None
    help: Optional[str] = None
    nullable: bool = False
    required: bool = False


CONFIG_FIELDS: List[ConfigField] = [
    ConfigField(
        "launch",
        "provider",
        "Provider padrão",
        default="aws",
        help="Nome do provider (ex.: aws).",
    ),
    ConfigField(
        "launch",
        "region",
        "Região AWS",
        default="us-east-1",
    ),
    ConfigField(
        "launch",
        "instance_type",
        "Tipo de instância",
        default="g5.xlarge",
    ),
    ConfigField(
        "launch",
        "ami",
        "AMI (imagem base)",
        default="ami-053b0d53c279acc90",
    ),
    ConfigField(
        "launch",
        "key_name",
        "Nome da chave EC2 (KeyPair)",
        default="",
        required=True,
    ),
    ConfigField(
        "launch",
        "name",
        "Tag Name padrão",
        default="BotScalp-GPU",
    ),
    ConfigField(
        "launch",
        "spot",
        "Usar Spot instances?",
        kind="bool",
        default=False,
    ),
    ConfigField(
        "launch",
        "max_price",
        "Preço máximo Spot (USD/h)",
        kind="str",
        default=None,
        nullable=True,
        help="Ex.: 0.35. Use 'none' para limpar.",
    ),
    ConfigField(
        "launch",
        "volume_size",
        "Tamanho do volume raiz (GB)",
        kind="int",
        default=200,
    ),
    ConfigField(
        "launch",
        "subnet_id",
        "Subnet ID (opcional)",
        default=None,
        nullable=True,
    ),
    ConfigField(
        "launch",
        "security_group_id",
        "Security Group ID (opcional)",
        default=None,
        nullable=True,
    ),
    ConfigField(
        "launch",
        "ssh_cidrs",
        "CIDRs liberados para SSH (separados por vírgula)",
        kind="list",
        default=["0.0.0.0/0"],
        nullable=True,
    ),
    ConfigField(
        "launch",
        "host",
        "Host/IP para provider local",
        default=None,
        nullable=True,
        help="Usado quando provider=local",
    ),
    ConfigField(
        "launch",
        "tags",
        "Tags extras (K=V, separadas por vírgula)",
        kind="tags",
        default=[],
        nullable=True,
    ),
    ConfigField(
        "launch",
        "ssh_user",
        "Usuário SSH padrão",
        default="ubuntu",
    ),
    ConfigField(
        "launch",
        "ssh_key_hint",
        "Hint de chave SSH mostrado após o launch",
        default="~/.ssh/botscalp-key.pem",
    ),
    ConfigField(
        "run",
        "ssh_user",
        "Usuário SSH para run/fullrun",
        default="ubuntu",
    ),
    ConfigField(
        "run",
        "ssh_key",
        "Caminho da chave SSH privada",
        default="~/.ssh/botscalp-key.pem",
        required=True,
    ),
    ConfigField(
        "run",
        "ssh_port",
        "Porta SSH",
        kind="int",
        default=22,
    ),
    ConfigField(
        "run",
        "ssh_attempts",
        "Tentativas de conexão SSH",
        kind="int",
        default=4,
    ),
    ConfigField(
        "run",
        "remote_cmd",
        "Comando remoto padrão (opcional)",
        default=None,
        nullable=True,
    ),
    ConfigField(
        "run",
        "workdir",
        "Diretório remoto de trabalho",
        default="~",
    ),
    ConfigField(
        "run",
        "fail_fast",
        "Interromper downloads se o comando remoto falhar?",
        kind="bool",
        default=False,
    ),
    ConfigField(
        "run",
        "s3_bucket",
        "Bucket S3 padrão (opcional)",
        default=None,
        nullable=True,
    ),
    ConfigField(
        "run",
        "s3_prefix",
        "Prefixo base no S3 (opcional)",
        default=None,
        nullable=True,
    ),
    ConfigField(
        "run",
        "s3_region",
        "Região S3 (override, opcional)",
        default=None,
        nullable=True,
    ),
    ConfigField(
        "run",
        "s3_profile",
        "Perfil AWS local para S3 (opcional)",
        default=None,
        nullable=True,
    ),
    ConfigField(
        "fullrun",
        "terminate_on_finish",
        "Encerrar instância automaticamente ao término do fullrun?",
        kind="bool",
        default=False,
    ),
]


def _load_cli_config(path: Path = CLI_CONFIG_PATH) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        warn(f"Configuração inválida em {path}: {exc}")
        return {}


def _save_cli_config(config: Dict[str, Any], path: Path = CLI_CONFIG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def _config_section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = config.get(name)
    if isinstance(section, dict):
        return dict(section)
    return {}


def _config_list(section: Dict[str, Any], key: str, fallback: Optional[List[str]] = None) -> List[str]:
    value = section.get(key)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    if fallback is None:
        return []
    return list(fallback)


def _config_bool(section: Dict[str, Any], key: str, fallback: bool = False) -> bool:
    value = section.get(key)
    if isinstance(value, bool):
        return value
    return bool(fallback)


def _config_value(section: Dict[str, Any], key: str, fallback: Any = None) -> Any:
    if key in section:
        return section[key]
    return fallback


def _format_field_value(field: ConfigField, value: Any) -> str:
    if field.kind in {"list", "tags"}:
        if not value:
            return "nenhum" if field.required else ""
        return ", ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "sim" if value else "não"
    if value in (None, ""):
        return "não definido" if field.required else ""
    return str(value)


def _convert_field_input(field: ConfigField, raw: str) -> Any:
    text = raw.strip()
    if not text:
        return None
    lowered = text.lower()
    if field.nullable and lowered in NULL_INPUTS:
        if field.kind in {"list", "tags"}:
            return []
        return None
    if field.kind == "bool":
        if lowered in YES_INPUTS:
            return True
        if lowered in NO_INPUTS:
            return False
        raise ValueError("responda com s/n")
    if field.kind == "int":
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError("use um número inteiro") from exc
    if field.kind == "list":
        items = [item.strip() for item in text.split(",") if item.strip()]
        return items
    if field.kind == "tags":
        items: List[str] = []
        for part in text.split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"tag inválida '{item}', use CHAVE=VALOR")
            items.append(item)
        return items
    return text


def _prompt_field(field: ConfigField, current: Any) -> Any:
    while True:
        if field.help:
            print(f"  {field.help}")
        label = field.prompt
        current_display = _format_field_value(field, current)
        if field.kind == "bool":
            answer = input(f"{label} [{current_display}] (s/n, Enter mantém): ").strip()
        else:
            suffix = " (Enter mantém)" if current_display else ""
            answer = input(f"{label} [{current_display}]{suffix}: ").strip()
        if not answer:
            value = current
        else:
            try:
                converted = _convert_field_input(field, answer)
            except ValueError as exc:
                print(f"  Valor inválido: {exc}")
                continue
            if converted is None and not field.nullable and field.kind not in {"list", "tags"}:
                # mantém comportamento anterior se usuário digitou 'none' em campo não nulo
                value = current
            else:
                value = converted
        if field.required:
            if field.kind in {"list", "tags"} and not value:
                print("  Informe ao menos um valor.")
                continue
            if value in (None, ""):
                print("  Esse campo é obrigatório.")
                continue
        return value


def _interactive_config(existing: Dict[str, Any]) -> Dict[str, Any]:
    print(f"{C.B}== Guia de configuração do orchestrator =={C.E}")
    print("Pressione Enter para manter o valor atual. Use 'none' para limpar quando disponível.\n")
    config: Dict[str, Dict[str, Any]] = {section: dict(_config_section(existing, section)) for section in CONFIG_SECTIONS}

    for field in CONFIG_FIELDS:
        section_values = config.setdefault(field.section, {})
        current = section_values.get(field.key, field.default)
        if field.section == "run" and field.key == "ssh_user" and (
            current in (None, "", field.default)
        ):
            launch_user = config.get("launch", {}).get("ssh_user")
            if launch_user:
                current = launch_user
        if field.section == "run" and field.key == "ssh_key" and (
            current in (None, "", field.default)
        ):
            key_hint = config.get("launch", {}).get("ssh_key_hint")
            if key_hint:
                current = key_hint
        value = _prompt_field(field, current)
        if field.kind in {"list", "tags"} and value is None:
            section_values[field.key] = []
        else:
            section_values[field.key] = value

    cleaned: Dict[str, Dict[str, Any]] = {}
    for section, values in config.items():
        normalized: Dict[str, Any] = {}
        for key, value in values.items():
            if value is None:
                normalized[key] = None
            elif isinstance(value, list):
                normalized[key] = list(value)
            else:
                normalized[key] = value
        if normalized:
            cleaned[section] = normalized
    return cleaned


def _cmd_config(args: argparse.Namespace) -> int:
    existing = {} if getattr(args, "reset", False) else _load_cli_config()
    if getattr(args, "show", False):
        print(json.dumps(existing, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    try:
        updated = _interactive_config(existing)
    except KeyboardInterrupt:
        print("\nConfiguração cancelada pelo usuário.")
        return 1
    _save_cli_config(updated)
    print(f"Configuração salva em {CLI_CONFIG_PATH}")
    return 0


def _parse_tags(items: Iterable[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Tag inválida: '{item}' (use CHAVE=VALOR)")
        key, value = item.split("=", 1)
        tags[key.strip()] = value.strip()
    return tags


def _expand_remote(path: str, user: str) -> str:
    if path.startswith("~/"):
        return posixpath.join("/home", user, path[2:])
    if path == "~":
        return posixpath.join("/home", user)
    return path


def _connect_with_key(host: str, port: int, user: str, key_path: str, attempts: int = 4, delay: int = 5):
    import paramiko

    key_path = str(Path(key_path).expanduser())
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, port=port, username=user, key_filename=key_path, timeout=45)
            return client
        except Exception as exc:  # pragma: no cover - network interaction
            last_exc = exc
            warn(f"SSH falhou ({attempt}/{attempts}): {exc}")
            time.sleep(delay * attempt)
    raise SystemExit(f"Não foi possível conectar ao host {user}@{host}:{port}: {last_exc}")


def _ensure_remote_dir(sftp, remote_dir: str) -> None:
    if not remote_dir:
        return
    remote_dir = posixpath.normpath(remote_dir)
    if remote_dir in ("", "."):
        return
    parts = remote_dir.split("/")
    current = "" if not remote_dir.startswith("/") else "/"
    for part in parts:
        if not part:
            continue
        next_path = posixpath.join(current, part) if current else part
        try:
            sftp.stat(next_path)
        except IOError:
            sftp.mkdir(next_path)
        current = next_path


def _is_remote_dir(sftp, path: str) -> bool:
    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        return False


def _require_boto3() -> "module":
    try:
        import boto3  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependência opcional
        raise SystemExit(
            "Operações com S3 exigem boto3. Instale com 'pip install boto3'."
        ) from exc
    return boto3


def _make_s3_client(region: Optional[str], profile: Optional[str]):
    boto3 = _require_boto3()
    session_kwargs: Dict[str, Any] = {}
    if region:
        session_kwargs["region_name"] = region
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    return session.client("s3")


def _join_s3_key(prefix: Optional[str], key: Optional[str]) -> str:
    parts: List[str] = []
    for value in (prefix, key):
        if not value:
            continue
        text = str(value).strip("/")
        if text:
            parts.append(text)
    return "/".join(parts)


def _s3_upload_paths(
    bucket: Optional[str],
    prefix: Optional[str],
    specs: Sequence[str],
    *,
    region: Optional[str],
    profile: Optional[str],
) -> None:
    if not specs:
        return
    if not bucket:
        raise SystemExit("Use --s3-bucket para enviar arquivos ao S3.")
    client = _make_s3_client(region, profile)
    for raw in specs:
        if not raw:
            continue
        if ":" in raw:
            src_text, key_text = raw.split(":", 1)
        else:
            src_text, key_text = raw, ""
        src_path = Path(src_text).expanduser().resolve()
        if not src_path.exists():
            raise SystemExit(f"Arquivo local '{src_path}' não encontrado para upload S3")
        if src_path.is_dir():
            base_key = key_text.strip() or src_path.name
            base_key = base_key.rstrip("/")
            if not base_key:
                raise SystemExit("Diretórios S3 precisam de uma chave/prefixo destino")
            for local_root, _, files in os.walk(src_path):
                root_path = Path(local_root)
                for name in files:
                    local_file = root_path / name
                    rel = local_file.relative_to(src_path)
                    object_key = _join_s3_key(prefix, f"{base_key}/{rel.as_posix()}")
                    info(f"[S3 upload] {local_file} -> s3://{bucket}/{object_key}")
                    client.upload_file(str(local_file), bucket, object_key)
        else:
            key_text = key_text.strip()
            object_key = key_text if key_text else src_path.name
            if object_key.endswith("/"):
                object_key = f"{object_key}{src_path.name}"
            object_key = _join_s3_key(prefix, object_key)
            info(f"[S3 upload] {src_path} -> s3://{bucket}/{object_key}")
            client.upload_file(str(src_path), bucket, object_key)


def _s3_object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:  # pragma: no cover - depende da conta AWS
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
        raise


def _s3_download_specs(
    bucket: Optional[str],
    prefix: Optional[str],
    specs: Sequence[str],
    *,
    region: Optional[str],
    profile: Optional[str],
) -> None:
    if not specs:
        return
    if not bucket:
        raise SystemExit("Use --s3-bucket para baixar arquivos do S3.")
    client = _make_s3_client(region, profile)
    paginator = client.get_paginator("list_objects_v2")
    for raw in specs:
        if not raw:
            continue
        if ":" in raw:
            key_text, local_text = raw.split(":", 1)
        else:
            key_text, local_text = raw, ""
        key_text = key_text.strip()
        if not key_text:
            raise SystemExit("Especifique a chave ou prefixo S3 para download")
        local_path = Path(local_text.strip() or Path(key_text).name).expanduser().resolve()
        full_key = _join_s3_key(prefix, key_text)
        if _s3_object_exists(client, bucket, full_key):
            ensure_dir(local_path.parent)
            info(f"[S3 download] s3://{bucket}/{full_key} -> {local_path}")
            client.download_file(bucket, full_key, str(local_path))
            continue
        prefix_key = full_key.rstrip("/") + "/"
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix_key):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.startswith(prefix_key):
                    continue
                rel = key[len(prefix_key) :]
                target = local_path / rel if rel else local_path
                ensure_dir(target.parent)
                info(f"[S3 download] s3://{bucket}/{key} -> {target}")
                client.download_file(bucket, key, str(target))
                found = True
        if not found:
            raise SystemExit(f"Nenhum objeto encontrado em s3://{bucket}/{full_key}")


def _upload_paths(ssh, uploads: Sequence[str], user: str) -> None:
    if not uploads:
        return
    sftp = ssh.open_sftp()
    try:
        for spec in uploads:
            if not spec:
                continue
            if ":" in spec:
                src, dst = spec.split(":", 1)
            else:
                src = spec
                dst = posixpath.join("~", Path(src).name)
            src_path = Path(src).expanduser().resolve()
            dst_path = _expand_remote(dst, user)
            if src_path.is_dir():
                for local_root, _, files in os.walk(src_path):
                    rel = Path(local_root).relative_to(src_path)
                    remote_root = posixpath.join(dst_path, rel.as_posix()) if str(rel) != "." else dst_path
                    _ensure_remote_dir(sftp, remote_root)
                    for name in files:
                        local_file = Path(local_root) / name
                        remote_file = posixpath.join(remote_root, name)
                        info(f"[UPLOAD] {local_file} -> {remote_file}")
                        sftp.put(str(local_file), remote_file)
            else:
                _ensure_remote_dir(sftp, posixpath.dirname(dst_path))
                info(f"[UPLOAD] {src_path} -> {dst_path}")
                sftp.put(str(src_path), dst_path)
    finally:
        sftp.close()


def _download_paths(ssh, downloads: Sequence[str], user: str) -> None:
    if not downloads:
        return
    sftp = ssh.open_sftp()
    try:
        for spec in downloads:
            if not spec:
                continue
            if ":" not in spec:
                raise SystemExit("Use --download REMOTO:LOCAL")
            remote, local = spec.split(":", 1)
            remote_path = _expand_remote(remote, user)
            local_path = Path(local).expanduser().resolve()
            if _is_remote_dir(sftp, remote_path):
                ensure_dir(local_path)
                for entry in sftp.listdir_attr(remote_path):
                    src = posixpath.join(remote_path, entry.filename)
                    dst = local_path / entry.filename
                    if stat.S_ISDIR(entry.st_mode):
                        ensure_dir(dst)
                        _download_paths(ssh, [f"{src}:{dst}"] , user)
                    else:
                        info(f"[DOWNLOAD] {src} -> {dst}")
                        sftp.get(src, str(dst))
            else:
                ensure_dir(local_path.parent)
                info(f"[DOWNLOAD] {remote_path} -> {local_path}")
                sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()


def _exec_remote(ssh, command: str, workdir: Optional[str], user: str) -> int:
    if workdir:
        remote_dir = _expand_remote(workdir, user)
        full_cmd = f"cd {shlex.quote(remote_dir)} && {command}"
    else:
        full_cmd = command
    info(f"[GPU] Executando: {full_cmd}")
    stdin, stdout, stderr = ssh.exec_command(full_cmd)
    stdin.close()
    for line in stdout:
        print(line.rstrip())
    err_lines = stderr.readlines()
    for line in err_lines:
        warn(line.rstrip())
    exit_code = stdout.channel.recv_exit_status()
    info(f"[GPU] comando finalizado (rc={exit_code})")
    return exit_code


def _cmd_launch(args: argparse.Namespace) -> int:
    provider = provider_factory(args.provider)
    provider_name = getattr(provider, "provider_name", args.provider).lower()
    tags = _parse_tags(args.tags)
    cloud_init = Path(args.cloud_init).read_text(encoding="utf-8") if args.cloud_init else None

    meta: Optional[InstanceMeta] = None
    if provider_name == "local":
        host_value = (getattr(args, "host", None) or "").strip()
        if not host_value:
            raise SystemExit("Provider local requer --host com IP ou hostname acessível")
        if args.reuse:
            meta = provider.reuse(host=host_value, name=args.name, region=args.region, tags=tags)
        if not meta:
            meta = provider.launch(host=host_value, name=args.name, region=args.region, tags=tags)
            info(f"Host local preparado: {meta.public_ip}")
    else:
        if args.reuse:
            meta = provider.reuse(region=args.region, name=args.name, tags=tags)
            if meta:
                info(f"Reutilizando instância {meta.instance_id} ({meta.state})")
                if getattr(provider, "refresh", None) and not args.no_wait:
                    meta = provider.refresh(meta, wait=True)
        if not meta:
            if not args.key_name:
                raise SystemExit("Informe --key-name para provider AWS")
            launch_kwargs = dict(
                region=args.region,
                instance_type=args.instance_type,
                ami=args.ami,
                key_name=args.key_name,
                name=args.name,
                spot=args.spot,
                max_price=args.max_price,
                volume_size=args.volume_size,
                subnet_id=args.subnet_id,
                security_group_id=args.security_group_id,
                ssh_cidrs=args.ssh_cidrs,
                cloud_init=cloud_init,
                wait=not args.no_wait,
                tags=tags,
            )
            meta = provider.launch(**launch_kwargs)
            info(f"Instância criada: {meta.instance_id}")

    save_metadata(meta, Path(args.write_meta))
    print(f"Metadados salvos em {args.write_meta}")
    if meta.public_ip:
        print(f"ssh -i {args.ssh_key_hint} {args.ssh_user}@{meta.public_ip}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    meta = load_metadata(Path(args.meta))
    host = meta.public_ip or meta.private_ip
    if not host:
        raise SystemExit("Metadados sem IP público ou privado disponível")
    if not args.remote_cmd:
        raise SystemExit("Informe --remote-cmd ou defina um padrão via 'orchestrator config'")
    s3_bucket = getattr(args, "s3_bucket", None) or None
    s3_prefix = getattr(args, "s3_prefix", None) or None
    if s3_prefix == "":
        s3_prefix = None
    s3_region = getattr(args, "s3_region", None) or None
    s3_profile = getattr(args, "s3_profile", None) or None
    s3_uploads = [item for item in getattr(args, "s3_upload", []) if item]
    s3_downloads = [item for item in getattr(args, "s3_download", []) if item]

    _s3_upload_paths(
        s3_bucket,
        s3_prefix,
        s3_uploads,
        region=s3_region,
        profile=s3_profile,
    )
    client = _connect_with_key(host, args.ssh_port, args.ssh_user, args.ssh_key, attempts=args.ssh_attempts)
    try:
        _upload_paths(client, args.upload, args.ssh_user)
        exit_code = _exec_remote(client, args.remote_cmd, args.workdir, args.ssh_user)
        if exit_code != 0 and args.fail_fast:
            raise SystemExit(exit_code)
        _download_paths(client, args.download, args.ssh_user)
    finally:
        client.close()
    _s3_download_specs(
        s3_bucket,
        s3_prefix,
        s3_downloads,
        region=s3_region,
        profile=s3_profile,
    )
    return 0


def _cmd_terminate(args: argparse.Namespace) -> int:
    meta = load_metadata(Path(args.meta))
    provider = provider_factory(meta.provider)
    provider.terminate(meta)
    info(f"Instância {meta.instance_id} encerrada")
    return 0


def _cmd_fullrun(args: argparse.Namespace) -> int:
    launch_args = argparse.Namespace(
        provider=args.provider,
        region=args.region,
        instance_type=args.instance_type,
        ami=args.ami,
        key_name=args.key_name,
        name=args.name,
        tags=args.tags,
        spot=args.spot,
        max_price=args.max_price,
        volume_size=args.volume_size,
        subnet_id=args.subnet_id,
        security_group_id=args.security_group_id,
        ssh_cidrs=args.ssh_cidrs,
        cloud_init=args.cloud_init,
        no_wait=args.no_wait,
        reuse=args.reuse,
        write_meta=args.meta,
        ssh_key_hint=args.ssh_key,
        ssh_user=args.ssh_user,
        host=args.host,
    )
    _cmd_launch(launch_args)
    try:
        run_args = argparse.Namespace(
            meta=args.meta,
            ssh_user=args.ssh_user,
            ssh_key=args.ssh_key,
            ssh_port=args.ssh_port,
            ssh_attempts=args.ssh_attempts,
            upload=args.upload,
            remote_cmd=args.remote_cmd,
            workdir=args.workdir,
            download=args.download,
            fail_fast=args.fail_fast,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_region=args.s3_region,
            s3_profile=args.s3_profile,
            s3_upload=args.s3_upload,
            s3_download=args.s3_download,
        )
        _cmd_run(run_args)
    finally:
        if args.terminate_on_finish:
            term_args = argparse.Namespace(meta=args.meta)
            _cmd_terminate(term_args)
    return 0


def _build_new_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BotScalp orchestrator (providers)")
    stored_config = _load_cli_config()
    launch_defaults = _config_section(stored_config, "launch")
    run_defaults = _config_section(stored_config, "run")
    full_defaults = _config_section(stored_config, "fullrun")

    if "ssh_user" not in run_defaults and launch_defaults.get("ssh_user"):
        run_defaults["ssh_user"] = launch_defaults["ssh_user"]
    if "ssh_key" not in run_defaults and launch_defaults.get("ssh_key_hint"):
        run_defaults["ssh_key"] = launch_defaults["ssh_key_hint"]

    sub = parser.add_subparsers(dest="command", required=True)

    launch_parent = argparse.ArgumentParser(add_help=False)
    launch_parent.add_argument("--provider", default=_config_value(launch_defaults, "provider", "aws"))
    launch_parent.add_argument("--region", default=_config_value(launch_defaults, "region", "us-east-1"))
    launch_parent.add_argument("--instance-type", default=_config_value(launch_defaults, "instance_type", "g5.xlarge"))
    launch_parent.add_argument("--ami", default=_config_value(launch_defaults, "ami", "ami-053b0d53c279acc90"))

    key_name_default = _config_value(launch_defaults, "key_name") or None
    launch_parent.add_argument("--key-name", default=key_name_default)

    launch_parent.add_argument("--name", default=_config_value(launch_defaults, "name", "BotScalp-GPU"))

    tags_default = _config_list(launch_defaults, "tags", [])
    launch_parent.add_argument("--tag", action="append", default=list(tags_default), dest="tags")

    spot_default = _config_bool(launch_defaults, "spot", False)
    launch_parent.add_argument("--spot", dest="spot", action="store_true", help="Solicita instância Spot")
    launch_parent.add_argument(
        "--no-spot",
        dest="spot",
        action="store_false",
        help="Força on-demand mesmo quando o padrão é Spot",
    )

    launch_parent.add_argument("--max-price", default=_config_value(launch_defaults, "max_price"))

    volume_size_default = _config_value(launch_defaults, "volume_size", 200)
    try:
        volume_size_default = int(volume_size_default)
    except (TypeError, ValueError):
        volume_size_default = 200
    launch_parent.add_argument("--volume-size", type=int, default=volume_size_default)

    subnet_id_default = _config_value(launch_defaults, "subnet_id")
    if not subnet_id_default:
        subnet_id_default = None
    launch_parent.add_argument("--subnet-id", default=subnet_id_default)

    sec_group_default = _config_value(launch_defaults, "security_group_id")
    if not sec_group_default:
        sec_group_default = None
    launch_parent.add_argument("--security-group-id", default=sec_group_default)

    ssh_cidrs_default = _config_list(launch_defaults, "ssh_cidrs", ["0.0.0.0/0"])
    launch_parent.add_argument(
        "--ssh-cidr",
        action="append",
        default=list(ssh_cidrs_default),
        dest="ssh_cidrs",
    )

    launch_parent.add_argument("--cloud-init")
    launch_parent.add_argument("--host", default=_config_value(launch_defaults, "host"))
    launch_parent.add_argument("--no-wait", action="store_true")
    launch_parent.add_argument("--reuse", action="store_true")
    launch_parent.set_defaults(spot=spot_default)

    run_parent = argparse.ArgumentParser(add_help=False)
    run_parent.add_argument("--meta", default=str(DEFAULT_METADATA_PATH))
    run_parent.add_argument("--ssh-user", default=_config_value(run_defaults, "ssh_user", "ubuntu"))

    ssh_key_default = _config_value(run_defaults, "ssh_key", "~/.ssh/botscalp-key.pem")
    if not ssh_key_default:
        ssh_key_default = "~/.ssh/botscalp-key.pem"
    run_parent.add_argument("--ssh-key", default=ssh_key_default, required=not bool(ssh_key_default))

    ssh_port_default = _config_value(run_defaults, "ssh_port", 22)
    try:
        ssh_port_default = int(ssh_port_default)
    except (TypeError, ValueError):
        ssh_port_default = 22
    run_parent.add_argument("--ssh-port", type=int, default=ssh_port_default)

    ssh_attempts_default = _config_value(run_defaults, "ssh_attempts", 4)
    try:
        ssh_attempts_default = int(ssh_attempts_default)
    except (TypeError, ValueError):
        ssh_attempts_default = 4
    run_parent.add_argument("--ssh-attempts", type=int, default=ssh_attempts_default)

    run_parent.add_argument("--upload", action="append", default=[])
    remote_cmd_default = _config_value(run_defaults, "remote_cmd")
    remote_required = not bool(remote_cmd_default)
    run_parent.add_argument("--remote-cmd", default=remote_cmd_default, required=remote_required)
    run_parent.add_argument("--workdir", default=_config_value(run_defaults, "workdir", "~"))
    run_parent.add_argument("--download", action="append", default=[])

    run_parent.add_argument("--fail-fast", dest="fail_fast", action="store_true")
    run_parent.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    run_parent.set_defaults(fail_fast=_config_bool(run_defaults, "fail_fast", False))

    run_parent.add_argument("--s3-bucket", default=_config_value(run_defaults, "s3_bucket"))
    run_parent.add_argument("--s3-prefix", default=_config_value(run_defaults, "s3_prefix"))
    run_parent.add_argument("--s3-region", default=_config_value(run_defaults, "s3_region"))
    run_parent.add_argument("--s3-profile", default=_config_value(run_defaults, "s3_profile"))
    run_parent.add_argument("--s3-upload", action="append", default=[])
    run_parent.add_argument("--s3-download", action="append", default=[])

    launch = sub.add_parser("launch", parents=[launch_parent])
    launch.add_argument("--write-meta", default=str(DEFAULT_METADATA_PATH))
    launch.add_argument("--ssh-key-hint", default=_config_value(launch_defaults, "ssh_key_hint", "~/.ssh/botscalp-key.pem"))
    launch.add_argument("--ssh-user", default=_config_value(launch_defaults, "ssh_user", "ubuntu"))
    launch.set_defaults(func=_cmd_launch)

    run_cmd = sub.add_parser("run", parents=[run_parent])
    run_cmd.set_defaults(func=_cmd_run)

    term = sub.add_parser("terminate", help="Encerra a instância do metadata")
    term.add_argument("--meta", default=str(DEFAULT_METADATA_PATH))
    term.set_defaults(func=_cmd_terminate)

    full = sub.add_parser("fullrun", parents=[launch_parent, run_parent])
    full.add_argument("--terminate-on-finish", dest="terminate_on_finish", action="store_true")
    full.add_argument("--no-terminate-on-finish", dest="terminate_on_finish", action="store_false")
    full.set_defaults(terminate_on_finish=_config_bool(full_defaults, "terminate_on_finish", False))
    full.set_defaults(func=_cmd_fullrun)

    config_cmd = sub.add_parser("config", help="Guia interativo de configuração da CLI")
    config_cmd.add_argument("--reset", action="store_true", help="ignora valores salvos e começa do zero")
    config_cmd.add_argument("--show", action="store_true", help="exibe a configuração atual e sai")
    config_cmd.set_defaults(func=_cmd_config)

    return parser


def _run_new_cli(argv):
    parser = _build_new_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def entrypoint(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in NEW_COMMANDS:
        return _run_new_cli(argv)
    legacy_main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
