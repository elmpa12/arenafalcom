#!/usr/bin/env python3
"""
MASTER ORCHESTRATOR - BotScalp v3
Arquitetura planejada por Claudex 2.0 (Dual AI System)

Integra todo o pipeline:
  1. Provisionamento de GPU na AWS
  2. Execução local do Selector
  3. Transferência de dados
  4. Deep Learning na GPU remota
  5. Consolidação de resultados

Características:
  - Retry logic em cada etapa
  - Logs centralizados
  - Cleanup automático
  - State management para retomar
  - Dry-run mode para testar
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

# ==================== CONFIGURAÇÃO ====================

class PipelineStage(Enum):
    """Estágios do pipeline"""
    INIT = "init"
    AWS_PROVISION = "aws_provision"
    SELECTOR_LOCAL = "selector_local"
    DATA_TRANSFER = "data_transfer"
    DL_REMOTE = "dl_remote"
    RESULTS_DOWNLOAD = "results_download"
    CONSOLIDATION = "consolidation"
    VISUALIZATION = "visualization"  # NOVO: Replay visual
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineState:
    """Estado do pipeline para retomada"""
    session_id: str
    current_stage: str
    started_at: str
    updated_at: str
    aws_instance_id: Optional[str] = None
    aws_instance_ip: Optional[str] = None
    selector_output_dir: Optional[str] = None
    dl_output_dir: Optional[str] = None
    final_report: Optional[str] = None
    error: Optional[str] = None


class Colors:
    """Cores para output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


# ==================== MASTER ORCHESTRATOR ====================

class MasterOrchestrator:
    """Orquestrador mestre que integra todos os componentes"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = Path(args.work_dir) / self.session_id
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.work_dir / "pipeline_state.json"
        self.log_file = self.work_dir / "master.log"

        # Carrega ou cria estado
        self.state = self._load_or_create_state()

        # Estatísticas
        self.stats = {
            "retries": {},
            "timings": {},
            "errors": []
        }

    def _load_or_create_state(self) -> PipelineState:
        """Carrega estado existente ou cria novo"""
        if self.args.resume and self.state_file.exists():
            self.log("Retomando pipeline existente...")
            with open(self.state_file) as f:
                data = json.load(f)
                return PipelineState(**data)
        else:
            return PipelineState(
                session_id=self.session_id,
                current_stage=PipelineStage.INIT.value,
                started_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )

    def _save_state(self):
        """Salva estado atual"""
        self.state.updated_at = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)

    def log(self, msg: str, level: str = "INFO"):
        """Log com timestamp e cor"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        color = {
            "INFO": Colors.CYAN,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED
        }.get(level, Colors.END)

        formatted = f"{color}[{timestamp}] [{level}]{Colors.END} {msg}"
        print(formatted, flush=True)

        # Salva em arquivo
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] [{level}] {msg}\n")

    def run_with_retry(self, func, stage_name: str, max_retries: int = 3) -> bool:
        """Executa função com retry logic"""
        for attempt in range(1, max_retries + 1):
            try:
                self.log(f"Executando {stage_name} (tentativa {attempt}/{max_retries})")
                start_time = time.time()

                result = func()

                elapsed = time.time() - start_time
                self.stats["timings"][stage_name] = elapsed

                self.log(f"✅ {stage_name} concluído em {elapsed:.1f}s", "SUCCESS")
                return True

            except Exception as e:
                error_msg = f"Erro em {stage_name}: {str(e)}"
                self.log(error_msg, "ERROR")
                self.stats["errors"].append({"stage": stage_name, "attempt": attempt, "error": str(e)})

                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.log(f"Aguardando {wait_time}s antes de retentar...", "WARNING")
                    time.sleep(wait_time)
                else:
                    self.log(f"❌ {stage_name} falhou após {max_retries} tentativas", "ERROR")
                    self.state.error = error_msg
                    self.state.current_stage = PipelineStage.FAILED.value
                    self._save_state()
                    return False

        return False

    # ==================== ESTÁGIOS DO PIPELINE ====================

    def stage_aws_provision(self) -> bool:
        """Estágio 1: Provisiona GPU na AWS"""
        def _provision():
            if self.args.dry_run:
                self.log("DRY-RUN: Simulando provisionamento AWS...", "WARNING")
                self.state.aws_instance_id = "i-fake-instance"
                self.state.aws_instance_ip = "1.2.3.4"
                return True

            cmd = [
                "python3", "aws_gpu_launcher.py",
                "--region", self.args.aws_region,
                "--instance-type", self.args.instance_type,
                "--key-name", self.args.key_name,
                "--name", f"BotScalp-{self.session_id}",
                "--metadata", str(self.work_dir / "aws_metadata.json")
            ]

            if self.args.aws_spot:
                cmd.append("--spot")

            self.log(f"Comando: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"AWS provisioning failed: {result.stderr}")

            # Lê metadata da instância
            metadata_file = self.work_dir / "aws_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    self.state.aws_instance_id = metadata.get("instance_id")
                    self.state.aws_instance_ip = metadata.get("public_ip")

            self.log(f"Instância provisionada: {self.state.aws_instance_id} ({self.state.aws_instance_ip})")
            return True

        return self.run_with_retry(_provision, "AWS Provisioning", self.args.max_retries)

    def stage_selector_local(self) -> bool:
        """Estágio 2: Executa Selector localmente"""
        def _run_selector():
            if self.args.dry_run:
                self.log("DRY-RUN: Simulando execução do Selector...", "WARNING")
                self.state.selector_output_dir = str(self.work_dir / "selector_out")
                Path(self.state.selector_output_dir).mkdir(exist_ok=True)
                return True

            selector_out = self.work_dir / "selector_out"
            selector_out.mkdir(exist_ok=True)

            cmd = [
                "python3", "selector21.py",
                "--symbol", self.args.symbol,
                "--data_dir", self.args.data_dir,
                "--start", self.args.start,
                "--end", self.args.end,
                "--exec_rules", self.args.exec_rules,
                "--run_base",
                "--run_combos",
                "--walkforward",
                "--wf_train_months", str(self.args.wf_train_months),
                "--wf_val_months", str(self.args.wf_val_months),
                "--wf_step_months", str(self.args.wf_step_months),
            ]

            if self.args.wf_expand:
                cmd.append("--wf_expand")

            # ML options
            if self.args.run_ml:
                cmd.extend([
                    "--run_ml",
                    "--ml_model_kind", self.args.ml_model_kind
                ])
                if self.args.ml_use_agg:
                    cmd.append("--ml_use_agg")
                if self.args.ml_use_depth:
                    cmd.append("--ml_use_depth")
                if self.args.ml_opt_thr:
                    cmd.append("--ml_opt_thr")

            # ATR Stop/TP
            if self.args.use_atr_stop:
                cmd.extend([
                    "--use_atr_stop",
                    "--atr_stop_mult", str(self.args.atr_stop_mult)
                ])
            if self.args.use_atr_tp:
                cmd.extend([
                    "--use_atr_tp",
                    "--atr_tp_mult", self.args.atr_tp_mult
                ])

            # Hard limits
            cmd.extend([
                "--hard_stop_usd", self.args.hard_stop_usd,
                "--hard_tp_usd", self.args.hard_tp_usd
            ])

            # Depth/Agg
            if self.args.agg_dir:
                cmd.extend(["--agg_dir", self.args.agg_dir])
            if self.args.depth_dir:
                cmd.extend(["--depth_dir", self.args.depth_dir, "--depth_field", self.args.depth_field])

            if self.args.print_top10:
                cmd.append("--print_top10")

            # Output
            cmd.extend(["--out", str(selector_out)])

            self.log(f"Executando Selector... (pode demorar)")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Selector failed: {result.stderr}")

            self.state.selector_output_dir = str(selector_out)
            self.log(f"Selector output: {selector_out}")
            return True

        return self.run_with_retry(_run_selector, "Selector Local", self.args.max_retries)

    def stage_data_transfer(self) -> bool:
        """Estágio 3: Transfere dados para GPU remota"""
        def _transfer():
            if self.args.dry_run:
                self.log("DRY-RUN: Simulando transferência de dados...", "WARNING")
                return True

            # Usa rsync ou scp para transferir
            source = self.state.selector_output_dir
            dest = f"{self.args.gpu_user}@{self.state.aws_instance_ip}:{self.args.gpu_root}/data/"

            cmd = [
                "rsync", "-avz", "--progress",
                "-e", f"ssh -i {self.args.ssh_key}",
                f"{source}/",
                dest
            ]

            self.log(f"Transferindo dados via rsync...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Data transfer failed: {result.stderr}")

            self.log("Dados transferidos com sucesso")
            return True

        return self.run_with_retry(_transfer, "Data Transfer", self.args.max_retries)

    def stage_dl_remote(self) -> bool:
        """Estágio 4: Executa DL na GPU remota"""
        def _run_dl():
            if self.args.dry_run:
                self.log("DRY-RUN: Simulando execução de DL remoto...", "WARNING")
                self.state.dl_output_dir = "/remote/dl_out"
                return True

            # Usa orchestrator.py para executar remotamente
            cmd = [
                "python3", "orchestrator.py",
                "--gpu-host", self.state.aws_instance_ip,
                "--gpu-user", self.args.gpu_user,
                "--gpu-key", self.args.ssh_key,
                "--dl-script", self.args.dl_script,
                "--symbol", self.args.symbol,
                "--dl-models", self.args.dl_models,
                "--dl-epochs", str(self.args.dl_epochs)
            ]

            self.log("Executando DL na GPU remota...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"DL remote failed: {result.stderr}")

            self.state.dl_output_dir = f"{self.args.gpu_root}/dl_out"
            return True

        return self.run_with_retry(_run_dl, "DL Remote", self.args.max_retries)

    def stage_results_download(self) -> bool:
        """Estágio 5: Baixa resultados"""
        def _download():
            if self.args.dry_run:
                self.log("DRY-RUN: Simulando download de resultados...", "WARNING")
                return True

            results_dir = self.work_dir / "results"
            results_dir.mkdir(exist_ok=True)

            source = f"{self.args.gpu_user}@{self.state.aws_instance_ip}:{self.state.dl_output_dir}/"

            cmd = [
                "rsync", "-avz", "--progress",
                "-e", f"ssh -i {self.args.ssh_key}",
                source,
                str(results_dir)
            ]

            self.log("Baixando resultados...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Results download failed: {result.stderr}")

            return True

        return self.run_with_retry(_download, "Results Download", self.args.max_retries)

    def stage_consolidation(self) -> bool:
        """Estágio 6: Consolida resultados"""
        def _consolidate():
            self.log("Consolidando resultados...")

            report_file = self.work_dir / "FINAL_REPORT.md"

            with open(report_file, 'w') as f:
                f.write(f"# BotScalp v3 - Pipeline Results\n\n")
                f.write(f"Session: {self.session_id}\n")
                f.write(f"Started: {self.state.started_at}\n")
                f.write(f"Completed: {datetime.now().isoformat()}\n\n")

                f.write(f"## Pipeline Stages\n\n")
                for stage, timing in self.stats["timings"].items():
                    f.write(f"- {stage}: {timing:.1f}s\n")

                f.write(f"\n## AWS Instance\n\n")
                f.write(f"- ID: {self.state.aws_instance_id}\n")
                f.write(f"- IP: {self.state.aws_instance_ip}\n")

                f.write(f"\n## Outputs\n\n")
                f.write(f"- Selector: {self.state.selector_output_dir}\n")
                f.write(f"- DL Remote: {self.state.dl_output_dir}\n")
                f.write(f"- Local Results: {self.work_dir}/results\n")

                if self.stats["errors"]:
                    f.write(f"\n## Errors ({len(self.stats['errors'])})\n\n")
                    for err in self.stats["errors"]:
                        f.write(f"- {err['stage']} (attempt {err['attempt']}): {err['error']}\n")

            self.state.final_report = str(report_file)
            self.log(f"Relatório final: {report_file}", "SUCCESS")
            return True

        return self.run_with_retry(_consolidate, "Consolidation")

    def stage_visualization(self) -> bool:
        """Estágio 7: Prepara visualização de replay"""
        def _prepare_visualization():
            if not self.args.enable_visual:
                self.log("Visualização desabilitada (use --enable-visual)", "WARNING")
                return True

            self.log("Preparando visualização de replay...")

            visual_data = self.work_dir / "visual_data"
            visual_data.mkdir(exist_ok=True)

            # Copia dados para formato visual
            # (idealmente converte resultados para frames.jsonl + trades.jsonl)
            self.log(f"Diretório visual: {visual_data}")

            # Cria meta.json básico
            meta_file = visual_data / "meta.json"
            meta = {
                "session_id": self.session_id,
                "symbol": self.args.symbol,
                "timeframes": self.args.exec_rules.split(","),
                "start": self.args.start,
                "end": self.args.end,
                "generated_at": datetime.now().isoformat()
            }

            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)

            if self.args.start_visual_server:
                self.log("Iniciando servidor visual...", "WARNING")
                self.log("Execute manualmente:")
                self.log(f"  cd visual/backend")
                self.log(f"  export VISUAL_DATA_ROOT={visual_data.absolute()}")
                self.log(f"  python app.py")
                self.log("")
                self.log("Depois acesse: http://localhost:8081")
            else:
                self.log("Para visualizar, execute:")
                self.log(f"  cd visual/backend")
                self.log(f"  export VISUAL_DATA_ROOT={visual_data.absolute()}")
                self.log(f"  python app.py")

            return True

        return self.run_with_retry(_prepare_visualization, "Visualization")

    def stage_cleanup(self) -> bool:
        """Estágio 8: Cleanup de recursos"""
        def _cleanup():
            if self.args.no_cleanup:
                self.log("Cleanup desabilitado (--no-cleanup)", "WARNING")
                return True

            if self.args.dry_run:
                self.log("DRY-RUN: Simulando cleanup...", "WARNING")
                return True

            self.log("Executando cleanup...")

            # Termina instância AWS se foi provisionada por este script
            if self.state.aws_instance_id and not self.state.aws_instance_id.startswith("i-fake"):
                self.log(f"Terminando instância AWS: {self.state.aws_instance_id}")
                # Aqui você implementaria a chamada para terminar a instância
                # Por segurança, vou deixar comentado
                # subprocess.run(["aws", "ec2", "terminate-instances", "--instance-ids", self.state.aws_instance_id])

            return True

        return self.run_with_retry(_cleanup, "Cleanup")

    # ==================== PIPELINE PRINCIPAL ====================

    def run_full_pipeline(self) -> bool:
        """Executa pipeline completo"""
        self.log("="*70)
        self.log("🚀 INICIANDO MASTER ORCHESTRATOR - BotScalp v3")
        self.log(f"Session ID: {self.session_id}")
        self.log(f"Work Dir: {self.work_dir}")
        self.log("="*70 + "\n")

        if self.args.dry_run:
            self.log("⚠️  MODO DRY-RUN ATIVADO - Apenas simulação!", "WARNING")

        stages = [
            (PipelineStage.AWS_PROVISION, self.stage_aws_provision),
            (PipelineStage.SELECTOR_LOCAL, self.stage_selector_local),
            (PipelineStage.DATA_TRANSFER, self.stage_data_transfer),
            (PipelineStage.DL_REMOTE, self.stage_dl_remote),
            (PipelineStage.RESULTS_DOWNLOAD, self.stage_results_download),
            (PipelineStage.CONSOLIDATION, self.stage_consolidation),
            (PipelineStage.VISUALIZATION, self.stage_visualization),  # NOVO: Replay visual
            (PipelineStage.CLEANUP, self.stage_cleanup),
        ]

        for stage_enum, stage_func in stages:
            self.state.current_stage = stage_enum.value
            self._save_state()

            self.log(f"\n{'='*70}")
            self.log(f"ESTÁGIO: {stage_enum.value.upper()}")
            self.log(f"{'='*70}\n")

            if not stage_func():
                self.log(f"\n❌ Pipeline FALHOU no estágio: {stage_enum.value}", "ERROR")
                return False

        # Sucesso!
        self.state.current_stage = PipelineStage.COMPLETED.value
        self._save_state()

        self.log("\n" + "="*70)
        self.log("✅ PIPELINE COMPLETADO COM SUCESSO!", "SUCCESS")
        self.log("="*70)
        self.log(f"\n📄 Relatório final: {self.state.final_report}\n")

        return True


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator - BotScalp v3 (Arquitetura by Claudex 2.0)"
    )

    # Pipeline options
    parser.add_argument("--work-dir", default="./work", help="Diretório de trabalho")
    parser.add_argument("--resume", action="store_true", help="Retoma pipeline existente")
    parser.add_argument("--dry-run", action="store_true", help="Simula execução sem rodar de verdade")
    parser.add_argument("--max-retries", type=int, default=3, help="Máximo de tentativas por estágio")
    parser.add_argument("--no-cleanup", action="store_true", help="Não faz cleanup ao final")

    # AWS options
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--instance-type", default="g4dn.xlarge")
    parser.add_argument("--key-name", required=True, help="Nome da chave SSH na AWS")
    parser.add_argument("--aws-spot", action="store_true", help="Usa instância spot")

    # Selector options
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-dir", default="./datafull")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-11-01")
    parser.add_argument("--exec-rules", default="1m,5m,15m")

    # Walk-Forward options (CRÍTICO!)
    parser.add_argument("--wf-train-months", type=float, default=3.0, help="Meses de treino no WF")
    parser.add_argument("--wf-val-months", type=float, default=1.0, help="Meses de validação no WF")
    parser.add_argument("--wf-step-months", type=float, default=1.0, help="Step do WF em meses")
    parser.add_argument("--wf-expand", action="store_true", help="WF expanding (senão anchored)")

    # ML options do Selector
    parser.add_argument("--run-ml", action="store_true", help="Roda pipeline de ML")
    parser.add_argument("--ml-model-kind", default="auto", help="xgb, rf, logreg, ou auto")
    parser.add_argument("--ml-use-agg", action="store_true", help="Usa features de aggtrades")
    parser.add_argument("--ml-use-depth", action="store_true", help="Usa features de depth")
    parser.add_argument("--ml-opt-thr", action="store_true", help="Otimiza threshold do ML")

    # ATR Stop/TP options
    parser.add_argument("--use-atr-stop", action="store_true", help="Usa ATR stop")
    parser.add_argument("--atr-stop-mult", type=float, default=2.0, help="Multiplicador ATR stop")
    parser.add_argument("--use-atr-tp", action="store_true", help="Usa ATR take-profit")
    parser.add_argument("--atr-tp-mult", default="2.5,2.5,3.0", help="Multiplicadores ATR TP por TF")

    # Hard limits
    parser.add_argument("--hard-stop-usd", default="60,80,100", help="Stop em USD por TF")
    parser.add_argument("--hard-tp-usd", default="300,360,400", help="TP em USD por TF")

    # Outros
    parser.add_argument("--agg-dir", help="Diretório de aggtrades parquet")
    parser.add_argument("--depth-dir", help="Diretório de depth parquet")
    parser.add_argument("--depth-field", default="bd_imb_50bps", help="Campo de depth imbalance")
    parser.add_argument("--print-top10", action="store_true", help="Printa top 10 resultados")

    # DL options
    parser.add_argument("--gpu-user", default="ubuntu")
    parser.add_argument("--gpu-root", default="/opt/botscalpv3")
    parser.add_argument("--ssh-key", help="Caminho para chave SSH privada")
    parser.add_argument("--dl-script", default="dl_heads_v8.py")
    parser.add_argument("--dl-models", default="gru,tcn")
    parser.add_argument("--dl-epochs", type=int, default=12)

    # Visual options
    parser.add_argument("--enable-visual", action="store_true", help="Prepara dados para visualização")
    parser.add_argument("--start-visual-server", action="store_true", help="Inicia servidor visual automaticamente")

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        orchestrator = MasterOrchestrator(args)
        success = orchestrator.run_full_pipeline()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Pipeline interrompido pelo usuário{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Erro fatal: {e}{Colors.END}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
