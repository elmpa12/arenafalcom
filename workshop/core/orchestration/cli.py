"""Entry point wrapper for orchestration workflows."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from core.orchestration import orchestrator
from core.orchestration.pipeline_runner import run_pipeline


def _run_pipeline_cmd(args: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="botscalp-orch pipeline")
    parser.add_argument("--config", required=True, help="Arquivo JSON/YAML descrevendo o pipeline.")
    ns = parser.parse_args(list(args))
    run_pipeline(ns.config)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Invoke the orchestration entrypoint or the new pipeline helper."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "pipeline":
        return _run_pipeline_cmd(args[1:])
    return orchestrator.entrypoint(args)
