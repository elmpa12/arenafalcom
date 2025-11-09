"""Entry point helpers for selector workflows."""

from __future__ import annotations

import sys
from typing import Sequence

from core.selectors import selector21


def _forward(argv: Sequence[str]) -> None:
    """Run selector21 with the provided argv sequence."""

    sys.argv = ["botscalp-select", *argv]
    selector21.main()


def main(argv: Sequence[str] | None = None) -> None:
    """
    Dispatch the selector CLI.

    ``botscalp-select`` currently acts as a thin wrapper around ``selector21``.
    A future revision may expose extra subcommands, so we keep the indirection
    here instead of calling :func:`selector21.main` directly from the entrypoint.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "run":
        args = args[1:]
    _forward(args)
