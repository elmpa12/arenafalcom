"""Entry points for deep-learning utilities."""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Sequence

from core.dl import dl_heads_v8, treinar_dl, validar_dl

CommandFn = Callable[[Sequence[str]], None]


def _forward(handler: Callable[[], None], argv: Sequence[str], prog: str) -> None:
    sys.argv = [prog, *argv]
    handler()


def main(argv: Sequence[str] | None = None) -> None:
    """Multiplexer for the DL-related scripts."""

    args: List[str] = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "botscalp-dl commands:\n"
            "  train [args...]     -> treinar_dl.main()\n"
            "  validate [args...]  -> validar_dl.main()\n"
            "  heads [args...]     -> dl_heads_v8.main()\n"
        )
        return

    command = args.pop(0)
    if command == "train":
        _forward(treinar_dl.main, args, "botscalp-dl train")
    elif command == "validate":
        _forward(validar_dl.main, args, "botscalp-dl validate")
    elif command == "heads":
        _forward(dl_heads_v8.main, args, "botscalp-dl heads")
    else:
        print(f"Unknown botscalp-dl command '{command}'. Use --help to list options.")
        raise SystemExit(2)
