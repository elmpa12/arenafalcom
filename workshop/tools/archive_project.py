"""Utility to bundle repository files into a zip for external download."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile
from typing import Iterable, Iterator

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "datafull",
    "mergedata",
    "out",
    ".venv",
    "venv",
}

DEFAULT_EXCLUDED_PATTERNS = {
    ".pyc",
    ".pyo",
    ".pyd",
    "~",
    ".swp",
    ".zip",
    ".log",
}


def iter_repository_files(
    root: Path,
    *,
    exclude_dirs: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
) -> Iterator[Path]:
    """Yield project files that should be exported."""

    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_dirs.update(exclude_dirs)

    excluded_patterns = set(DEFAULT_EXCLUDED_PATTERNS)
    excluded_patterns.update(exclude_patterns)

    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories in-place
        dirnames[:] = [
            name
            for name in dirnames
            if name not in excluded_dirs and not name.startswith(".")
        ]

        for filename in filenames:
            if filename.startswith("."):
                continue
            if any(filename.endswith(pattern) for pattern in excluded_patterns):
                continue

            yield Path(dirpath, filename)


def create_archive(
    destination: Path,
    *,
    root: Path,
    files: Iterable[Path],
) -> None:
    """Create a zip archive with the given files relative to *root*."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file_path in files:
            relative = file_path.relative_to(root)
            bundle.write(file_path, arcname=str(relative))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("botscalpv3_export.zip"),
        help="Destination zip file path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to export.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Additional directory names to exclude.",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        metavar="SUFFIX",
        help="Additional filename suffixes to exclude (e.g. .csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    files = list(
        iter_repository_files(
            root,
            exclude_dirs=args.exclude_dir,
            exclude_patterns=args.exclude_pattern,
        )
    )

    if not files:
        raise SystemExit("No files found to export.")

    create_archive(args.output, root=root, files=files)
    print(f"Created archive with {len(files)} files: {args.output}")


if __name__ == "__main__":
    main()
