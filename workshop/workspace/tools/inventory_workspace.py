#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
rows = []
for p in root.rglob("*"):
    if p.is_file():
        rows.append({"path": str(p.relative_to(root)), "size": p.stat().st_size})

out = Path("reports/workspace_inventory.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "size"])
    writer.writeheader()
    writer.writerows(rows)

print(f"[inventory] {out} — {len(rows)} files")
