#!/usr/bin/env python3
import json, sys, subprocess, time
from pathlib import Path

config_file = sys.argv[1] if len(sys.argv) > 1 else "ultra_fast_tests_config.json"
parallel = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--parallel" else 15

with open(config_file) as f:
    tests = json.load(f)

print(f"🚀 Rodando {len(tests)} testes ({parallel} paralelos)")

running, pending = [], list(tests)
completed = []

while pending or running:
    while len(running) < parallel and pending:
        test = pending.pop(0)
        Path(test["args"][test["args"].index("--out_root")+1]).mkdir(parents=True, exist_ok=True)
        log = f"{test['args'][test['args'].index('--out_root')+1]}/test.log"
        proc = subprocess.Popen(
            ["python3", "-m", "core.selectors.selector21"] + test["args"],
            stdout=open(log, "w"), stderr=subprocess.STDOUT
        )
        running.append({"name": test["name"], "proc": proc, "start": time.time()})
        print(f"[{len(completed)+len(running)}/{len(tests)}] ▶️  {test['name']}")
    
    time.sleep(2)
    done = [r for r in running if r["proc"].poll() is not None]
    for r in done:
        elapsed = time.time() - r["start"]
        status = "✅" if r["proc"].returncode == 0 else "❌"
        print(f"{status} {r['name']} ({elapsed:.1f}s)")
        running.remove(r)
        completed.append(r)

print(f"\n✅ {len(completed)} testes completados!")
