# 🚀 QUICKSTART FOR NEXT AI

**TL;DR:** Read these 3 files (25 minutes total), then you're ready to code.

---

## 📖 READ IN THIS ORDER (25 min)

### 1. PIB_v1.md (15 min)
- **What:** Complete Project Intelligence Brief
- **Sections:** Architecture + Components + Dataflow + Runbook + Roadmap
- **Goal:** Understand full system in one document
- **Action:** Read end-to-end, pay attention to ASCII diagrams

### 2. JOURNAL.txt (5 min)
- **What:** Operational log with decision trail
- **Format:** Structured entries with timestamp | actor | phase | summary
- **Goal:** See what was done, why, and what's next
- **Action:** Read entries in order (latest at bottom)

### 3. NEXT_COMMIT.md (5 min)
- **What:** Exact commit-ready checklist + Phase 2 roadmap
- **Sections:** Git commands + Validation checklist + Next phases
- **Goal:** Know exactly what to commit and what comes next
- **Action:** Copy git commands if committing now, else just read roadmap

---

## 🎯 YOUR IMMEDIATE TASKS (in order)

### ✅ Already Done (verify works)
```bash
# Health check
curl https://bs3.falcomlabs.com/codex/health
# Expected: {"status":"ok"}

# Test BICHO mode
flabs "create simple trading bot"
# Expected: Production-ready code in 3-5 seconds

# Test CODEX mode  
flabs -c "design elite system"
# Expected: Innovative code with 10+ optimization suggestions
```

### 📋 Choose ONE (Phase 2 focus)

**Option A: Enhanced CODEX Module** (Recommended)
- Create: `/opt/botscalpv3/backend/analyze.py`
- Functions: `analyze_code_file()`, `suggest_optimizations()`, `generate_refactor_plan()`
- Use journal function: `format_journal_entry() + append_to_journal()`
- Integration: Modify flabs to call analyze.py on `-a` flag
- Commit msg: `"feat: Add CODEX code analysis module + optimization suggestions"`

**Option B: Auto-logging Integration**
- Modify: `/opt/botscalpv3/flabs`
- Add: Auto-capture of execution time, phase, files touched
- Call: `append_to_journal()` after success (use `format_journal_entry()`)
- Result: Every flabs invocation logs to JOURNAL.txt automatically
- Commit msg: `"feat: Auto-logging integration for flabs CLI"`

**Option C: Performance Dashboard**
- Create: `/opt/botscalpv3/dashboard/app.py` (Streamlit)
- Display: Last 20 JOURNAL entries + gateway uptime + API stats
- Run: `streamlit run dashboard/app.py`
- Commit msg: `"feat: Add Streamlit performance dashboard"`

---

## 🔑 KEY FUNCTIONS TO USE

### From `backend/system_prompts.py`:

```python
# Get system prompt for a mode
from backend.system_prompts import get_system_prompt
prompt = get_system_prompt("codex")  # or "bicho"

# Format a journal entry
from backend.system_prompts import format_journal_entry, append_to_journal

entry = format_journal_entry(
    actor="CODEX",
    phase="impl",
    summary="Built analyze.py module for code review",
    files_touched=["/opt/botscalpv3/backend/analyze.py"],
    decisions=["Used AST module for code parsing", "Integrated with OpenAI for suggestions"],
    todos={"PERF": ["Optimize parsing for large files"]},
    next_actions=["Test on real codebases", "Add refactoring recommendations"],
    artifacts=["analyze.py v1"],
    commit_msg="feat: Add CODEX code analysis module"
)

# Append to journal
append_to_journal(entry)
# Output: Appends to /opt/botscalpv3/JOURNAL.txt automatically
```

---

## 🌳 CURRENT ARCHITECTURE

```
User → flabs CLI → Nginx (SSL) → FastAPI Gateway → OpenAI API
         ↓                              ↓
    Payload with mode         system_prompts.py
    (mode: bicho|codex)       inject system prompt
```

**Two AI Modes:**
- 🐍 **BICHO** (default): Practical, production-ready code
- 🏆 **CODEX** (flag: -c): Innovative, championship-grade solutions

---

## ⚡ COMMON COMMANDS

```bash
# Test system
flabs "test"                    # BICHO mode
flabs -c "test"                # CODEX mode
flabs -f file.py "analyze"    # Analyze file (BICHO)
flabs -c -f file.py "refactor" # Analyze file (CODEX)

# Gateway management
curl https://bs3.falcomlabs.com/codex/health         # Health
curl https://bs3.falcomlabs.com/codex/api/models    # List models
ps aux | grep uvicorn                                # Check PID
pkill -f "uvicorn backend.openai_gateway"            # Stop
cd /opt/botscalpv3 && . ../.venv/bin/activate && nohup uvicorn ... # Restart

# Git
git add .
git commit -m "feat: [description]"
git push origin botscalpv3

# Python validation
python3 -m py_compile backend/system_prompts.py  # Syntax check
python3 -c "from backend.system_prompts import get_system_prompt; print('✅ OK')"
```

---

## 📞 IF YOU GET STUCK

1. **Gateway not responding?**
   - Check: `ps aux | grep uvicorn`
   - Restart: See "Gateway management" above
   - Logs: `tail -50 /tmp/gateway.log`

2. **Import errors?**
   - Syntax: `python3 -m py_compile backend/system_prompts.py`
   - Import: `python3 -c "from backend.system_prompts import ..."`

3. **JOURNAL.txt append failing?**
   - Check permissions: `ls -l JOURNAL.txt`
   - Test: `echo "test" >> JOURNAL.txt`

4. **Lost context?**
   - Re-read: PIB_v1.md (15 min to get back on track)

---

## 🎊 NEXT AI CHECKLIST

Before starting code:
- [ ] Read PIB_v1.md (15 min)
- [ ] Read JOURNAL.txt (5 min)
- [ ] Read NEXT_COMMIT.md (5 min)
- [ ] Run: `flabs "test"` ✅ Works?
- [ ] Run: `flabs -c "test"` ✅ Works?
- [ ] Run: `curl https://bs3.falcomlabs.com/codex/health` ✅ OK?
- [ ] Review: `backend/system_prompts.py` (especially new functions)
- [ ] Choose: One of 3 Phase 2 options (A, B, or C)
- [ ] Start coding!

---

**Questions?** Check PIB_v1.md section "Runbook" (12 operational procedures).

**Ready to commit?** Follow commands in NEXT_COMMIT.md.

**Ready to code?** Pick Option A/B/C above and go.

**Good luck! 🚀**
