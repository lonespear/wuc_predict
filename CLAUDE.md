# KC-135 WUC Maintenance Analytics — context for future sessions

## What this is

A unified Streamlit platform for KC-135 maintenance analytics. Synthesizes a WUC predictor (BERT classifier) and a maintenance-record query tool, plus a new WUC-profile + LLM-summary capability.

**Repo:** `github.com/lonespear/wuc_predict` (default branch: `main`, NOT `master`)

**Owner GitHub account:** `lonespear` (Windows machine has multiple accounts cached; auth as `lonespear` may need re-prompting via Git Credential Manager).

---

## File layout

| File | Role |
|---|---|
| `main_app.py` | **Entry point.** 3-tab Streamlit app (Predict / Query / Profile). |
| `app.py` | Original standalone WUC predictor (legacy — kept for reference). |
| `sum_app.py` | Original standalone query app (legacy — superseded by Tab 2). |
| `model_loader.py` | Loads `jonday/wuc-model` (BERT) at import time. CPU-only currently. |
| `wuc_profile.py` | Pure-pandas deterministic profile (why/when/where/lifecycle/co-occurrence). No ML. |
| `llm_adapter.py` | `SummaryAdapter` Protocol + 3 implementations: `NullAdapter` (template, offline), `GemmaAdapter` (Ollama, streaming), `ClaudeAdapter` (Anthropic API). |
| `data_config.py` | Path resolution + `WHEN_DISCOVERED_PHASE` / `TYPE_MAINT_PHASE` code dicts. |
| `sum_utils.py` | NL-query parser + record analysis (used by Tab 2). |
| `requirements.txt` | streamlit, torch, transformers, pandas, matplotlib, ollama, anthropic |
| `.gitignore` | Excludes `__pycache__/`, `.venv/`, `FinalData.csv`, `kc135_wuc_lookup_levels.csv`, `*.parquet` |

**Data files (NOT committed — sensitivity):**
- `FinalData.csv` — maintenance records (the main dataset)
- `kc135_wuc_lookup_levels.csv` — teammate's enriched lookup (optional; `_dictionary.csv` is the committed fallback)

---

## Architecture

### Tab 1 — Predict WUC
- Wraps `model_loader.predict_discrepancy(text, method=2)`.
- Single-text prediction; CSV batch upload still lives in legacy `app.py` and was NOT carried into the unified app (could be re-added).
- Sets `st.session_state["predicted_wuc"]` so user can jump to Tab 3 with the WUC pre-filled.

### Tab 2 — Query Records
- `sum_utils.parse_user_query()` regex-parses the question (tail number, WUC, date ranges).
- `sum_utils.analyze_results()` filters + summarizes (top discrepancies/fixes, monthly histogram, top WUCs).
- Renders text + matplotlib bar chart + ranked table.

### Tab 3 — WUC Profile (new — the synthesis value-add)
- `wuc_profile.build_profile(df, wuc, desc_map)` produces a structured dict — purely deterministic from the data.
- Pluggable summarizer chosen from a dropdown (`available_adapters()` filters to whatever has its dependencies).
- Renders: streaming narrative + metric cards + bar charts (seasonality, year-over-year, base distribution) + tables (lifecycle, discovery phase, top phrases, co-occurring WUCs).

---

## LLM summarization

**Single shared prompt** in `llm_adapter.py` → `ANALYST_PROMPT`:
```
You are a KC-135 maintenance analyst. Given the structured profile
below, write a concise report (4-6 short paragraphs) answering:
(1) why this WUC occurs, (2) when seasonally and which years,
(3) where (which bases), (4) at what point in the airframe
lifecycle and discovery phase. Use ONLY numbers present in the
profile — do not invent figures. Quote representative discrepancy
text verbatim where helpful.

PROFILE:
{json-serialized profile dict}
```

**Why constraints work:**
- `temperature=0.3` (Gemma adapter) keeps it sober and data-faithful.
- "Use ONLY numbers present in the profile" prevents hallucinated figures.
- "Quote representative discrepancy text verbatim" produces the verbatim quote pattern in outputs.

**Adapters:**
- `NullAdapter` — offline template, deterministic, enterprise-safe (no network). Always available.
- `GemmaAdapter` — local Ollama. Default model `gemma4:e4b`. Implements `summarize_stream()` (yields chunks) for the progressive-render UX. Drops out of the dropdown if Ollama daemon unreachable.
- `ClaudeAdapter` — Anthropic API. Activates only if `ANTHROPIC_API_KEY` env var is set.

**Adding a new adapter:** define a class with `name`, `available()`, `summarize(profile)`, optionally `summarize_stream(profile)`. Add to `available_adapters()` candidate list. Done — no other file changes needed.

---

## Deployment — USMA dockerized JupyterHub GPU box

Container: `jovyan@bb2053806b66` (non-root). RTX 6000 Ada, 48 GB VRAM, CUDA 12.8.

### One-time setup
```bash
# Ollama install (no sudo, no zstd CLI needed — uses Python zstandard)
curl -L https://github.com/ollama/ollama/releases/download/v0.22.0/ollama-linux-amd64.tar.zst \
  -o /tmp/ollama.tar.zst
pip install --user zstandard
python -c "import zstandard, tarfile; tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(open('/tmp/ollama.tar.zst','rb')), mode='r|').extractall('/home/jovyan/.local/'); print('ok')"
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

# Python deps
pip install --user -r requirements.txt

# Model pull
nohup ollama serve > ~/ollama.log 2>&1 &
sleep 3
ollama pull gemma4:e4b
# Optional upgrade given 48 GB VRAM headroom:
# ollama pull gemma4:26b-a4b
```

### Launch (every session)
```bash
# Ensure ollama is running
pgrep -f "ollama serve" > /dev/null || nohup ollama serve > ~/ollama.log 2>&1 &

# Start streamlit (no baseUrlPath — jupyter-server-proxy strips the prefix)
pkill -9 -f streamlit ; sleep 2
nohup streamlit run main_app.py \
  --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --server.enableCORS false \
  --server.enableXsrfProtection false --browser.gatherUsageStats false \
  > ~/streamlit.log 2>&1 &
```

### Access URL
```
https://icsarl.westpoint.edu/jupyter-cdas2/user/jonathan.day/proxy/8501/
```
Trailing slash matters. `jupyter_server_proxy` 4.1.2 strips the prefix, so do NOT set `--server.baseUrlPath`.

---

## Open follow-ups (ranked)

1. **Model-size dropdown** in Tab 3 — flip `gemma4:e2b` ↔ `e4b` ↔ `26b-a4b` from the UI without restart.
2. **Prompt-style selector** — "Maintenance brief / Engineering analysis / Executive summary" — three named templates instead of one shared `ANALYST_PROMPT`.
3. **Move BERT to GPU** — `model_loader.py` currently runs on CPU. Two-line patch: `model.to('cuda')` + `inputs.to('cuda')`. Will be milliseconds per inference on the RTX 6000.
4. **CSV batch prediction in Tab 1** — wasn't carried over from the legacy `app.py`. Easy to port.
5. **Bump Claude default** in `ClaudeAdapter` from `claude-opus-4-6` → `claude-opus-4-7`.
6. **Recommendations step** in the prompt — append "Propose 2-3 prioritized maintenance/supply-chain actions justified by the data."
7. **Verify `kc135_wuc_lookup_levels.csv`** vs the committed `_dictionary.csv` — teammate's file may have richer columns. Currently the unified app falls back to the dictionary file (which has columns `wuc`, `description`, `full_context`).

---

## Gotchas / lessons learned this session

- **Auth on Windows multi-account:** the machine had `usma-stats` cached as the GitHub credential. Pushing to a `lonespear` repo required GCM re-prompt via the system browser. `cmdkey /delete:git:https://github.com` didn't find the entry — credentials were stored under a different key name. Worked after manual re-auth.
- **Streamlit + JupyterHub at a non-root path:** `jupyter_server_proxy` 4.x strips the prefix before forwarding, so `--server.baseUrlPath` should NOT be set even though the proxy URL is deeply nested (`/jupyter-cdas2/user/jonathan.day/proxy/8501/`).
- **No sudo, no conda, no zstd CLI:** the dockerized container is minimal. Path forward was Python `zstandard` + `tarfile` for streaming-extraction. Avoid any plan that needs system packages.
- **Ollama latest releases ship `.tar.zst` only** (no `.tgz` fallback). Manual install bypasses the install script's zstd dependency entirely.
- **Terminal auto-indents pasted heredocs** in this user's shell — heredocs with a `<<EOF` opener get bricked because the closing `EOF` ends up indented. Use single-line `python -c "..."` with semicolons instead.

---

## What's been pushed vs working-tree

As of session end (`a090e77` — "Add unified analytics platform with WUC profiling and pluggable LLM summarization"):
- Pushed: `main_app.py`, `wuc_profile.py`, `llm_adapter.py`, `data_config.py`, `.gitignore`, `requirements.txt` modification.
- Local-only: `__pycache__/` (ignored), data CSVs (intentionally never committed).
