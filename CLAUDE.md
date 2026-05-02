# KC-135 WUC Maintenance Analytics — context for future sessions

> Read this first when picking up the project. `README.md` has the full
> architecture + flow charts; this file has the **mutable state** — what's
> currently shipped, what's the latest decision, what to do next.

## What this is

A unified Streamlit platform for KC-135 maintenance analytics. Three tabs:
WUC predictor (ModernBERT-large), maintenance-record query, WUC profile
narrated by Gemma 4 via local Ollama.

**Repo:** `github.com/lonespear/wuc_predict` (default branch `main`, NOT `master`)
**Owner GitHub account:** `lonespear` — Windows machine has multiple accounts
cached; pushing may need GCM re-prompt as `lonespear`.

---

## Status (2026-05-02)

**🚀 Shipped and live** at:
```
https://icsarl.westpoint.edu/jupyter-cdas2/user/jonathan.day/proxy/8501/
```

Currently running:
- ModernBERT-large hierarchical fine-tune (`./wuc-model-hier`) for Tab 1
- Gemma 4 (`gemma4:e4b`) via local Ollama for Tab 3
- Streamlit pointed at `WUC_MODEL_PATH=./wuc-model-hier`

---

## Trained models on disk (school GPU box)

| Model | Test acc | Macro F1 | Test loss | Status |
|---|---|---|---|---|
| `wuc-model-v2` (flat) | 0.904 | 0.772 | 1.035 | superseded |
| `wuc-model-v2-extended` (10 ep) | 0.906 | 0.771 | 1.290 | superseded (overfit) |
| **`wuc-model-hier`** | **0.903** | **0.772** | **0.555** | **🚀 deployed** |

Tied on macro F1, but hierarchical has 47% lower test loss → much better
calibrated. That calibration win is what makes the top-3 + confidence-band UX
honest.

---

## Critical workflow decision

**The model is deployed for the post-fix verification workflow** — maintainer
has done the work, types both discrepancy AND corrective action, model fills
the WUC slot. It's NOT a live pre-fix predictor.

Training input format: `<discrepancy> [SEP] <corrective_action> [SEP] <wce_narrative> [SEP] <how_mal> [SEP] <action_taken>`

Inference at deployment time: same format, but only discrepancy + corrective
action are required (other fields if available).

**Do NOT feed it informal lowercase pre-fix descriptions** like
`"seatbelt is frayed"` — model expects maintenance-report style (UPPERCASE,
terse, technical) like `PILOT SEAT BELT FRAYED, MISSING STITCHING`.

For pre-fix / live prediction, train a **discrepancy-only model variant** —
that's an open follow-up, not done yet.

---

## Smoke test that validated deployment

| Field | Value |
|---|---|
| Discrepancy input | `PILOT SEAT BELT FRAYED, MISSING STITCHING` |
| Corrective action | `INSPECTED PILOT SEAT BELT, REPLACED IAW TM 1C-135-06` |
| Top-1 prediction | **`12AAN — FUSELAGE COMPARTMENTS / Safety Belt`** at 76.8% |

`12AAN` is literally "Safety Belt" in the WUC dictionary. **Correct answer.**
The lower-vs-old confidence is a calibration improvement, not a regression
(class-weighted CE + hierarchical regularization deliberately damp
overconfidence). See `compare_models.py` for empirical head-to-head if anyone
doubts it.

---

## File layout (current)

| File | Role |
|---|---|
| `main_app.py` | **Entry point.** 3-tab Streamlit app. |
| `model_loader.py` | Reads `WUC_MODEL_PATH` env var; `predict_top_k(text, k=3)` + `build_input_text(discrepancy, action)`. Auto-reads `id2label` from `model.config`. |
| `wuc_profile.py` | Pure-pandas deterministic profile (why/when/where/lifecycle). |
| `llm_adapter.py` | `SummaryAdapter` Protocol + `NullAdapter`/`GemmaAdapter`/`ClaudeAdapter`. Shared `ANALYST_PROMPT`. |
| `sum_utils.py` | NL-query parser + record analysis (Tab 2). |
| `data_config.py` | Path resolution + WHEN_DISCOVERED / TYPE_MAINT code dicts. |
| `prepare_data.py` | Merge raw extracts → train/val/test parquet splits. |
| `train_fresh.py` | Fresh fine-tune (single classifier head). |
| `train_continue.py` | Continue from existing checkpoint with reset optimizer. |
| `train_hierarchical.py` | Joint system/subsystem/WUC fine-tune; **produces the shipped model**. |
| `compare_models.py` | Head-to-head old vs new on test set (accuracy + calibration). |
| `app.py` / `sum_app.py` | Legacy standalone apps; kept for reference, NOT used by main_app. |
| `README.md` | Public-facing architecture + flow charts. |
| `CLAUDE.md` | ← this file (mutable state). |

**Gitignored — never committed:**
- `FinalData.csv`, `new_data.csv`, `kc135_wuc_lookup_levels.csv` (CUI data)
- `data_splits/`, `wuc-model-v2/`, `wuc-model-v2-extended/`, `wuc-model-hier/`

---

## Tab 1 UX (current)

Two text fields side-by-side:
- **Discrepancy** (required)
- **Corrective Action** (optional but improves accuracy substantially)

Joined with `[SEP]` → `predict_top_k(text, k=3)`.

Top-1 displayed with **confidence-band coloring**:

| Confidence | Display |
|---|---|
| ≥70% | 🟢 Green success |
| 30-70% | 🟡 Yellow warning ("review alternatives") |
| <30% | 🔴 Red error ("likely OOD input") |

Top-2 and top-3 always shown below as bullet points under "Other candidates".

---

## Restart (every session, after `git pull`)

```bash
# Ensure ollama daemon is up
pgrep -f "ollama serve" > /dev/null || nohup ollama serve > ~/ollama.log 2>&1 &

# Restart streamlit pointing at the local hier model
pkill -9 -f streamlit ; sleep 2
WUC_MODEL_PATH=./wuc-model-hier nohup streamlit run main_app.py \
  --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --server.enableCORS false \
  --server.enableXsrfProtection false --browser.gatherUsageStats false \
  > ~/streamlit.log 2>&1 &
```

**`WUC_MODEL_PATH` is REQUIRED.** Without it, `model_loader.py` falls back to
`jonday/wuc-model` (legacy BERT-base on HF) — that's NOT the shipped model.

---

## One-time setup (already done on the box, but here for restoring)

```bash
# Ollama in user space (no sudo, no zstd CLI required)
curl -L https://github.com/ollama/ollama/releases/download/v0.22.0/ollama-linux-amd64.tar.zst \
  -o /tmp/ollama.tar.zst
pip install --user zstandard
python -c "import zstandard, tarfile; tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(open('/tmp/ollama.tar.zst','rb')), mode='r|').extractall('/home/jovyan/.local/'); print('ok')"
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

# Python deps
pip install --user -r requirements.txt

# Model
nohup ollama serve > ~/ollama.log 2>&1 &
sleep 3 && ollama pull gemma4:e4b
```

---

## Sensitivity & deployment policy

**Training data is treated as CUI.** Implications baked into the project:

- Data CSVs are gitignored (`FinalData.csv`, `new_data.csv`,
  `kc135_wuc_lookup_levels.csv`).
- **Trained model weights stay on the school's GPU disk.** Do NOT push
  `wuc-model-hier/`, `wuc-model-v2/`, etc. to Hugging Face Hub.
- The legacy `jonday/wuc-model` is on HF from before this constraint took
  effect; user has not yet decided whether to delete that.
- For ANY remote LLM adapters: only `ClaudeAdapter` (Anthropic API) was
  considered, and it's gated behind `ANTHROPIC_API_KEY`. Default deployment
  uses local Gemma only.
- Streamlit Community Cloud is **not viable** (data sensitivity + no Ollama
  support + 1 GB RAM ceiling).

---

## Open follow-ups (ranked)

1. **Discrepancy-only model variant** — train on `["Discrepancy", "How Mal"]`
   only for the live pre-fix workflow. Expected ~0.55-0.65 macro F1 (down
   from 0.77 — informal text is harder). Tab 1 would route by inputs supplied.
2. **Hand-label ~100 production samples** and re-evaluate. Test set is sampled
   from the same QC pipeline as training; hand-labeled prod samples are the
   number to actually trust.
3. **Confusion matrix / error analysis** on the held-out test set. Errors
   often cluster around adjacent WUCs in the same subsystem; reveals
   relabeling opportunities.
4. **Prompt-style selector in Tab 3** — three named templates ("brief /
   engineering / executive") instead of one shared `ANALYST_PROMPT`.
5. **Recommendations step in prompt** — append "Propose 2-3 prioritized
   maintenance/supply-chain actions justified by the data."
6. **Move BERT to GPU at inference** — currently `model_loader.py` lets
   torch pick the device (works because of the `_model_device()` helper),
   but inference defaults to CPU on the streamlit launch unless `.to('cuda')`
   is explicit. Single-line patch.
7. **CSV batch prediction in Tab 1** — wasn't carried over from legacy
   `app.py`. Useful for bulk re-validation; easy to port.
8. **Bump `ClaudeAdapter` default** from `claude-opus-4-6` → `claude-opus-4-7`.
9. **Verify `kc135_wuc_lookup_levels.csv`** vs committed `_dictionary.csv` —
   teammate's file may have richer columns. App falls back to dictionary
   currently and that's fine.
10. **Decide fate of legacy `jonday/wuc-model` on HF** — user expressed
    discomfort about leaving CUI-trained weights on a public hub.

---

## Dataset shape (after `prepare_data.py`)

| Stage | Rows |
|---|---|
| Raw A + B combined | 260,467 |
| After dedup | 157,545 (~40% were duplicates between extracts) |
| After rare-class filter (min 5) | 156,359 / **1,251 classes** |
| Train / Val / Test | 125,087 / 15,636 / 15,636 |

Class distribution: median 15 examples/class, max 2,808 (heavy long tail).
Top-2-char system prefix shows ~10 dominant systems concentrating ~40% of
the data.

---

## Gotchas (lessons learned)

- **Auth on Windows multi-account:** machine has `usma-stats` cached; pushing
  to `lonespear/wuc_predict` needs GCM re-prompt via system browser.
  `cmdkey /delete:git:https://github.com` may not find the cached entry —
  manual re-auth via the GCM popup is the path.
- **Streamlit + JupyterHub at non-root path:** `jupyter_server_proxy` 4.x
  strips the prefix before forwarding. Do NOT set `--server.baseUrlPath`,
  even though the URL is deeply nested.
- **No sudo, no conda, no zstd CLI on the GPU box.** Use Python `zstandard` +
  `tarfile` for any extraction needs. Manual user-space Ollama install
  pattern in CLAUDE.md works.
- **Ollama latest releases ship `.tar.zst` only**, no `.tgz` fallback.
- **Terminal auto-indents pasted heredocs** in JupyterHub web terminal —
  `<<EOF ... EOF` gets bricked because closing `EOF` ends up indented.
  Use single-line `python -c "..."` with semicolons.
- **Terminal sessions are not nohup-safe by default** — original training
  run died mid-validation when terminal disconnected. ALWAYS use
  `nohup … > log 2>&1 &` for long-running training.
- **`load_best_model_at_end + LR scheduler exhaustion`** — calling
  `Trainer.train(resume_from_checkpoint=True)` after a completed run wastes
  compute because LR is at ~0. Use `train_continue.py` (fresh optimizer +
  scheduler from saved weights) instead.
- **TF auto-import in `transformers`** can clash with PyTorch CUDA libs.
  Set `os.environ["USE_TF"] = "0"` before any transformers import.
- **`protobuf>=5` removed `MessageFactory.GetPrototype`** — older deps may
  break. Workaround: `pip install --user "protobuf<5"` or use the env-var
  TF skip above to avoid the call path.

---

## Recent commit history

| Commit | Summary |
|---|---|
| `63a202f` | Add README.md — architecture, flow charts, training pipeline, deployment guide |
| `621317f` | Add compare_models.py — head-to-head old vs new with calibration table |
| `9e52cda` | Show low-confidence as warning/error instead of green success |
| `085adaa` | Wire app to local hierarchical model — two-input UI + top-3 with confidence |
| `adf6d9f` | Add train_hierarchical.py — joint system/subsystem/WUC loss |
| `1545d7d` | Add train_continue.py — 5 more epochs from wuc-model-v2 |
| `233d576` | Switch to ModernBERT-large for better accuracy |
| `ec2f37f` | Add train_fresh.py — ModernBERT fresh fine-tune with class-weighted loss |
| `b4f878b` | Fix stratified split — random val/test |
| `669c08b` | Add prepare_data.py — merge raw extracts |
| `e013ed1` | Add CLAUDE.md (initial) |
| `a090e77` | Add unified analytics platform with WUC profiling + pluggable LLM summarization |
