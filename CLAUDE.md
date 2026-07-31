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

## Status (2026-07-31)

**🚀 Shipped and live** at:
```
https://icsarl.westpoint.edu/jupyter-cdas2/user/jonathan.day/proxy/8501/
```

Currently running:
- ModernBERT-large hierarchical fine-tune (`./wuc-model-hier`) for Tab 1
- Gemma 4 (`gemma4:e4b`) via local Ollama for Tab 3
- Streamlit pointed at `WUC_MODEL_PATH=./wuc-model-hier`
- **Data: `app_data.csv`** (162,565 records, 2019-01-01 → 2026-03-31),
  built by `training/build_app_data.py` from `data/data1.csv` +
  `data/data2.csv`. NOT `FinalData.csv` — see below.

**Read `GLIDEPATH.md` for what to work on next.** It holds the finish line
and, more importantly, the parked list with un-park triggers.

### Phase 0 closed 2026-07-31 — the app had been reading the wrong file

`resolve_data_path()` resolved to `FinalData.csv`, a stale artifact that was
an *input* to an earlier `prepare_data.py`. It had 20 columns and was data1
only — missing `Base`, `Flight Hours`, `JCN`, `When Discovered Code`,
`Type Maint Code`, **and all 35,060 of data2's records (27% of the data).**

`wuc_profile.py` guards each column with `if "X" in df.columns`, so they were
silently skipped. Six profile sections had been empty since the beginning:
base_distribution, base_geo, flight_hour_buckets, cooccurring_wucs,
when_discovered_phase, maint_type_phase.

That explains two earlier dead ends: the Tab 3 map (bubbles key off `Base`,
so four commits went into a map with nothing to plot) and the analyst-prompt
"insufficient data" fight (the fields really were absent — the `c42cd87`
revert masked a data bug). **With real data behind it, the sectioned prompt
from `c42cd87` is worth reconsidering.**

Verified working on `12AAN`: 171 records, 106 airframes, 100% of bases
geolocated, all seven sections populated.

---

## ⚠️ Pooling mismatch — found and fixed 2026-07-31

**Tab 1 served degraded predictions from `adf6d9f` (2026-05) until 2026-07-31.**

`train_hierarchical.py::HierarchicalModel._pooled()` pools by hand:

```python
cls = encoder_out.last_hidden_state[:, 0, :]   # CLS
return self.wuc_model.head(cls)
```

but saved through `wuc_model.save_pretrained()`, and the config inherited
`classifier_pooling="mean"` from the base checkpoint. So every consumer going
through `AutoModelForSequenceClassification` — `model_loader.py`,
`batch_predict.py`, `compare_models.py`, **and the live app** — mean-pooled
across tokens before `head`, feeding the classifier a representation it never
trained on.

Nothing errored. Predictions stayed plausible. Confidences looked reasonable.

Measured on the held-out test split (`training/check_pooling.py`):

| classifier_pooling | top-1 |
|---|---|
| `mean` (what was served) | 0.7557 |
| `cls` (how it was trained) | **0.8973** |

Full test set after patching `wuc-model-hier/config.json` to `"cls"`:
**top-1 0.9032, top-3 0.9781** — matching the originally reported 0.903.

Confidence distribution also changed completely: the ≥70% band went from
4,596 to **14,500** of 15,636 rows. Tab 1's low-confidence warnings were
largely an artifact of the wrong pooled vector.

**Consequences still outstanding:**
- **The v2-vs-hierarchical table below was NOT corrupted.** Both figures are
  training-time evals, each model scoring itself in-process. The pooling
  defect only affected the serving path. Re-verified 2026-07-31.
- **`compare_models.py` is broken for the OLD model** and its output should
  be ignored, not acted on. It maps predictions through
  `model.config.id2label`, but `jonday/wuc-model` only ever had HF's default
  `LABEL_0..LABEL_1726` placeholders there — so it scores 0.0000 by
  construction while reporting 99% confidence. Re-run 2026-07-31 gave
  `OLD 0.0000` (meaningless) vs `NEW 0.9080 acc / 0.8071 macro F1` on 2,000
  held-out rows. Only the NEW column is real. Note macro F1 over 1,251
  classes on 2,000 rows is very noisy — do not quote it against the 0.772
  training-time figure.
- **Same root cause hides a dead fallback in `model_loader.py`.** The
  `wuc_mapping.json` fallback is guarded by
  `if model.config.id2label and len(...) == num_labels`, which the legacy
  model's placeholder labels satisfy. The fallback never fires; running
  without `WUC_MODEL_PATH` yields `LABEL_847`-style output. Moot once
  `WUC_MODEL_PATH` is made mandatory (see open follow-ups).
- The smoke-test figure below (`12AAN` at 76.8%) was measured under the
  defect and will now read differently.
- Backup of the original config is at `wuc-model-hier/config.json.bak`.

**If you ever train a model with a custom `forward`, verify the saved config
describes what the forward actually does.** `save_pretrained()` persists the
config, not your Python.

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

## Verified held-out performance (2026-07-31)

Measured with `training/batch_predict.py --exclude-seen`, on 17,041 records
from `app_data.csv` with every train/val row removed by exact input-text
match. **These are the numbers to quote**, not the training-time figures.

| Metric | Value | Meaning |
|---|---|---|
| top-1, answerable rows | **0.9162** | model quality |
| top-3, answerable rows | **0.9797** | with the top-3 UI, the right code is nearly always on screen |
| top-1, all rows | **0.8536** | end-to-end, what a user experiences |
| label coverage | **93.2%** | share of real WUCs the model can emit at all |

Calibration on answerable rows:

| Band | n | top-1 |
|---|---|---|
| ≥70% | 14,795 | 0.9463 |
| 30-70% | 980 | 0.5245 |
| <30% | 101 | 0.3069 |

Two separate things, two separate fixes:

- **Accuracy** is model quality. 0.9162 held-out.
- **Coverage** is a label-map decision. 6.8% of real records carry a WUC
  that `prepare_data.py`'s `MIN_PER_CLASS = 5` filter kept out of the label
  space entirely — structurally impossible to predict, no matter how good
  the model gets. Lower the threshold, or document the gap. Do not conflate
  it with accuracy.

87% of records land in the ≥70% band, and that band is honest at 0.9463.
The model rarely claims uncertainty — which is what makes the hand-labeled
check of that band the number that actually matters.

**Still pending: the hand-labeled result.** Everything above is scored
against QC-pipeline labels, i.e. the same process that produced the training
targets. `labeling_worksheet.csv` (102 records, 34 per confidence band) is
the instrument for replacing it.

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
| Top-1 prediction | **`12AAN — FUSELAGE COMPARTMENTS / Safety Belt`** at **99.8%** |
| Runners-up | `12AGA` Safety Belts (TCI) 0.1%, `12AAJ` Inertia Reel Assembly 0.0% |

`12AAN` is literally "Safety Belt" in the WUC dictionary. **Correct answer**,
with semantically adjacent runners-up correctly ranked far below.

### A warning about how this entry used to read

Before 2026-07-31 this smoke test returned **76.8%**, and this file explained
the lower confidence as *"a calibration improvement, not a regression —
class-weighted CE + hierarchical regularization deliberately damp
overconfidence."*

That explanation was wrong. It was a **plausible, technically literate story
invented to account for a bug's symptom**, and it stood for three months
because it sounded right and nothing contradicted it. The real cause was the
pooling mismatch above.

The tell was available the whole time: the story explained *why the number
was low* but nobody checked whether the number *should* be low. When a
metric moves and you can immediately explain why, that is the moment to
measure, not to write the explanation down.

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
| `training/build_app_data.py` | **Builds `app_data.csv`** — merges the data/ extracts, parses Excel-serial dates, regenerates normalized text columns, reports profile-column coverage. |
| `training/batch_predict.py` | Batched CUDA top-k inference. Bulk re-validation + `--worksheet N` writes the Phase 1 hand-labeling sheet. |
| `training/prepare_data.py` | Merge raw extracts → train/val/test parquet splits. |
| `training/train_fresh.py` | Fresh fine-tune (single classifier head). |
| `training/train_continue.py` | Continue from existing checkpoint with reset optimizer. |
| `training/train_hierarchical.py` | Joint system/subsystem/WUC fine-tune; **produces the shipped model**. |
| `training/compare_models.py` | Head-to-head old vs new on test set (accuracy + calibration). |
| `archive/` | `app.py`, `sum_app.py`, `batch_wuc_legacy.ipynb` — reference only, nothing imports them. |
| `README.md` | Public-facing architecture + flow charts. |
| `GLIDEPATH.md` | **Finish line + parked list.** Read before starting work. |
| `CLAUDE.md` | ← this file (mutable state). |

Root holds only what `main_app.py` imports, plus data assets. Training
scripts have no local imports and still run from repo root:
`~/.venvs/wuc/bin/python training/train_hierarchical.py`.

**Gitignored — never committed:**
- `data/`, `app_data.csv`, `FinalData*.csv`, `new_data.csv`,
  `kc135_wuc_lookup_levels.csv`, `nov_24_wuc_predict.csv` (CUI data + derived)
- `labeling_worksheet.csv`, `*_predictions.csv`, `*.log`
- `data_splits/`, `wuc-model-v2/`, `wuc-model-v2-extended/`, `wuc-model-hier/`

---

## Tab 1 UX (current)

**Note (2026-05-12):** `build_input_text()` now **uppercases** all inputs and
accepts all 5 training fields — Discrepancy, Corrective Action, WCE Narrative,
How Mal, Action Taken (the last 3 are wired through a `st.expander` of optional
inputs below the two main text areas). This fixes the OOD-at-inference issue
(training text is uppercase maintenance-report style; inference used to be
lowercase + only 2 fields). Also: the old **"Unknown Definition"** WUC-lookup
gap is fixed — `model_loader.py` now merges `codes.json` with
`kc135_wuc_lookup_dictionary.csv` (codes.json wins, CSV fills gaps); if a code
is still missing the fallback string is now `(no dictionary entry for <wuc>)`.

Two text fields side-by-side:
- **Discrepancy** (required)
- **Corrective Action** (optional but improves accuracy substantially)
- plus **WCE Narrative / How Mal / Action Taken** in an optional expander

Joined (uppercased) with `[SEP]` → `predict_top_k(text, k=3)`.

Top-1 displayed with **confidence-band coloring**:

| Confidence | Display |
|---|---|
| ≥70% | 🟢 Green success |
| 30-70% | 🟡 Yellow warning ("review alternatives") |
| <30% | 🔴 Red error ("likely OOD input") |

Top-2 and top-3 always shown below as bullet points under "Other candidates".

---

## Restart (every session, after `git pull`)

**Paste these as SINGLE LINES.** The JupyterHub web terminal auto-indents
pasted text and silently truncates backslash continuations — it has mangled
this command repeatedly.

```bash
pgrep -f "ollama serve" > /dev/null || nohup ollama serve > ~/ollama.log 2>&1 &
```

```bash
pkill -9 -f streamlit; sleep 2; WUC_MODEL_PATH=./wuc-model-hier nohup ~/.venvs/wuc/bin/streamlit run main_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false > ~/streamlit.log 2>&1 &
```

**Note the venv path** — `~/.venvs/wuc/bin/streamlit`, not bare `streamlit`.
Same for scripts that import torch: `~/.venvs/wuc/bin/python training/...`.

**`WUC_MODEL_PATH` is REQUIRED.** Without it, `model_loader.py` falls back to
`jonday/wuc-model` (legacy BERT-base on HF) — that's NOT the shipped model,
and it has a different label space (1727 vs 1251 classes). It loads without
complaint and returns confident wrong answers. `training/batch_predict.py`
hard-fails when it's unset; the Streamlit app does not (open follow-up).

**Ollama cold start takes ~150 seconds** loading 8.9 GiB onto the GPU. Tab 3
appears to hang — showing "Narrative Summary" and nothing below, because the
breakdowns render after the stream completes. It is not broken. Use the
**Template (offline)** engine to check data independent of the LLM.

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

## Open follow-ups

**Moved to `GLIDEPATH.md`** — that file ranks them into phases and, more
usefully, records what is deliberately NOT being built with an un-park
trigger for each. The ten-item list that used to live here is what project
creep looks like: only four of them were ever blocking.

Current state: **Phase 0 closed** (data source fixed). **Phase 1 is next** —
`batch_predict.py --worksheet 34`, then hand-check 100 production records.
That number is the only thing standing between this project and done.

Two small items found 2026-07-31, not yet fixed:

- `model_loader.py` should **raise** when `WUC_MODEL_PATH` is unset rather
  than silently loading the legacy HF model.
- `build_input_text()` guards with `isinstance(value, str)`, which silently
  drops `How Mal` / `Action Taken` when they arrive as pandas numerics.
  `batch_predict.py` works around it; the shared helper should be fixed.

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

- **A custom `forward()` does not change the saved config.** The pooling bug
  above cost 14 points of top-1 accuracy in production for three months with
  no error, no log line, and plausible-looking output. Any time training
  hand-rolls the forward pass, assert that the config round-trips: load the
  saved checkpoint with `AutoModel...` and confirm it scores what training
  reported. `training/check_pooling.py` does exactly this for pooling.
- **Trust held-out re-measurement over documented metrics.** The 0.903 in
  this file was correct and the deployment was still broken — training-time
  numbers say nothing about the serving path.
- **The container gets rebuilt under you.** As of 2026-07-31 the image moved
  Python 3.10 → 3.12, orphaning every `pip install --user` package in
  `~/.local/lib/python3.10`. Symptom: `ModuleNotFoundError: No module named
  'streamlit'` even though `~/.local/bin/streamlit` exists. Diagnose with
  `python -V` vs `ls ~/.local/lib/`.
- **The 3.12 image ships `torch 2.12.0+cu130` with working CUDA. NEVER let
  pip replace it.** `requirements.txt` lists `torch`, so
  `pip install -r requirements.txt` would pull a PyPI wheel that may not
  match the driver (595.71 / CUDA 13.2, RTX 6000 Ada 48 GB). Install the
  other packages explicitly instead.
- **System Python is PEP 668 externally-managed** — plain `pip install --user`
  fails. The fix in place is a venv that inherits the image's torch:
  `python -m venv --system-site-packages ~/.venvs/wuc`, then
  `~/.venvs/wuc/bin/pip install streamlit altair transformers huggingface-hub ollama anthropic`.
  Verify torch survived: it must still report `2.12.0+cu130` and `cuda True`.
- **Raw extracts store dates as Excel serials** (`43718` = 2019-09-10).
  `pd.to_datetime` on an integer column reads them as *nanoseconds since
  epoch*, so everything lands in Jan 1970 and every date filter silently
  matches nothing — `sum_utils.py:180` coerces, and `NaT >= start_date` is
  always False. `build_app_data.py::_parse_dates` detects and converts;
  it now fails loudly rather than passing bad dates to the UI.
- **Quartile buckets computed on the subset are tautological.** Fixed in
  `_flight_hour_buckets` — edges now come from the full fleet, so departure
  from 25% is the signal. Watch for this pattern anywhere else: deriving
  thresholds from the same slice you're binning guarantees a flat result.
- **`altair` and `numpy` are imported directly** (`main_app.py:153` et al.)
  but were missing from `requirements.txt` until 2026-07-31 — they came in
  transitively via streamlit and would have broken a clean install.

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
