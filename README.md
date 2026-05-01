# KC-135 WUC Maintenance Analytics

A unified Streamlit platform for KC-135 maintenance analytics. Combines a
fine-tuned text classifier (predict Work Unit Code from free text), a natural-
language record query tool, and an LLM-narrated WUC profile.

> **Note on data sensitivity.** The training data is treated as CUI. The
> dataset CSVs are gitignored and never committed; the trained model weights
> are kept on internal infrastructure (not pushed to public model hubs).

---

## What it does

Three tabs in one Streamlit app, sharing a single dataframe and a single
WUC→description map:

| Tab | What it does | Engine |
|---|---|---|
| 🔮 **Predict WUC** | Input a discrepancy + corrective action; get top-3 WUC predictions with confidence | Fine-tuned ModernBERT-large classifier (1,251 classes) |
| 🔎 **Query Records** | Natural-language question over historical records; returns counts, monthly trend, top WUCs | Regex-parsed filters → pandas |
| 📊 **WUC Profile** | Pick a WUC; get a deterministic profile (why / when / where / lifecycle / co-occurrence) plus an LLM-narrated summary | Pandas + pluggable LLM adapter (Gemma 4 / Claude / template) |

---

## Architecture

### Top-level data flow

```
                ┌──────────────────────────────────────┐
                │  Streamlit (main_app.py)             │
                │  ┌────────────────────────────────┐  │
                │  │ Shared state                   │  │
                │  │  • df       (FinalData.csv)    │  │
                │  │  • desc_map (WUC → description)│  │
                │  └────────────────────────────────┘  │
                │       │            │           │     │
                │       ▼            ▼           ▼     │
                │  ┌─────────┐  ┌─────────┐  ┌──────┐  │
                │  │ Tab 1   │  │ Tab 2   │  │Tab 3 │  │
                │  │ Predict │  │ Query   │  │ Prof.│  │
                │  └────┬────┘  └────┬────┘  └──┬───┘  │
                └───────┼────────────┼──────────┼──────┘
                        │            │          │
                        ▼            ▼          ▼
              ┌──────────────┐ ┌──────────┐ ┌──────────┐
              │model_loader  │ │sum_utils │ │wuc_profile│
              │  ModernBERT  │ │  pandas  │ │  pandas   │
              └──────────────┘ └──────────┘ └─────┬────┘
                                                  │
                                                  ▼
                                          ┌────────────────┐
                                          │ llm_adapter    │
                                          │  Null / Gemma  │
                                          │  Claude        │
                                          └────────────────┘
```

### Tab 1 — Predict WUC

```
   user input ───► build_input_text(discrepancy, corrective_action)
                          │
                          ▼  "<discrepancy> [SEP] <corrective_action>"
                   ┌──────────────────────────┐
                   │ predict_top_k(text, k=3) │
                   │  • tokenize              │
                   │  • model.forward         │
                   │  • softmax → top-k       │
                   └──────────────┬───────────┘
                                  │
                                  ▼
                   ┌──────────────────────────┐
                   │ Confidence banding       │
                   │  ≥70%   → green success  │
                   │  30-70% → yellow warning │
                   │   <30%  → red error      │
                   └──────────────────────────┘
```

### Tab 3 — WUC Profile + LLM summary

The full streaming pipeline from a structured profile dict to live-rendering
narrative:

```
                                         ┌─────────────────────┐
   user clicks "Build Profile" ─────────►│ wuc_profile.        │
   in WUC Profile tab                    │ build_profile()     │
                                         └──────────┬──────────┘
                                                    │
                                                    ▼
                                         ┌─────────────────────┐
                                         │ profile dict        │
                                         │ {wuc, total, top_   │
                                         │  phrases, hist…}    │
                                         └──────────┬──────────┘
                                                    │
   user picks "Gemma 4 — gemma4:e4b…"               │
   from adapter dropdown                            │
                                                    ▼
                                         ┌─────────────────────┐
                                         │ adapter.summarize_  │
                                         │   stream(profile)   │
                                         └──────────┬──────────┘
                                                    │
                                          _build_prompt(profile)
                                          = ANALYST_PROMPT + json.dumps(profile, indent=2)
                                                    │
                                                    ▼
                                         ┌─────────────────────┐
                                         │ ollama.chat(        │
                                         │   model=…,          │
                                         │   messages=[{user}],│
                                         │   stream=True,      │
                                         │   options={temp 0.3,│
                                         │     num_ctx 8192})  │
                                         └──────────┬──────────┘
                                                    │
   ┌────────────────── HTTP POST localhost:11434 ───┘
   │  (local, not cloud — Ollama daemon)
   ▼
┌──────────────────────────┐
│ Ollama daemon            │
│  loads gemma4 GGUF       │
│  → generates tokens      │
│  → emits chunks          │
└──────────┬───────────────┘
           │   NDJSON stream:
           │   {"message": {"content": "WUC "}}
           │   {"message": {"content": "12AA0 "}}
           │   {"message": {"content": "is the…"}}
           ▼
┌──────────────────────────┐
│ summarize_stream yields  │ ───►  for chunk in adapter.summarize_stream(profile):
│  each chunk["message"]   │           narrative += chunk
│  ["content"]             │           placeholder.markdown(narrative)
└──────────────────────────┘
                                       (Streamlit re-renders the panel each iteration
                                        → user sees text appearing live)
```

---

## Repo structure

```
wuc_predict/
├── main_app.py              # Streamlit entry point — 3-tab unified app
├── model_loader.py          # Loads classifier; exposes predict_discrepancy / predict_top_k
├── wuc_profile.py           # Deterministic WUC profile builder (no ML)
├── sum_utils.py             # NL query parser + record analysis
├── llm_adapter.py           # SummaryAdapter Protocol + Null/Gemma/Claude implementations
├── data_config.py           # Path resolution + WHEN_DISCOVERED / TYPE_MAINT code dicts
│
├── prepare_data.py          # Merge raw extracts → train/val/test parquet splits
├── train_fresh.py           # Fresh fine-tune (single classifier head)
├── train_continue.py        # Continue training from existing checkpoint
├── train_hierarchical.py    # Joint system/subsystem/WUC fine-tune
├── compare_models.py        # Head-to-head old vs new on test set with calibration
│
├── codes.json               # WUC → human-readable definition
├── main_system.json         # 2-char prefix → main system name
├── kc135_wuc_lookup_dictionary.csv  # WUC → description fallback lookup
│
├── requirements.txt
├── CLAUDE.md                # Session-context notes
└── README.md                # ← you are here

# Gitignored — never committed
├── FinalData.csv            # Maintenance records (CUI)
├── new_data.csv             # Additional raw extract
├── kc135_wuc_lookup_levels.csv
├── data_splits/             # train.parquet / val.parquet / test.parquet
├── wuc-model-v2/            # Trained ModernBERT-large checkpoint
├── wuc-model-hier/          # Trained hierarchical checkpoint
└── wuc-model-v2-extended/   # Continuation-training checkpoint
```

---

## Models

### Classifier (Tab 1)

| Model | Architecture | Train set | Test acc | Test macro F1 |
|---|---|---|---|---|
| Original `jonday/wuc-model` | bert-base-uncased | unknown | — | — |
| `wuc-model-v2` (flat) | ModernBERT-large | 125k / 1,251 classes | **0.904** | **0.772** |
| `wuc-model-hier` (hierarchical) | ModernBERT-large + aux heads | 125k / 1,251 classes | 0.903 | 0.772 |

Macro F1 ties between flat and hierarchical, but **hierarchical has 47% lower
test loss** (1.04 → 0.55) — significantly better calibrated, which matters for
top-k display and confidence-threshold rejection.

### Summarizer (Tab 3) — pluggable

| Adapter | Backend | Network | Streaming | Notes |
|---|---|---|---|---|
| `NullAdapter` | Python templates | None | No | Always available; deterministic; enterprise-safe |
| `GemmaAdapter` | Local Ollama (`gemma4:e4b` default) | `localhost:11434` | **Yes** | Default LLM. No data leaves the host. |
| `ClaudeAdapter` | Anthropic API | Yes (api.anthropic.com) | No | Activates only if `ANTHROPIC_API_KEY` is set. Disabled by default. |

Adding a new adapter = one class implementing the Protocol; no other file
changes needed. See `llm_adapter.py` for the interface.

---

## Quick start

### Local (development)

```bash
git clone https://github.com/lonespear/wuc_predict.git
cd wuc_predict
pip install -r requirements.txt

# Place your data files (gitignored — never committed)
cp /path/to/FinalData.csv .
# kc135_wuc_lookup_dictionary.csv ships with the repo as a fallback

# Run
streamlit run main_app.py
# → http://localhost:8501
```

### Production (USMA dockerized JupyterHub GPU box)

```bash
# 1. Install Ollama in user space (no sudo, no zstd CLI required)
curl -L https://github.com/ollama/ollama/releases/download/v0.22.0/ollama-linux-amd64.tar.zst \
  -o /tmp/ollama.tar.zst
pip install --user zstandard
python -c "import zstandard, tarfile; tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(open('/tmp/ollama.tar.zst','rb')), mode='r|').extractall('/home/jovyan/.local/'); print('ok')"
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

# 2. Start Ollama daemon and pull the model
nohup ollama serve > ~/ollama.log 2>&1 &
sleep 3 && ollama pull gemma4:e4b

# 3. Install Python deps
pip install --user -r requirements.txt

# 4. Launch Streamlit pointed at the local trained model
WUC_MODEL_PATH=./wuc-model-hier nohup streamlit run main_app.py \
  --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --server.enableCORS false \
  --server.enableXsrfProtection false --browser.gatherUsageStats false \
  > ~/streamlit.log 2>&1 &
```

Access via JupyterHub proxy:
```
https://<jupyterhub-host>/user/<user>/proxy/8501/
```

---

## Training pipeline

```
                 raw_csv_a     raw_csv_b
                     │             │
                     └──────┬──────┘
                            ▼
                  ┌──────────────────┐
                  │ prepare_data.py  │
                  │  • schema reduce │
                  │  • dedupe        │
                  │  • [SEP] join    │
                  │  • rare filter   │
                  │  • 80/10/10      │
                  └────────┬─────────┘
                           ▼
                  data_splits/
                    train.parquet
                    val.parquet
                    test.parquet
                    wuc_mapping.json
                           │
        ┌──────────────────┼──────────────────────┐
        ▼                  ▼                      ▼
  train_fresh.py    train_continue.py    train_hierarchical.py
        │                  │                      │
        ▼                  ▼                      ▼
   wuc-model-v2/   wuc-model-v2-extended/  wuc-model-hier/
                                                 │
                                                 ▼
                                       compare_models.py
                                       (old vs new on test)
```

| Script | Use case | Output |
|---|---|---|
| `prepare_data.py` | Merge raw extracts and produce splits | `data_splits/{train,val,test}.parquet` + `wuc_mapping.json` |
| `train_fresh.py` | Fresh fine-tune from a SOTA base (default: ModernBERT-large) | `wuc-model-v2/` |
| `train_continue.py` | Train N more epochs from an existing checkpoint with reset optimizer | `wuc-model-v2-extended/` |
| `train_hierarchical.py` | Joint system/subsystem/WUC fine-tune; aux heads regularize the encoder | `wuc-model-hier/` |
| `compare_models.py` | Head-to-head accuracy + calibration table on held-out test set | stdout report |

### Data prep specifics

- **Label hygiene** — `Corrected WUC` is treated as ground truth (raw `WUC` is
  what the maintainer typed; corrected is QC-validated).
- **Text construction** — `Discrepancy [SEP] Corrective Action [SEP] WCE
  Narrative [SEP] How Mal [SEP] Action Taken`. Maintenance-report style
  (uppercase, terse).
- **Deduplication** — exact `(text, label)` duplicates removed (often eliminates
  ~40% of rows when merging overlapping extracts).
- **Rare-class filter** — classes with `< MIN_PER_CLASS=5` examples dropped.
- **Stratified split** — first 80/20 stratified by `Corrected WUC`, then random
  50/50 inside the 20% temp.

### Training specifics

- **Class-weighted CrossEntropyLoss** with inverse-frequency weights to handle
  the heavy long tail (max class freq is ~190× median).
- **fp16 mixed precision** — fits comfortably on a 48 GB RTX 6000 Ada.
- **Best-checkpoint metric** — **macro F1** (treats all classes equally);
  accuracy is reported but not optimized for.
- **Hierarchical loss** (in `train_hierarchical.py`):
  `0.20·L_system + 0.30·L_subsystem + 0.50·L_wuc`. Auxiliary heads operate on
  the same pooled encoder representation; only the WUC head ships at inference.

---

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `WUC_MODEL_PATH` | Path/repo of the classifier checkpoint | `jonday/wuc-model` (legacy) |
| `WUC_DATA_PATH` | Override path to `FinalData.csv` | `./FinalData.csv` then `../kc135/kc_135.csv` |
| `ANTHROPIC_API_KEY` | Enables `ClaudeAdapter` in Tab 3 | (unset → adapter hidden) |
| `USE_TF` | Set to `0` to skip TensorFlow auto-import in `transformers` | `0` (recommended) |

---

## Confidence band UX (Tab 1)

The classifier's max-probability output is bucketed and rendered with intent:

| Confidence | Display | Message |
|---|---|---|
| **≥ 70%** | 🟢 Green success | Trust the top-1 |
| **30 – 70%** | 🟡 Yellow warning | Moderate — review alternatives |
| **< 30%** | 🔴 Red error | Likely OOD input — treat top-1 as a guess; review all 3 candidates |

This makes uncertainty visible. The hierarchical model's calibration ensures
that "76%" actually correlates with ~76% empirical accuracy, not overconfidence.

---

## Performance reference

Training run on RTX 6000 Ada, 48 GB VRAM, 125k examples, 1,251 classes:

| Model | Epochs | Wall time | Test acc | Test macro F1 | Test loss |
|---|---|---|---|---|---|
| `wuc-model-v2` (flat) | 5 | ~57 min | 0.904 | 0.772 | 1.035 |
| `wuc-model-v2-extended` (10 total) | 5 more | ~57 min | 0.906 | 0.771 | 1.290 |
| **`wuc-model-hier`** (hierarchical) | 5 | ~60 min | **0.903** | **0.772** | **0.555** |

The flat model and hierarchical model tie on macro F1, but the hierarchical
model is much better calibrated (~half the test loss). Continuation training
beyond 5 epochs slightly hurt macro F1 — the dataset's signal is exhausted at
5 epochs for this architecture.

---

## Open follow-ups

1. **Discrepancy-only model variant** — train on `["Discrepancy", "How Mal"]`
   only for the live-prediction (pre-fix) workflow; expected ~0.55-0.65 macro
   F1, honest baseline for that use case.
2. **Tab 1 model routing** — automatically use `wuc-model-discrepancy` when
   only the discrepancy is provided, `wuc-model-hier` when both fields are
   available.
3. **Confusion matrix / error analysis** — inspect where the model is wrong;
   often clusters around adjacent WUCs in the same subsystem and reveals data
   relabeling opportunities.
4. **Production sample re-evaluation** — the held-out test set is sampled from
   the same QC pipeline as training. Hand-label ~100 actual recent app
   submissions and measure on those — that's the number to trust.
5. **Distribution-drift monitoring** — periodic re-evaluation as text style
   evolves; retrain quarterly or annually.
6. **Prompt-style selector** in Tab 3 — three named templates (maintenance
   brief / engineering analysis / executive summary) instead of one shared
   `ANALYST_PROMPT`.

---

## License

(Internal — not yet licensed for external distribution.)
