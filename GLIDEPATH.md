# Glidepath to done

> Scope control document. `CLAUDE.md` holds mutable state, `README.md` holds
> architecture. **This file holds the finish line and everything deliberately
> not being built.**
>
> Rule: if a task is not in Phase 1-3 below, it does not get worked on. New
> ideas go to **Parked**, not into the current phase.

---

## Definition of done

> **A KC-135 maintainer types a discrepancy and corrective action, gets a WUC
> they can trust, and I can state the real-world accuracy with a number I
> defended by hand.**

That's it. The app already does the first half. Phase 1 is the second half.
Everything else is optional polish on a system that already works.

**The project is not "done" when it is feature-complete. It is done when the
accuracy claim is defensible.** Recognizing that is what stops the creep —
most of the open follow-up list is features, and features are not the blocker.

---

## Current layout

```
wuc_predict/
├── main_app.py           # entry point — 3-tab Streamlit app
├── model_loader.py       # WUC_MODEL_PATH, predict_top_k, build_input_text
├── wuc_profile.py        # deterministic pandas profile (Tab 3)
├── llm_adapter.py        # Null / Gemma / Claude adapters + ANALYST_PROMPT
├── sum_utils.py          # NL query parser + record analysis (Tab 2)
├── data_config.py        # path resolution + code dicts
├── base_geo.py           # base -> lat/lon for the map
├── training/             # not needed to run the app
│   ├── prepare_data.py
│   ├── train_fresh.py
│   ├── train_continue.py
│   ├── train_hierarchical.py
│   └── compare_models.py
├── archive/              # reference only, nothing imports these
│   ├── app.py
│   ├── sum_app.py
│   └── batch_wuc_legacy.ipynb
└── GLIDEPATH.md          # this file
```

Root now holds **only what `main_app.py` imports**, plus data assets. Training
scripts have no local imports, so they still run from repo root:
`python training/train_hierarchical.py`.

---

## Phase 0 — The data source fix  ✅ CLOSED 2026-07-31

Verified on `12AAN`: 171 records, 106 airframes, `base_geo_coverage` 171/171
(100%), and all seven sections populated. `flight_hour_buckets` came back
56/48/30/37 — a real skew toward lower-time airframes, not the forced-even
result the old code produced.

Four bugs fixed along the way, **three of them pre-existing rather than
regressions** — they had simply been invisible while five sections around
them were also empty:

1. Data source pointed at stale `FinalData.csv` (below).
2. `Start Date` arrives as an Excel serial; `pd.to_datetime` read it as
   nanoseconds since epoch, so every date landed in Jan 1970 and every Tab 2
   filter silently matched nothing.
3. `year_histogram` only read a `YEAR` column with no `Start Date` fallback,
   unlike `_month_histogram`. Empty since before the swap.
4. `_flight_hour_buckets` derived quartile edges from the subset it was
   binning — tautologically ~25% per bucket for every WUC.

Also resolved: the box's container was rebuilt on Python 3.12, orphaning the
3.10 user site-packages. Now on a `--system-site-packages` venv at
`~/.venvs/wuc` that inherits the image's `torch 2.12.0+cu130`. See CLAUDE.md
gotchas.

---

<details>
<summary>Original Phase 0 analysis (kept for the reasoning)</summary>

The app has been reading the wrong file for three months.

`resolve_data_path()` resolves to `FinalData.csv` at repo root — a **stale
artifact** that was an *input* to an earlier version of `prepare_data.py`
(see commit `370fd3a`, which repointed `PATH_A` to `data/data1.csv`). It has
20 columns and lacks `Base`, `Flight Hours`, `JCN`, `When Discovered Code`,
and `Type Maint Code`.

`wuc_profile.py` guards every one of those with `if "X" in df.columns`, so
they are **silently omitted** — no error. Six profile sections have been
empty this whole time: `base_distribution`, `base_geo`,
`flight_hour_buckets`, `cooccurring_wucs`, `when_discovered_phase`,
`maint_type_phase`.

That is the root cause of two things already fought:
- **The prompt war.** The sectioned `ANALYST_PROMPT` said "insufficient data"
  because the fields were literally absent. The revert in `c42cd87` masked a
  data bug rather than fixing it.
- **The map.** Four commits (`bdb98f2`, `f88b90b`, `8c3b78c`) went into a map
  whose bubbles key off `Base`. The topojson-behind-the-proxy fix was real and
  worth keeping, but there was never any data to plot.

The real extracts — `data/data1.csv` (31 cols) and `data/data2.csv` (21 cols)
— have every missing column. `data2 ⊂ data1`, so the schema intersection in
`prepare_data.py` preserves all of them; it only drops 10 derived columns
(`SYSTEM`, `NOUN`, `YEAR`, `MONTH`, …).

| # | Task | Est. |
|---|---|---|
| 0.1 | Uppercase audit (below) — settle it before anything is labeled. | 5 min |
| 0.2 | Point the app at merged `data1 + data2` instead of `FinalData.csv`. | 1 session |
| 0.3 | Regenerate `discrepancy_normalized` / `corrective_action_normalized` in the merge. Not urgent — `sum_utils.py:270-271` falls back to raw text with `or`. | 20 min |
| 0.4 | Confirm the profile populates and the map draws bubbles. | 10 min |

**Exit criteria:** Tab 3 renders all seven sections from real data. **Do not
touch map styling** — it will render, and that is where the creep lives.

### Train/inference format — RESOLVED 2026-07-31

Audited and matching. Field order, `" [SEP] "` separator, skip-empty
semantics, and `.strip()` are identical between `prepare_data.py` and
`build_input_text()`.

The one asymmetry — training does `str(v).strip()`, inference does
`.strip().upper()` — is a **no-op on this data**: `Discrepancy` and
`Corrective Action` in `data2.csv` are 100.0% already-uppercase
(n=132,962). Keep the `.upper()`; it defends against lowercase typed into
the Streamlit UI, which is the OOD case `1b3405e` was fixing.

**Phase 1 is unblocked.**

</details>

### Still to fix — silent field drop from pandas

Training accepts any non-null via `str(v)`; `build_input_text()` requires
`isinstance(value, str)`. Short code columns (`How Mal`, `Action Taken`)
often parse as numeric from a CSV, so they are silently dropped from the
input string — producing systematically shorter text than training used, and
an accuracy number that reads low for reasons unrelated to the model.
Harmless in Streamlit (widgets always return `str`). Coerce explicitly in
`batch_predict.py`; fix `build_input_text()` too.

---

## Phase 1 — Trust the number  ← *you are here*

**Goal:** replace "0.903 accuracy on a test set drawn from the same QC pipeline
as training" with "N% top-1 / N% top-3 on hand-checked production records."

### Already done 2026-07-31

`training/batch_predict.py` exists and works: batched CUDA inference,
`--exclude-seen` to drop training overlap, `--sample` for seeded random
draws, `--text-col` + parquet input, in-label-space reporting, `--worksheet`.

**Building it found a 14-point production bug.** Held-out scoring disagreed
with the documented 0.903, and chasing that disagreement — through
contamination, then truncation, then label coverage — ended at a
train/serve pooling mismatch. See CLAUDE.md. Verified held-out numbers on
the QC-labeled test split are now **top-1 0.9032 / top-3 0.9781**.

That is the argument for this phase in one sentence: none of the ten
follow-ups this project started with would have surfaced it.

**Still outstanding: re-run `compare_models.py`.** The v2-vs-hierarchical
comparison behind the deployment decision was measured through the broken
pooling path.

| # | Task | Est. |
|---|---|---|
| 1.1 | ~~`batch_predict.py`~~ ✅ done — see above | — |
| 1.2 | ~~Stratified sample~~ ✅ done — `labeling_worksheet.csv`, 102 records, 34 per band | — |
| 1.3 | Hand-check the 100 against the WUC dictionary. Mark top-1 correct / top-3 correct / wrong. | 2-3 sessions, and this is the tedious one |
| 1.4 | Write the number into `CLAUDE.md` and `README.md`. | 15 min |

**Exit criteria:** a defensible accuracy figure and the labeled CSV that backs
it. Once 1.4 is written down, Phase 1 is closed — do not reopen it to "get a
better number."

Note 1.1 is the same tool that does bulk re-validation later, so it is not a
detour — it's the instrument for 1.3.

---

## Phase 2 — Fix only what the labels expose

**Do not start until Phase 1 is closed.** The labeled set from 1.3 tells you
what's actually broken; guessing beforehand is how the follow-up list got to
ten items.

| # | Task | Trigger |
|---|---|---|
| 2.1 | Confusion analysis on the 100 labeled records — cluster errors by 2-char system prefix. | always |
| 2.2 | Relabel / merge WUC pairs that are genuinely ambiguous. | only if 2.1 shows adjacent-WUC clustering |
| 2.3 | `.to('cuda')` in `model_loader.py`. | do it while you're in the file — one line |
| 2.4 | Retrain. | **only if 2.2 finds a systematic labeling problem.** A retrain to chase +0.01 macro F1 is creep. |

**Exit criteria:** either a documented "errors are idiosyncratic, no fix
warranted" (a legitimate and likely outcome), or one retrain. Not two.

---

## Phase 3 — Freeze and hand off

| # | Task | Est. |
|---|---|---|
| 3.1 | Sync the box to `origin/main` and confirm what's running matches the repo. | 15 min |
| 3.2 | Delete `jonday/wuc-model` from Hugging Face. CUI-trained weights on a public hub is a liability, not a feature. | 5 min |
| 3.3 | Resolve the local CUI copies (`current_wire/`) — data lives on the box, not the laptop. | 5 min |
| 3.4 | Final `CLAUDE.md` pass: restart command, real accuracy number, known limits. | 30 min |

**Exit criteria:** someone else can restart the app from `CLAUDE.md` alone.

---

## Parked — not now, with the trigger that would un-park

These are good ideas. That's exactly why they're dangerous right now.

| Item | Un-park when |
|---|---|
| **Chart axis tick density** — the co-occurrence and Base bar charts render every 0.05 increment on small integer ranges, producing ~60 tick labels. One-line Altair fix (`tickMinStep=1`, `format='d'`). | Doing any other Tab 3 work — fold it in then, not before |
| **Reconsider the sectioned `ANALYST_PROMPT`** reverted in `c42cd87` | It was reverted because Gemma kept writing "insufficient data" — which was true at the time. The data now exists, so the sectioned version may be strictly better. Revisit when Phase 1 is closed |
| **Discrepancy-only model variant** (pre-fix live prediction) | Someone actually asks for pre-fix prediction. This is a **second model with its own training run, its own eval, and UI routing logic** — it is a v2 project, not a follow-up. Biggest single creep risk on the list. |
| Prompt-style selector, Tab 3 (brief/engineering/executive) | A reader complains about the current narrative style |
| Recommendations section in the analyst prompt | Same |
| CSV batch prediction *in the Tab 1 UI* | `batch_predict.py` (1.1) proves insufficient — the script covers the actual need |
| Verify `kc135_wuc_lookup_levels.csv` vs committed dictionary | The dictionary fallback visibly fails on a real code |

---

## Killed

| Item | Why |
|---|---|
| `ClaudeAdapter` default model bump | Default deployment is local Gemma. Dead code path on the box. |
| Streamlit Community Cloud | Not viable — CUI, no Ollama, 1 GB ceiling. Settled. |
| Any further UI polish | Tabs 1-3 all work. Polish is the creep. |

---

## The rule when a new idea arrives

Write it in **Parked** with its un-park trigger. Do not open a file. The list
above went from ten ranked follow-ups to four real tasks by asking one
question of each: *does the project fail without this?* For seven of them the
answer was no.
