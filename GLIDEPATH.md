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

## Phase 1 — Trust the number  ✅ CLOSED 2026-07-31

**Exit criterion met: the accuracy claim is now defensible.**

> 0.9162 top-1 / 0.9797 top-3 on 15,876 held-out answerable records.
> 98.31% system-level. 93.2% label coverage. Scored against QC-pipeline
> labels; not independently hand-verified.

Getting there found a **14-point production bug** (train/serve pooling
mismatch, live for three months) that no amount of code review would have
caught — every path loaded cleanly and returned believable WUCs. See
CLAUDE.md.

Error character says the model is sound: only 1.7% of records are
cross-system errors, and 62.8% of all mistakes are same-subsystem near
misses. Several top confusion pairs are annotation convention, not model
failure — `72LA0 ↔ 72VA0` is bidirectional, and rollup-vs-specific pairs
like `624A0→62400` recur. **0.9162 is a floor.**

One gap remains open by design: `adjudication_worksheet.csv`, 25
high-confidence disagreements needing ~30 minutes from a KC-135 maintainer.
**This does not block anything.** It is a known limitation with a written
fallback claim, not an unfinished task.

---

<details>
<summary>Original Phase 1 plan</summary>

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
| 1.3 | **REPLANNED 2026-07-31** — see below. | — |
| 1.4 | Write the number into `CLAUDE.md` and `README.md`. | 15 min |

### Why 1.3 was replanned

The original plan — hand-label 102 records — assumed a KC-135 maintainer was
doing it. Neither the project owner nor Claude is one. Reading a write-up and
independently naming the correct WUC is expert judgment; guessing at it would
produce an authoritative-looking number backed by nothing, which is exactly
how `0.903` survived three months in `CLAUDE.md` while the deployed model was
serving `0.7557`.

**A wrong number with a confident provenance is worse than an honest gap.**

Replaced with two tracks:

| # | Task | Who | Est. |
|---|---|---|---|
| 1.3a | `training/error_analysis.py` — exact / subsystem / system agreement from the WUC code hierarchy, near-miss rate, top confusion pairs. Tells you the *character* of the errors. | no expertise needed | 10 min |
| 1.3b | `adjudication_worksheet.csv` — the ~25 highest-confidence disagreements, where either the model or the QC label must be wrong. Expert marks MODEL / LABEL / NEITHER. | needs a maintainer | ~30 min of their time |

A stratified sample is the wrong instrument when the expert is the scarce
resource. High-confidence disagreements are where the information is: at 99%
confidence against a conflicting label, one side is definitively wrong, and
**if it is the label, every accuracy figure in this repo is understated.**

If no maintainer is ever available, 1.3b stays open and the honest claim is
"0.9162 held-out against QC-pipeline labels, which were produced by the same
process that generated the training targets." State the limitation; do not
invent a number to close the gap.

**Exit criteria:** a defensible accuracy figure and the labeled CSV that backs
it. Once 1.4 is written down, Phase 1 is closed — do not reopen it to "get a
better number."

Note 1.1 is the same tool that does bulk re-validation later, so it is not a
detour — it's the instrument for 1.3.

---

</details>

---

## Phase 2 — Fix only what the labels expose  ✅ CLOSED 2026-07-31

| # | Task | Outcome |
|---|---|---|
| 2.1 | Confusion analysis | ✅ `training/error_analysis.py`. 98.31% system-level; only 1.7% of records are cross-system errors; 62.8% of mistakes are same-subsystem near misses. |
| 2.2 | Relabel ambiguous WUC pairs | ✅ **Identified, not applied.** `confusion_pairs.csv` exports every pair with counts and a bidirectional flag. |
| 2.3 | `.to('cuda')` at inference | ✅ done — Tab 1 was running on CPU |
| 2.4 | Retrain | ❌ **Declined. See below.** |
| 2.5 | `WUC_MODEL_PATH` must raise | ✅ done — silent legacy-model fallback removed |
| 2.6 | `build_input_text` field coercion | ✅ done — `str()` instead of `isinstance(str)` |

### Why no retrain

2.1 did find systematic labeling problems, which is the trigger 2.4 was
written for. Sized before acting: the top 15 confusion pairs total ~137
records out of 15,876 — under 1%. Merging every one of them buys roughly a
point of exact-match.

Against that, a retrain changes the label space, invalidates the deployed
checkpoint, and re-opens the evaluation Phase 1 just closed. **That is the
"+0.01 macro F1" trade this document was written to prevent.**

The relabeling finding is more valuable as **output to whoever owns the
source data** than as a model change. `72LA0 ↔ 72VA0` being bidirectional
means their own records use two codes interchangeably — actionable for them
regardless of any model. `confusion_pairs.csv` is that deliverable.

**Un-park a retrain only if:** the data owners actually merge codes upstream
and ship a corrected extract, or coverage becomes a stated requirement and
`MIN_PER_CLASS` has to drop. Not for accuracy chasing.

---

<details>
<summary>Original Phase 2 plan</summary>

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

</details>

---

## Phase 3 — Freeze and hand off  ← *you are here*

| # | Task | Est. |
|---|---|---|
| 3.1 | Sync the box to `origin/main` and confirm what's running matches the repo. | 15 min |
| 3.2 | ✅ **DONE 2026-07-31 — deleted.** Dangling references cleaned up: `train_fresh.py` no longer prints push-to-hub instructions for it, `compare_models.py` now baselines against local `./wuc-model-v2`, README env-var table corrected. Original rationale: delete `jonday/wuc-model` from Hugging Face. Not because the weights are exploitable — they demonstrably are not (see CLAUDE.md) — but because it is superseded, its placeholder-label config makes it useless, nothing references it, and publishing a CUI-derived artifact to a commercial hub is an authorization question with zero upside. | 5 min |
| 3.3 | ~~Resolve local CUI copies~~ ❌ **NOT A TASK.** Corrected 2026-07-31: the laptop, the school Jupyter server, and OneDrive are all whitelisted NIPR and authorized to hold CUI. The boundary is NIPR vs public internet, not copy count. Earlier revisions implied otherwise — that was an incorrect inference by Claude, not a stated policy. | — |
| 3.4 | Final `CLAUDE.md` pass: restart command, real accuracy number, known limits. | 30 min |

**Exit criteria:** someone else can restart the app from `CLAUDE.md` alone.

---

## Phase 4 — Make it answer "so what?"  ✅ CLOSED 2026-07-31

| # | Outcome |
|---|---|
| 4.1 | ✅ `labor` (burden index), `base_concentration`, `trend` in the profile |
| 4.2 | ✅ prompt + offline template lead with burden, use concentration not raw counts |
| 4.3 | ✅ dark theme, one registered Altair theme, redundant charts deleted, tick density fixed, map and bar chart reconciled with the narrative |
| 4.4 | ✅ **`gemma4:31b` selected.** Every local Ollama model is now a dropdown option |
| 4.5 | ✅ breakdowns render before the model is called |

**Model decision, on evidence.** Same WUC, same profile, both local:

- `gemma4:e4b` (4 s) opened with *"the maintenance burden … is disproportionately **high**"* — a factual inversion of burden index 0.3 — then contradicted itself, and expanded base names into `"Pittsburgh ANGB, PA"` etc. that appear nowhere in the data.
- `gemma4:31b` (37 s warm) led correctly with *"frequent … low maintenance burden"*, cited `<11,901 hrs` exactly where e4b rounded, invented nothing, and drew the distinction the whole phase was built for: *"While McConnell has the most raw records, the problem is genuinely concentrated at Birmingham, Pittsburgh and Mitchell."*

Its own cold run had hallucinated co-occurring WUCs `12C11`/`12C12`. **Size reduced fabrication; it did not eliminate it.** Treat narrative specifics as unverified.

**Latency stopped being a tradeoff** once 4.5 landed: 37 s of generation costs nothing when the charts are already on screen. Pin the model with `OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1` on `ollama serve` and the cold start is once per boot.

### Declined: fp16/bf16 for the classifier

Proposed as a speed win, then dropped. BERT inference on one short string is already ~10 ms — the saving is invisible — and changing precision perturbs logits, which would invalidate the **0.9162 held-out figure Phase 1 spent a day establishing**. Not worth re-running the evaluation to save nothing a user can perceive.

---

<details>
<summary>Original Phase 4 plan</summary>

**Why this is not creep, when "UI polish" is in Killed below.** That entry was
written when the tabs were rendering empty sections and the real problem was
data. The complaint now is different: the tabs are full of real content that
reads as *"a bunch of run-on visuals and summary tables."* That is an
information-architecture finding, not gold-plating.

**The framework is not the problem.** Streamlit already survives the four
constraints that matter — JupyterHub reverse proxy, no sudo, CUI so nothing
leaves the box, no CDN reachability (see the topojson fight in `8c3b78c`).
Dash, NiceGUI and Reflex all look better out of the box and would each mean
re-litigating proxy paths and asset loading, to arrive at the same run-on
layout in a different framework. **Do not port.**

**The model is not the problem either.** `gemma4:e4b` reads fine; it has
nothing to reason *with*. The profile hands it counts and no basis for
judging whether 171 records is a lot. Asked for a "so what", any model of any
size can only reword the counts.

Ordered so each step makes the next worth doing:

| # | Task | Why it comes first |
|---|---|---|
| 4.1 | **Enrichment.** See the column audit below — scope narrowed after checking. `Labor` (man-hours) plus fleet-relative comparisons. | Turns "appears 171 times" into "costs N man-hours, 2.3× the fleet rate per event, over-represented at McConnell" |
| 4.2 | **Prompt rework** against the enriched profile. Revisit the sectioned `ANALYST_PROMPT` reverted in `c42cd87` — it was reverted for saying "insufficient data" when the fields genuinely were absent. | Needs 4.1 to have anything to say |
| 4.3 | **Visual pass.** Kill redundant charts (calendar heatmap + month bars + year bars all answer "when" — keep the heatmap). BLUF card of 3 numbers at the top. `st.container(border=True)` for hierarchy. Sub-tabs inside Tab 3 (Why / When / Where / Lifecycle) to end the scroll. One registered Altair theme — which also fixes the 60-tick axis problem globally. Dark theme via `.streamlit/config.toml`. | Worth doing once there is something worth displaying |
| 4.4 | **Optional: bigger local model.** An RTX 6000 Ada with 48 GB is running a 9.6 GB model. `gemma3:27b` or `qwen3:32b` at Q4 fits alongside the BERT model. One `ollama pull`, fully local, no CUI implication. | Cheapest quality step, but 4.1 matters more |

### Column audit, 2026-07-31 — two enrichment inputs are dead

Checked before building, after today's lesson about unverified column
assumptions. On all 162,565 records:

| Column | Verdict |
|---|---|
| **`Labor`** | ✅ **USE.** float64, zero nulls, median 2.0 h, mean 4.40, IQR 1–5, max 90. Man-hours per maintenance action, clean. |
| `Stop Date − Start Date` | ❌ **DEAD. Do not re-attempt.** 161,438 of 162,565 (99.3%) are exactly zero days; max is 1. Stop and Start are the same calendar day for essentially every record. These are day-granularity job records, not down-time tracking. There is no aircraft-downtime signal in this data. |
| `How Mal Class` | ❌ **DEAD.** 162,564 records are `1`, one is `16`. Constant. |
| `Units Produced` | ⚠️ Marginal. Mean 1.03, median 1, one outlier at 4,514. Not worth a multiplier. |

So "aircraft-days down" is **not** a claim this dataset can support, and the
tool must not imply otherwise. The remaining levers are labor and *relative*
measures — which need no new columns, only comparison against the fleet.

**Non-goals for Phase 4** — write these down before starting:

- No framework port.
- No new tabs. Three is right.
- No chart that does not answer a question a maintainer would actually ask.
- Not a retrain. Phase 2 declined that and nothing here changes it.

**Exit criteria:** Tab 3 opens with three numbers that matter, the narrative
cites burden and downtime rather than counts alone, and the page fits a
screen without scrolling to find the point.

</details>

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
