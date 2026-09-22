# Validating an NLP Coding-Assist Model for KC-135 Maintenance Records

A fine-tuned text classifier that suggests the Work Unit Code (WUC) for a KC-135
maintenance record from the maintainer's own free-text discrepancy and corrective
action, delivered as a Streamlit tool, with the train-versus-serve verification that
decides whether its numbers can be believed.

> **Data sensitivity.** The training records are treated as CUI. Dataset CSVs are
> gitignored and never committed, and the trained weights stay on internal
> infrastructure (not pushed to public model hubs). Every figure below is a summary
> statistic; no record-level data appears in this repository.

---

## Bottom line

Maintainers hand-code one of roughly 1,300 WUCs per record. Miscoded records flow
straight into reliability and sustainment analysis, so a wrong code is not a clerical
problem, it is a bad input to a readiness decision.

The deployed model picks the QC-corrected code on **90.3%** of held-out records
(95% CI 89.9–90.8, n = 15,636), and its top-3 list contains the right code **97.8%**
of the time (97.6–98.0). Errors stay close to home: **98.3%** of records land in the
correct system even when the exact code is wrong.

The number to treat carefully is not the accuracy, it is the label. The model is
measured against QC-corrected codes, not against subject-matter-expert adjudication,
and no baseline has been measured yet for how often the maintainer's own typed code
already matches QC. Until that baseline exists, this model's *value added* is
unquantified even though its *accuracy* is measured.

## Decision supported

| Who | Decision | How the model is used |
|---|---|---|
| Maintainer | Which WUC to enter | Top-3 suggestions with a confidence band |
| QC reviewer | Which records to re-check | Records in the low-confidence band route to review |
| Data owner | Which code pairs to disambiguate | Recurring confusion pairs (e.g. `72LA0` ↔ `72VA0`) |
| Analyst | Whether WUC-coded history can carry an analysis | Coverage and error structure reported separately |

The tool never writes a code on its own. It proposes, and a person decides.

---

## Results

Every figure below names the script that produces it. Those scripts read CUI data on
authorized infrastructure, so their outputs are **not committed here**; rerun them on
the authorized box to reproduce. Intervals are Wilson 95% intervals.

### Primary: held-out test split

`training/batch_predict.py` on `data_splits/test.parquet` (random stratified 80/10/10
split, exact-duplicate text removed), n = 15,636:

| Metric | Value | 95% CI |
|---|---|---|
| Top-1 accuracy | **0.903** | 0.899 – 0.908 |
| Top-3 accuracy | **0.978** | 0.976 – 0.980 |

### End-to-end, including records no model could code

Across all 17,041 records in the app dataset (`training/build_app_data.py`):

| Metric | Value | 95% CI | Note |
|---|---|---|---|
| Top-1 accuracy, all records | 0.854 | 0.848 – 0.859 | Includes records whose label is not in the class set |
| Label coverage | 0.932 | 0.928 – 0.935 | 15,876 of 17,041 **records** are answerable |

Coverage and accuracy are reported separately on purpose. A tool that quietly counts
unanswerable records as correct, or drops them, tells you the wrong thing.

### How wrong the errors are

`training/error_analysis.py`, on the 15,876 answerable records:

| Metric | Value | 95% CI |
|---|---|---|
| Correct system | 0.983 | 0.981 – 0.985 |
| Correct subsystem | 0.969 | 0.966 – 0.971 |

Of 1,330 errors: 62.8% land in the same subsystem, 17.0% in the same system but a
different subsystem, and 20.2% cross systems (1.7% of all answerable records).

### Does the confidence band mean anything?

`training/batch_predict.py`, answerable records:

| Band | n | Accuracy | 95% CI |
|---|---|---|---|
| ≥ 70% | 14,795 | 0.946 | 0.943 – 0.950 |
| 30 – 70% | 980 | 0.525 | 0.493 – 0.556 |
| < 30% | 101 | 0.307 | 0.225 – 0.403 |

The bands separate cleanly, which is what the review-routing rule depends on. This is
**not** a claim that a displayed "76%" equals 76% empirical accuracy; no reliability
diagram or calibration error has been computed, and three bands are too coarse to
support a point claim.

### A second figure that is not yet reconciled

A frequency-weighted run over the full record file, excluding exact train/validation
text matches, gives top-1 0.916 (0.912–0.920) and top-3 0.980 (0.977–0.982),
n = 15,876. Its interval does not overlap the test-split figure above. The likely
cause is how that set is built rather than the model: repeated easy records count
more than once, and texts appearing in training under a different label are dropped.
**Until that is reconciled, 0.903 is the number to quote.**

### Baselines: not yet measured

No baseline is reported here, which is a gap, not an omission. Required before this
work supports any claim of value added:

1. How often the maintainer's originally typed WUC already matches the QC-corrected
   code. This is the decision-relevant baseline and it runs on CPU against existing
   data.
2. TF-IDF + logistic regression, to show what the transformer buys.
3. Majority class, as the floor.

---

## Verification and validation: the serving bug

The most important result in this project is a bug, not a score.

The model was trained with a custom forward pass that classifies from the CLS token.
An early serving path pooled token embeddings by mean instead. Both configurations
load, run, and return confident predictions. Re-measuring the model *as served*
rather than as trained exposed the gap:

| Pooling used at inference | Top-1 accuracy |
|---|---|
| Mean pooling (as served) | 0.756 |
| CLS token (as trained) | 0.897 |

Fourteen points of accuracy were disappearing silently. `training/check_pooling.py`
now guards the configuration, and the result is why every number in this README is
measured through the serving path.

Two related checks are worth stating:

- **Flat versus hierarchical model.** On 2,000 records the two models differ on 67
  one-sided disagreements, 38 to 29. An exact sign test gives p ≈ 0.33, so the
  models are statistically indistinguishable. The hierarchical model ships because it
  was already validated end to end, not because it is better.
- **A retracted claim.** An earlier version of this README argued the hierarchical
  model was "significantly better calibrated" from a 47% lower test loss. That
  comparison was invalid: the hierarchical loss is a weighted sum of three heads
  (0.2 system + 0.3 subsystem + 0.5 WUC) and is not on the same scale as the flat
  model's loss. The claim has been removed.

---

## Data

| Stage | Records |
|---|---|
| Source maintenance records | 17,041 in the app dataset |
| Answerable (label in class set, `MIN_PER_CLASS = 5`) | 15,876 |
| Classes | 1,251 |

The label is the QC `Corrected WUC` field. Input text is the discrepancy plus the
corrective action, so the model describes a **post-fix** record; predicting from a
discrepancy alone, before the fix is known, is a harder problem and a separate model.

The data and weights are CUI and cannot be distributed with this repository.

## Method

ModernBERT-large, fine-tuned with class-weighted cross-entropy plus auxiliary system
and subsystem heads (loss weights 0.2 / 0.3 / 0.5), max sequence length 128, 5 epochs,
checkpoint selected on validation macro-F1, seed 42. Training runs about an hour on an
RTX 6000 Ada (48 GB).

---

## Limitations

- **Labels are QC-corrected, not SME-adjudicated.** The direction of any residual
  label bias is unknown. Note that label noise can push measured accuracy in either
  direction: an earlier claim that the accuracy figure is "a floor" was wrong and has
  been removed.
- **The split is random, not temporal.** There is no later-in-time holdout, so drift
  in writing style is unmeasured.
- **Near-duplicate leakage is not controlled.** Only exact text matches are removed,
  so similar write-ups of the same job can appear in both train and test.
- **Post-fix text only.** Accuracy would be materially lower on discrepancy-only input.
- **6.8% of records are structurally unanswerable** with the current class set.
- **Single seed.** No variance across seeds or folds is reported.
- **The Tab 3 narrative is LLM-generated** and can fabricate specifics; the
  deterministic profile above it is the authoritative part.
- **No downtime or cost signal** exists in this data, so nothing here speaks to impact.

---

## The application

Three Streamlit tabs over one dataframe and one WUC→description map:

| Tab | What it does | Engine |
|---|---|---|
| Predict WUC | Discrepancy + corrective action → top-3 WUCs with confidence band | Fine-tuned ModernBERT-large (1,251 classes) |
| Query Records | Natural-language question → counts, monthly trend, top WUCs | Regex-parsed filters over pandas |
| WUC Profile | One WUC → deterministic profile (why / when / where / lifecycle / co-occurrence) plus an LLM summary | pandas + pluggable LLM adapter |

LLM adapters are pluggable: `NullAdapter` (templates, no network), `GemmaAdapter`
(local Ollama, nothing leaves the host, default), `ClaudeAdapter` (only if
`ANTHROPIC_API_KEY` is set; off by default).

## Quick start

```bash
pip install -r requirements.txt

# Point the app at the (gitignored, CUI) data file
export WUC_DATA_PATH=/full/path/to/app_data.csv

streamlit run main_app.py     # http://localhost:8501
```

`kc135_wuc_lookup_dictionary.csv` ships with the repo as the WUC-description fallback.
Deployment on the internal JupyterHub GPU box is documented in `docs/`.

### Reproducing the results

On authorized infrastructure, with the CUI dataset in place:

```bash
python training/prepare_data.py        # splits, class filtering
python training/train_hierarchical.py  # fine-tune (~60 min, RTX 6000 Ada)
python training/check_pooling.py       # guard: CLS vs mean pooling
python training/batch_predict.py       # accuracy, coverage, confidence bands
python training/error_analysis.py      # system/subsystem error structure
```

`requirements.txt` is not yet pinned; pin it before treating a rerun as a replication.

---

## Next steps

1. Measure the maintainer-typed-versus-QC baseline. Nothing else here means much
   without it.
2. Reconcile the 0.903 and 0.916 figures, or retire the second one.
3. Hand-label about 100 recent live submissions and measure on those; that is the
   number to trust for the live workflow.
4. Add a temporal holdout and seed-variance reporting.
5. Train and route a discrepancy-only variant for the pre-fix workflow.
6. Commit non-CUI evidence artifacts (metric JSONs with counts only) for each headline
   figure.

### Data backlog (added 2026-09-22)

A third extract (Feb 2025 – Jul 2026) arrived 2026-09-21. All raw data now lives in
one master folder, OneDrive `Documents/kc135/data/`, and `training/build_corpus.py`
merges it into one record per job: **163,145 records, 2019-01-01 → 2026-07-31**.

- [x] Commit `training/build_corpus.py`.
- [x] Point `build_app_data.py` and `prepare_data.py` at the combined corpus instead
      of their own data1 + data2 merge.
- [x] Check for train/test leakage from the old merge. About 6,000 jobs appeared in both
      data1 and data2, about 1,700 of them under two different Corrected WUC labels.
      **Result:** re-running the old split exactly (125,087 / 15,636 / 15,636) puts
      264 test rows (1.7%) in the same job as a train/val row, and 259 of those carry
      a *different* label than the training copy. That scores the model against the
      label it was not taught, so the 0.903 test figure is, if anything, slightly
      understated rather than inflated. The 0.9162 figure is unaffected:
      `--exclude-seen` already dropped the 266 test rows whose input text appears in
      train/val.
- [x] Build the temporal holdout. `prepare_data.py` now holds back every record on or
      after 2026-04-01, so 5,218 records (all of Apr–Jul 2026) go to
      `data_splits/temporal_holdout.parquet`, newer than anything any model has
      trained on.
- [ ] Score the deployed model on the holdout (GPU box):
      `WUC_MODEL_PATH=./wuc-model-hier python training/batch_predict.py --input data_splits/temporal_holdout.parquet --text-col text`
- [ ] Write `docs/EVAL_METRICS.md` from the cadet's agreed metric set, then report the
      holdout with it.
- [ ] Retrain on the full corpus only after the holdout and metrics are fixed.
- [x] Delete the duplicate raw copies outside `kc135/data/`, including the
      Excel-corrupted `kc135/kc_135.csv`. They went to the Recycle Bin; the original
      deliveries are kept in `kc135/data/originals/`.

## License and provenance

Code: MIT (see `LICENSE`). Author: Jonathan Day. The data, labels, and trained weights
are CUI and are not covered by that license; they are not distributed here.
