# Maintenance Precedent Assistant — cadet research project, AY26-27

**Sponsor system:** KC-135 WUC Maintenance Analytics (`lonespear/wuc_predict`)
**Status:** proposed 2026-07-31
**Effort:** two semesters, single cadet or a pair. A one-semester version is
described under *Reduced scope*.

---

## 1. The pitch, in a paragraph

A junior KC-135 crew chief facing an unfamiliar write-up has two resources: a
technical order that tells them what a part *is*, and an experienced maintainer
who remembers what happened last time. The second one does not scale and
retires. This project builds a tool that makes 162,565 historical maintenance
records answer the question *"has this happened before, and what fixed it?"* —
grounded in real corrective actions rather than generated advice, and anchored
to the authoritative Work Unit Code taxonomy from TO 1C-135-06.

---

## 2. What already exists (the cadet does not build this)

A deployed three-tab Streamlit platform on the school GPU box:

| Capability | State |
|---|---|
| WUC classifier (ModernBERT-large, 1,251 classes) | **0.9162 top-1 / 0.9797 top-3** on 15,876 held-out records; 98.31% system-level |
| Maintenance record query (Tab 2) | natural-language filters over the corpus |
| WUC profile (Tab 3) | burden index, base concentration, trend, seasonality, lifecycle, co-occurrence |
| Corpus | 162,565 records, 2019-01-01 → 2026-03-31, ~70 bases |
| Local LLM | `gemma4:31b` via Ollama, fully offline |

Read `CLAUDE.md` and `GLIDEPATH.md` in the repo root before writing any code.
They record what was built, what was deliberately *not* built, and why.

**The corpus fields that matter here:** `Discrepancy`, `Corrective Action`,
`WCE Narrative`, `Corrected WUC`, `Tail Number`, `Base`, `Start Date`,
`Labor` (man-hours), `JCN`, `Flight Hours`, `When Discovered Code`,
`Type Maint Code`.

---

## 3. The gap

The classifier is a **paperwork** tool. It requires the corrective action as
input, so it only works *after* the job is done — it fills in a WUC field on a
completed record. It cannot help someone standing at the aircraft.

Nothing in the system answers the question a junior maintainer actually has.

---

## 4. Research questions

**RQ1 — Retrieval.** For a free-text maintenance discrepancy, can historical
records be retrieved that a subject-matter expert would judge relevant? How do
lexical methods (TF-IDF, BM25) compare to dense embedding retrieval on this
corpus, which is short, abbreviation-heavy, and highly domain-specific
(`R2`, `O/C/G`, `REQ`, `(X)`)?

**RQ2 — Context.** Does conditioning retrieval on aircraft context — same
tail, same base, same season, similar airframe hours — surface more relevant
precedent than text similarity alone?

**RQ3 — Evaluation without experts.** SME time is the binding constraint (see
§9). What proxy measures of retrieval quality can be computed automatically,
and how well do they agree with the limited expert judgment available?

RQ3 is the genuinely hard one and the most publishable. The sponsor system hit
exactly this wall: its accuracy figure is defensible but not hand-verified,
because no maintainer was available to label 100 records.

---

## 5. Scope

### In scope

- Retrieval over historical discrepancy → corrective-action pairs
- Parsing the WUC hierarchy from TO 1C-135-06 (338 pages, text-extractable)
- Faceting retrieval by tail, base, season, airframe hours
- A fourth tab in the existing Streamlit app
- An evaluation protocol and its results

### Explicitly out of scope

- **Fault isolation / diagnostic procedures.** These live in the `-2` series
  maintenance manuals, which are not in hand. **The WUC hierarchy is a parts
  taxonomy, not a diagnostic sequence** — knowing `45175` sits under hydraulics
  says where a part lives, not what to check first. Presenting taxonomy
  navigation as a troubleshooting tree would mislead the exact user this
  project is for.
- **Generating repair instructions.** The tool surfaces what *was recorded*;
  it never tells anyone what to do. See §8.
- Retraining or modifying the WUC classifier.
- Any change to Tabs 1–3.

---

## 6. Technical approach

### Stage A — Retrieval baseline *(target: week 5)*

Index the corpus on `Discrepancy` (and optionally `WCE Narrative`). Given a
typed symptom, return the *k* nearest historical records with their corrective
actions, labor hours, and WUC.

Start with TF-IDF/BM25. It is a genuine contender here, not a strawman: the
vocabulary is small and highly conventionalized, and exact abbreviation matches
carry real signal. Establish it as the baseline before reaching for embeddings.

### Stage B — Hierarchy from the tech order *(target: week 9)*

Parse TO 1C-135-06 into a system → subsystem → component tree keyed by WUC.
Extraction is clean (~1,200 chars/page, not scanned). Validate the parse
against the existing `codes.json` and `kc135_wuc_lookup_dictionary.csv` —
disagreements are findings, since those files' provenance is undocumented.

Also resolve `kc135_wuc_lookup_levels.csv`, a teammate's file believed to carry
richer level columns (logged as an open item in `GLIDEPATH.md`).

### Stage C — Context-conditioned retrieval *(target: week 14)*

Re-rank or filter by tail, base, month, and flight-hour band. Answers RQ2. Much
of the machinery exists in `wuc_profile.py` — reuse rather than rebuild.

### Stage D — Dense retrieval comparison *(spring)*

Sentence-embedding retrieval versus the Stage A baseline, evaluated with the
Stage-A protocol. Candidate encoders are already on the box. **Report the
comparison honestly, including if the lexical baseline wins** — that is a
legitimate and likely result on a corpus this conventionalized.

### Stage E — Synthesis and interface *(spring)*

`gemma4:31b` summarizes retrieved precedent into a readable brief. Constrain it
to quoting retrieved records; hallucination has already been observed in this
system, including a 31B inventing WUC codes that did not exist in its input.

---

## 7. Evaluation plan

This is where the project earns the word *research*.

**Automatic proxies** (compute all, report all):

- *WUC agreement* — does the retrieved record share the query's true WUC?
  Subsystem and system-level agreement give partial credit, as in
  `training/error_analysis.py`.
- *Corrective-action overlap* — token/phrase overlap between the retrieved
  fix and the query's actual fix.
- *Labor plausibility* — is the retrieved job's man-hour figure predictive of
  the query's?

**Expert validation** (small, targeted):

Budget **≤30 minutes** of maintainer time. Do not design an evaluation that
needs more; the sponsor project proved that a 102-record hand-labeling task
never got done for exactly this reason. Sample where disagreement is highest,
not randomly — `training/batch_predict.py --worksheet` and
`training/error_analysis.py` both demonstrate the stratified pattern.

**The RQ3 deliverable:** correlation between the automatic proxies and the
expert judgments. If a proxy tracks expert opinion, future work can be
evaluated without SMEs at all — which is worth more than the tool.

---

## 8. Safety and framing (non-negotiable)

A tool advising a junior maintainer on an aircraft carries a different
liability posture than one filling a paperwork field.

- The interface presents **"here is what was recorded for similar write-ups"** —
  never "do this."
- Retrieved text is displayed **verbatim with its record count**. Paraphrase
  invites invention.
- The technical order is authoritative. This tool is precedent. That must be
  stated in the UI, not merely understood by whoever built it.
- Any LLM-generated text is labeled as generated and is traceable to the
  records it drew from.

---

## 9. Prerequisites and access

| Need | Notes |
|---|---|
| CUI handling / NIPR account | Start early — the usual long pole |
| GPU box account (`icsarl.westpoint.edu`) | Python 3.12 venv at `~/.venvs/wuc`; see `CLAUDE.md` gotchas |
| Repo access | `github.com/lonespear/wuc_predict` |
| SME point of contact | ~30 min, twice. Identify by week 4 or the evaluation slips |
| TO 1C-135-06 | In hand. Export-controlled — read the disclosure notice |
| TO `-2` series | **Not in hand.** Nice-to-have; do not let it block A–E |

---

## 10. Milestones

### Fall

| Week | Milestone |
|---|---|
| 2 | Environment running; existing app rebuilt and Tab 3 reproduced from `CLAUDE.md` alone |
| 4 | Corpus characterized; SME contact identified; evaluation protocol drafted |
| 5 | **Stage A** — retrieval baseline returning results |
| 7 | Automatic proxy metrics implemented, baseline scored |
| 9 | **Stage B** — TO hierarchy parsed and validated against `codes.json` |
| 12 | **Stage C** — context-conditioned retrieval |
| 14 | First SME session (≤30 min); RQ3 correlation analysis |
| 15 | Fall report + brief |

### Spring

| Week | Milestone |
|---|---|
| 3 | **Stage D** — dense retrieval implemented |
| 6 | Head-to-head evaluation, lexical vs dense |
| 9 | **Stage E** — synthesis layer and Tab 4 |
| 11 | Second SME session; end-to-end walkthrough |
| 13 | Technical report draft |
| 15 | Final report, poster, deployed tool |

---

## 11. Deliverables

1. **Tab 4** in the deployed app, running on the GPU box
2. **Technical report** — approach, evaluation, honest negative results
3. **Poster / brief** for the research day
4. **Evaluation harness** committed to the repo, reusable
5. **Updated `CLAUDE.md` and `GLIDEPATH.md`** so the next person inherits state

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| **SME never materializes** | Automatic proxies are the primary evaluation; expert input validates them. Design so the project completes without it. This risk has already been realized once on the sponsor project. |
| CUI onboarding delays | Stage A works on de-identified text; start there |
| Dense retrieval underperforms lexical | That is a **result**, not a failure. Report it. |
| TO parse is messier than expected | Validate against `codes.json` early — week 9 exists to catch this |
| Scope creep toward fault isolation | §5 is explicit. The data does not support it. |

---

## 13. Learning outcomes

Information retrieval and its evaluation; working with a real, messy,
non-public corpus; evaluation design under expert scarcity; retrieval-augmented
generation and its failure modes; human-factors framing for safety-relevant
tools; and inheriting, reading, and extending a production codebase rather than
starting from a blank file.

---

## 14. First week

```bash
# on the box
cd ~/wuc_predict && git pull
~/.venvs/wuc/bin/python training/build_app_data.py     # rebuild app_data.csv
export WUC_MODEL_PATH=./wuc-model-hier
~/.venvs/wuc/bin/python training/error_analysis.py     # see how the model fails
```

Then read `CLAUDE.md` end to end, especially **Gotchas**. It documents a
14-point production accuracy loss that went unnoticed for three months because
a config value said `mean` where the training code did `cls`, and a dataset
whose dates all landed in January 1970 because Excel serials were read as
nanoseconds. Both were found by measuring rather than trusting.

**That habit is the actual curriculum.**
