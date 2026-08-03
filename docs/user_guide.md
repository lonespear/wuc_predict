## Purpose

This tool assigns Work Unit Codes to completed maintenance actions and
analyzes the KC-135 maintenance record history. It runs entirely on this
server. No record leaves the machine.

Three capabilities: code assignment, record search, and single-code analysis.

---

## Predict WUC — assign a code

Enter the discrepancy and the corrective action. The tool returns three
candidate Work Unit Codes with confidence percentages.

Write as the record is written: uppercase, terse, technical.

**Example**

| Field | Entry |
|---|---|
| Discrepancy | `PILOT SEAT BELT FRAYED, MISSING STITCHING` |
| Corrective Action | `INSPECTED PILOT SEAT BELT, REPLACED IAW TM 1C-135-06` |
| Result | **12AAN — FUSELAGE COMPARTMENTS / Safety Belt · 99.8%** |

Add WCE Narrative, How Mal, and Action Taken when you have them. Each field
raises accuracy.

**Read the confidence**

| Band | Action |
|---|---|
| 🟢 70% and above | Correct 95% of the time. Enter it. |
| 🟡 30 – 70% | Compare all three candidates before entering. |
| 🔴 Below 30% | Re-check your entry against record format. |

Review all three candidates every time. The correct code appears in the top
three 98% of the time. Candidates two and three are usually adjacent codes —
opposite side of a component, or a specific code where a general one applies.

The tool reads the corrective action heavily. Enter both fields.

---

## Query Records — search the history

Ask in plain language. The parser reads tail numbers, bases, Work Unit Codes,
date ranges, and seasons.

```
all issues in 2024
WUC 12AAN at McConnell
tail 61-0313 last 6 months
hydraulic problems in winter
```

The tool echoes the filter it applied above the results. Verify it matches
your intent.

Results export to CSV. Send any Work Unit Code from the results directly to
the Profile tab.

---

## WUC Profile — analyze one code

Enter a Work Unit Code. The tool returns maintenance burden, geographic
concentration, trend, seasonality, airframe age distribution, discovery
method, and associated codes.

The charts and tables compute directly from the records. The written summary
at the top is machine-generated. Where the two disagree, use the numbers.

### Cost per event

Labor hours per occurrence, measured against the fleet average of 4.4 hours.

- **Above 1.0×** — costs more labor per occurrence than the average
  maintenance action.
- **Below 1.0×** — costs less.

This separates expensive codes from merely frequent ones. WUC 12AAN appears
171 times at 0.30× — high volume, low cost, low priority. A code appearing 20
times at 4.0× consumes more of your maintenance capacity.

### Concentration

Records at a base, measured against that base's total maintenance volume.

- **Near 1.0×** — matches what the base's workload predicts. Normal.
- **Above 1.5×** — occurs more often than the base's volume accounts for.
  Investigate.

Large bases lead every raw count. This adjusts for size. On WUC 12AAN,
McConnell records the most events and rates 1.0×. Birmingham, Pittsburgh, and
Gen Mitchell each exceed 2.5×.

### Remaining panels

The calendar heat map shows month and year in one view. Airframe lifecycle
compares against fleet quartiles — 25% per band means no age pattern.
Discovery phase shows how these are found. Associated codes share a job
control number.

---

## Accuracy

Measured on 15,876 records the model had not seen, with all training data
removed.

| Measure | Result |
|---|---|
| Correct code, first candidate | **91.6%** |
| Correct code, top three | **98.0%** |
| Correct system (first two characters) | **98.3%** |

Errors stay close to the answer. 63% land in the correct subsystem. 1.7% land
in the wrong system.

---

## Coverage and confirmed limits

**The model covers 1,251 codes — 93.2% of codes in current use.** Codes
appearing fewer than five times in the source data were excluded from
training. When no candidate fits, the code may be outside coverage. Use the
technical order.

**Labor hours are the only cost measure.** Start and stop dates match on 99.3%
of records, so this data supports no downtime or NMC figure.

**Accuracy is measured against existing record labels.** A maintainer has not
independently verified them. If those labels contain errors, true accuracy is
higher than stated.

**The written summary comes from a language model and occasionally states a
figure not present in the data.** The tables and charts do not.

**TO 1C-135-06 is authoritative. This tool supports the record, not the
repair.**

---

## Data

Two maintenance extracts, merged and de-duplicated. 38% of combined rows
appeared in both.

Report errors or questions:
[`github.com/lonespear/wuc_predict`](https://github.com/lonespear/wuc_predict)
