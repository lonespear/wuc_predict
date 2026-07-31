"""Side-by-side comparison: old jonday/wuc-model vs new wuc-model-hier.

Runs both models on the held-out test set and prints:
  - Per-model accuracy and macro F1 (truth: Corrected WUC)
  - Confidence distribution histograms
  - Disagreement breakdown — top-K cases where they differ, with the truth

This settles the empirical question: which model is more accurate, and how
honest is each one's confidence relative to its actual accuracy.

Run on the GPU box:
    python compare_models.py
"""
from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

OLD_MODEL = "jonday/wuc-model"
NEW_MODEL = "./wuc-model-hier"
TEST_PATH = Path("data_splits/test.parquet")
SAMPLE_SIZE = 2000     # subset for speed; set to None for full test set
MAX_LEN = 128
BATCH = 64


def predict_all(model_path: str, texts: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (predicted_indices, max_probs, id_to_label_dict)."""
    print(f"\nLoading {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    preds, confs = [], []
    n = len(texts)
    for i in range(0, n, BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        max_p, top_idx = torch.max(probs, dim=-1)
        preds.extend(top_idx.cpu().tolist())
        confs.extend(max_p.cpu().tolist())
        if (i // BATCH) % 10 == 0:
            print(f"  {min(i + BATCH, n)}/{n}")

    del model
    torch.cuda.empty_cache()
    return np.array(preds), np.array(confs), id2label


def main() -> None:
    df = pd.read_parquet(TEST_PATH)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    texts = df["text"].tolist()
    truth_wuc = df["Corrected WUC"].astype(str).str.upper().str.strip().tolist()
    print(f"Comparing on {len(df):,} held-out test examples")

    # OLD model
    old_preds_idx, old_confs, old_id2label = predict_all(OLD_MODEL, texts)
    old_preds_wuc = [old_id2label.get(int(i), "?") for i in old_preds_idx]

    # NEW model
    new_preds_idx, new_confs, new_id2label = predict_all(NEW_MODEL, texts)
    new_preds_wuc = [new_id2label.get(int(i), "?") for i in new_preds_idx]

    # Need predictions/truth in a common label space for sklearn — use string WUCs directly
    # (sklearn handles string labels fine for accuracy/macro-F1)
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD ACCURACY (truth: Corrected WUC)")
    print("=" * 60)
    print(f"OLD ({OLD_MODEL}):")
    print(f"  accuracy = {accuracy_score(truth_wuc, old_preds_wuc):.4f}")
    print(f"  macro F1 = {f1_score(truth_wuc, old_preds_wuc, average='macro', zero_division=0):.4f}")
    print(f"NEW ({NEW_MODEL}):")
    print(f"  accuracy = {accuracy_score(truth_wuc, new_preds_wuc):.4f}")
    print(f"  macro F1 = {f1_score(truth_wuc, new_preds_wuc, average='macro', zero_division=0):.4f}")

    # Confidence distributions
    print("\n" + "=" * 60)
    print("CONFIDENCE DISTRIBUTIONS (mean / median / std)")
    print("=" * 60)
    print(f"OLD: mean={old_confs.mean():.3f}  median={np.median(old_confs):.3f}  std={old_confs.std():.3f}")
    print(f"NEW: mean={new_confs.mean():.3f}  median={np.median(new_confs):.3f}  std={new_confs.std():.3f}")

    # Calibration: when each model is X% confident, how often is it actually right?
    print("\n" + "=" * 60)
    print("CALIBRATION (when model says X%, how often correct?)")
    print("=" * 60)
    old_correct = np.array([p == t for p, t in zip(old_preds_wuc, truth_wuc)])
    new_correct = np.array([p == t for p, t in zip(new_preds_wuc, truth_wuc)])
    bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.001)]
    print(f"{'Confidence band':<20} {'OLD acc / count':<25} {'NEW acc / count':<25}")
    for lo, hi in bins:
        old_mask = (old_confs >= lo) & (old_confs < hi)
        new_mask = (new_confs >= lo) & (new_confs < hi)
        old_n, new_n = int(old_mask.sum()), int(new_mask.sum())
        old_acc = old_correct[old_mask].mean() if old_n else float("nan")
        new_acc = new_correct[new_mask].mean() if new_n else float("nan")
        print(f"{lo:.2f}-{hi:.2f}".ljust(20)
              + f"{old_acc:.3f} / {old_n}".ljust(25)
              + f"{new_acc:.3f} / {new_n}".ljust(25))

    # Cases where they disagree — show first 15
    print("\n" + "=" * 60)
    print("DISAGREEMENTS (first 15)")
    print("=" * 60)
    print(f"{'TRUTH':<8} {'OLD':<8} {'OLD%':<7} {'NEW':<8} {'NEW%':<7}  TEXT")
    print("-" * 100)
    disagree = 0
    for i in range(len(df)):
        if old_preds_wuc[i] != new_preds_wuc[i]:
            disagree += 1
            if disagree <= 15:
                t = truth_wuc[i]
                snippet = texts[i][:50].replace("\n", " ")
                print(
                    f"{t:<8} "
                    f"{old_preds_wuc[i]:<8} {old_confs[i]*100:5.1f}%  "
                    f"{new_preds_wuc[i]:<8} {new_confs[i]*100:5.1f}%  "
                    f"{snippet}"
                )
    print(f"\nTotal disagreements: {disagree}/{len(df)} ({100*disagree/len(df):.1f}%)")

    # On disagreements, which model wins?
    both_disagree = np.array([o != n for o, n in zip(old_preds_wuc, new_preds_wuc)])
    if both_disagree.sum():
        old_right_on_disagree = ((np.array(old_preds_wuc) == np.array(truth_wuc)) & both_disagree).sum()
        new_right_on_disagree = ((np.array(new_preds_wuc) == np.array(truth_wuc)) & both_disagree).sum()
        print(f"On disagreements: OLD right {old_right_on_disagree}, NEW right {new_right_on_disagree}, "
              f"both wrong {both_disagree.sum() - old_right_on_disagree - new_right_on_disagree}")


if __name__ == "__main__":
    main()
