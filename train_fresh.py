"""Fresh fine-tune of a SOTA encoder on the cleaned WUC splits.

Reads from ./data_splits/ (created by prepare_data.py). Defaults to
ModernBERT-base — drop-in replacement for bert-base-uncased that
typically wins on text classification benchmarks.

Outputs:
    ./wuc-model-v2/
        config.json, model.safetensors, tokenizer files
        wuc_mapping.json   (label -> id)
        test_metrics.json  (final test-set scores)

Run on the GPU box (RTX 6000 Ada handles this comfortably):
    python train_fresh.py
"""
from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# =============================================================================
# Config
# =============================================================================
BASE_MODEL = "answerdotai/ModernBERT-base"   # alternatives: bert-base-uncased,
                                              # microsoft/deberta-v3-base,
                                              # answerdotai/ModernBERT-large
DATA_DIR = Path("data_splits")
OUT_DIR = Path("wuc-model-v2")
MAX_LEN = 128
NUM_EPOCHS = 5
TRAIN_BS = 32
EVAL_BS = 64
LR = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42


def main() -> None:
    # =========================================================================
    # 1. Load splits + label mapping
    # =========================================================================
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    wuc_to_id: dict[str, int] = json.load(open(DATA_DIR / "wuc_mapping.json"))
    id_to_wuc = {v: k for k, v in wuc_to_id.items()}
    n_labels = len(wuc_to_id)

    # Map labels — drop rows whose WUC isn't in the mapping (split-orphans)
    for df in (train_df, val_df, test_df):
        df["label"] = df["Corrected WUC"].map(wuc_to_id)
    train_df = train_df.dropna(subset=["label"]).copy()
    val_df = val_df.dropna(subset=["label"]).copy()
    test_df = test_df.dropna(subset=["label"]).copy()
    for df in (train_df, val_df, test_df):
        df["label"] = df["label"].astype(int)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Classes: {n_labels}")

    # =========================================================================
    # 2. Tokenizer + tokenize
    # =========================================================================
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch: dict) -> dict:
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    cols = ["text", "label"]
    train_ds = Dataset.from_pandas(train_df[cols]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[cols]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[cols]).map(tokenize, batched=True)

    # =========================================================================
    # 3. Class weights for imbalance — inverse frequency, normalized
    # =========================================================================
    counts = train_df["label"].value_counts().sort_index()
    full_counts = np.zeros(n_labels, dtype=np.float64)
    for idx, c in counts.items():
        full_counts[int(idx)] = float(c)
    full_counts = np.maximum(full_counts, 1.0)  # avoid div-by-zero for split-orphans
    weights = 1.0 / full_counts
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights → min: {weights.min():.3f}, "
          f"max: {weights.max():.3f}, mean: {weights.mean():.3f}")

    # =========================================================================
    # 4. Model
    # =========================================================================
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=n_labels,
        id2label=id_to_wuc,
        label2id=wuc_to_id,
    )

    # =========================================================================
    # 5. Custom Trainer with class-weighted CE
    # =========================================================================
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=weights_t.to(logits.device)
            )
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # =========================================================================
    # 6. Metrics — track macro-F1 (long-tail) + accuracy + weighted-F1
    # =========================================================================
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    # =========================================================================
    # 7. Training arguments
    # =========================================================================
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        seed=SEED,
        logging_steps=200,
    )

    # =========================================================================
    # 8. Train
    # =========================================================================
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # =========================================================================
    # 9. Final eval on the held-out test set
    # =========================================================================
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("\n=== FINAL TEST RESULTS ===")
    for k, v in test_results.items():
        print(f"{k}: {v}")

    # =========================================================================
    # 10. Save model + tokenizer + label map + metrics
    # =========================================================================
    OUT_DIR.mkdir(exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    with open(OUT_DIR / "wuc_mapping.json", "w") as f:
        json.dump(wuc_to_id, f, indent=2)
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in test_results.items()}, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/")
    print("To push to Hugging Face:")
    print("    huggingface-cli login")
    print("    python -c \"from transformers import AutoModelForSequenceClassification, AutoTokenizer; "
          "m = AutoModelForSequenceClassification.from_pretrained('./wuc-model-v2'); "
          "t = AutoTokenizer.from_pretrained('./wuc-model-v2'); "
          "m.push_to_hub('jonday/wuc-model'); t.push_to_hub('jonday/wuc-model')\"")


if __name__ == "__main__":
    main()
