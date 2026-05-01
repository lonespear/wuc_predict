"""Hierarchical fine-tune: joint prediction of system (2-char), subsystem
(3-char), and full WUC (5-char) over a shared ModernBERT-large encoder.

Loss = α * L_system + β * L_subsystem + γ * L_wuc

The system/subsystem heads act as auxiliary supervision: they push the encoder
toward representations that respect the WUC hierarchy, which typically improves
macro-F1 on the long-tail full-WUC task by 2-5 points vs flat classification.

At save time, only the encoder + WUC head are persisted as a standard
AutoModelForSequenceClassification — the auxiliary heads are training-only and
don't ship. The saved model is a drop-in replacement for the v2 baseline.

Run on the GPU box:
    nohup python train_hierarchical.py > train_hier.log 2>&1 &
    tail -f train_hier.log
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
import torch.nn as nn
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
BASE_MODEL = "answerdotai/ModernBERT-large"
DATA_DIR = Path("data_splits")
OUT_DIR = Path("wuc-model-hier")
MAX_LEN = 128
NUM_EPOCHS = 5
TRAIN_BS = 32
EVAL_BS = 64
LR = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

# Hierarchical loss weights (sum to 1.0 by convention)
ALPHA_SYS = 0.20      # system  (2-char prefix)
BETA_SUB = 0.30       # subsystem (3-char prefix)
GAMMA_WUC = 0.50      # full WUC (5-char) — primary objective


# =============================================================================
# Hierarchical model — wraps a standard SequenceClassification model and adds
# two auxiliary heads operating on the same pooled representation.
# =============================================================================
class HierarchicalModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        n_systems: int,
        n_subsystems: int,
        n_wucs: int,
        wuc_weights: torch.Tensor,
        id2label: dict,
        label2id: dict,
    ):
        super().__init__()
        # The wuc_model holds the encoder + ModernBERT prediction head + WUC classifier.
        # save_pretrained on this child is what we'll persist for inference.
        self.wuc_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=n_wucs,
            id2label=id2label,
            label2id=label2id,
        )
        hidden = self.wuc_model.config.hidden_size
        # Auxiliary heads operate on the post-prediction-head pooled representation
        self.system_head = nn.Linear(hidden, n_systems)
        self.subsystem_head = nn.Linear(hidden, n_subsystems)
        self.register_buffer("wuc_weights", wuc_weights)

    def _pooled(self, input_ids, attention_mask):
        """Run encoder + ModernBERT prediction head, return [CLS] pooled vector."""
        encoder_out = self.wuc_model.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = encoder_out.last_hidden_state[:, 0, :]
        return self.wuc_model.head(cls)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        system_label=None,
        subsystem_label=None,
        **kwargs,
    ):
        pooled = self._pooled(input_ids, attention_mask)
        wuc_logits = self.wuc_model.classifier(pooled)

        loss = None
        if labels is not None:
            ce = nn.CrossEntropyLoss()
            ce_weighted = nn.CrossEntropyLoss(weight=self.wuc_weights)
            loss_wuc = ce_weighted(wuc_logits, labels)
            loss_sys = ce(self.system_head(pooled), system_label) if system_label is not None else 0.0
            loss_sub = ce(self.subsystem_head(pooled), subsystem_label) if subsystem_label is not None else 0.0
            loss = ALPHA_SYS * loss_sys + BETA_SUB * loss_sub + GAMMA_WUC * loss_wuc

        return {"loss": loss, "logits": wuc_logits}


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    # 1. Load splits
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    wuc_to_id: dict[str, int] = json.load(open(DATA_DIR / "wuc_mapping.json"))
    id_to_wuc = {v: k for k, v in wuc_to_id.items()}
    n_wucs = len(wuc_to_id)

    # 2. Map WUC label and derive system/subsystem strings
    for df in (train_df, val_df, test_df):
        df["label"] = df["Corrected WUC"].map(wuc_to_id)
        df["system_str"] = df["Corrected WUC"].str[:2]
        df["subsystem_str"] = df["Corrected WUC"].str[:3]
    train_df = train_df.dropna(subset=["label"]).copy()
    val_df = val_df.dropna(subset=["label"]).copy()
    test_df = test_df.dropna(subset=["label"]).copy()

    # 3. Hierarchical label maps from train+val (test stays unseen)
    sys_labels = sorted(pd.concat([train_df, val_df])["system_str"].unique())
    sub_labels = sorted(pd.concat([train_df, val_df])["subsystem_str"].unique())
    sys_to_id = {s: i for i, s in enumerate(sys_labels)}
    sub_to_id = {s: i for i, s in enumerate(sub_labels)}
    for df in (train_df, val_df, test_df):
        df["system_label"] = df["system_str"].map(sys_to_id)
        df["subsystem_label"] = df["subsystem_str"].map(sub_to_id)
    # Drop rows with unmappable system/subsystem (test orphans)
    train_df = train_df.dropna(subset=["system_label", "subsystem_label"]).copy()
    val_df = val_df.dropna(subset=["system_label", "subsystem_label"]).copy()
    test_df = test_df.dropna(subset=["system_label", "subsystem_label"]).copy()
    for df in (train_df, val_df, test_df):
        df["label"] = df["label"].astype(int)
        df["system_label"] = df["system_label"].astype(int)
        df["subsystem_label"] = df["subsystem_label"].astype(int)

    n_systems = len(sys_labels)
    n_subsystems = len(sub_labels)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Classes — System: {n_systems} | Subsystem: {n_subsystems} | WUC: {n_wucs}")
    print(f"Loss weights — α(sys)={ALPHA_SYS}, β(sub)={BETA_SUB}, γ(wuc)={GAMMA_WUC}")

    # 4. Class weights for WUC head (long-tail handling)
    counts = train_df["label"].value_counts().sort_index()
    full_counts = np.zeros(n_wucs, dtype=np.float64)
    for idx, c in counts.items():
        full_counts[int(idx)] = float(c)
    full_counts = np.maximum(full_counts, 1.0)
    weights = 1.0 / full_counts
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32)

    # 5. Tokenize
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch: dict) -> dict:
        return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    cols = ["text", "label", "system_label", "subsystem_label"]
    train_ds = Dataset.from_pandas(train_df[cols]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[cols]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[cols]).map(tokenize, batched=True)

    # 6. Build model
    model = HierarchicalModel(
        BASE_MODEL,
        n_systems=n_systems,
        n_subsystems=n_subsystems,
        n_wucs=n_wucs,
        wuc_weights=weights_t,
        id2label=id_to_wuc,
        label2id=wuc_to_id,
    )

    # 7. Metrics — primary head only (WUC); aux heads are training-time scaffolding
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        }

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
        # Tell Trainer that "labels" is the primary target; system_label and
        # subsystem_label get passed through as kwargs to forward() since they
        # match the model's signature.
        label_names=["labels"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 8. Final test eval (held-out)
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("\n=== HIERARCHICAL TEST RESULTS ===")
    for k, v in test_results.items():
        print(f"{k}: {v}")

    # 9. Save deployable model — only encoder + WUC head, as a standard
    #    AutoModelForSequenceClassification. Auxiliary heads are dropped.
    OUT_DIR.mkdir(exist_ok=True)
    model.wuc_model.save_pretrained(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    with open(OUT_DIR / "wuc_mapping.json", "w") as f:
        json.dump(wuc_to_id, f, indent=2)
    with open(OUT_DIR / "system_mapping.json", "w") as f:
        json.dump(sys_to_id, f, indent=2)
    with open(OUT_DIR / "subsystem_mapping.json", "w") as f:
        json.dump(sub_to_id, f, indent=2)
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in test_results.items()}, f, indent=2)
    print(f"\nSaved deployable model to {OUT_DIR}/")
    print("(auxiliary system/subsystem heads were training-only and are not persisted)")


if __name__ == "__main__":
    main()
