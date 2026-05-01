"""WUC predictor — loads a fine-tuned classifier from a configurable path.

Set WUC_MODEL_PATH to point at a local checkpoint directory (e.g. produced by
train_hierarchical.py). Defaults to the legacy HF model for backward compat.

The active model expects text formatted as:
    "<discrepancy> [SEP] <corrective_action>"
matching prepare_data.py's training format. predict_discrepancy() accepts a
single string; the caller is responsible for the [SEP] join when both fields
are available.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configurable model path — point at the local hierarchical checkpoint on the
# GPU box: export WUC_MODEL_PATH=./wuc-model-hier
MODEL_PATH = os.environ.get("WUC_MODEL_PATH", "jonday/wuc-model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Build the index→WUC map. Prefer model.config (modern checkpoints have
# id2label baked in); fall back to the legacy wuc_mapping.json shipped with
# the original BERT-base model.
if model.config.id2label and len(model.config.id2label) == model.config.num_labels:
    index_to_wuc = {int(k): str(v) for k, v in model.config.id2label.items()}
else:
    with open("wuc_mapping.json", "r") as f:
        wuc_mapping = json.load(f)
    index_to_wuc = {v: k for k, v in wuc_mapping.items()}

# Lookups for the human-readable response
with open("codes.json", "r") as f:
    wuc_defs = json.load(f)
with open("main_system.json", "r") as f:
    main_system = json.load(f)


def _model_device() -> torch.device:
    return next(model.parameters()).device


def predict_discrepancy(text: str, method: int = 1):
    """Top-1 WUC prediction.

    method=1 -> formatted string
    method=2 -> tuple (wuc, definition, system, confidence_pct)
    """
    if not isinstance(text, str) or not text.strip():
        return "Invalid input"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(_model_device()) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = int(torch.argmax(outputs.logits, dim=1).item())
    confidence = float(probs[0, predicted].item()) * 100.0

    wuc = index_to_wuc.get(predicted, "Unknown WUC")
    definition = wuc_defs.get(wuc, "Unknown Definition")
    system = main_system.get(wuc[:2], "Unknown Main System")
    if method == 1:
        return f"{wuc}: {system}, {definition} (Confidence: {confidence:.2f}%)"
    return wuc, definition, system, confidence


def predict_top_k(text: str, k: int = 3) -> list[dict]:
    """Top-k WUC predictions with confidences.

    Returns a list of dicts: {wuc, definition, system, confidence}
    sorted by confidence descending. Useful for surfacing model uncertainty
    in the UI rather than blindly trusting the top-1.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k_: v.to(_model_device()) for k_, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=min(k, probs.shape[0]))

    results = []
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        wuc = index_to_wuc.get(int(idx), "Unknown WUC")
        results.append({
            "wuc": wuc,
            "definition": wuc_defs.get(wuc, "Unknown Definition"),
            "system": main_system.get(wuc[:2], "Unknown Main System"),
            "confidence": float(p) * 100.0,
        })
    return results


def build_input_text(discrepancy: str, corrective_action: str = "") -> str:
    """Combine discrepancy + corrective action into the [SEP]-joined format
    used at training time. Returns just the discrepancy if no action provided.
    """
    parts = []
    if discrepancy and discrepancy.strip():
        parts.append(discrepancy.strip())
    if corrective_action and corrective_action.strip():
        parts.append(corrective_action.strip())
    return " [SEP] ".join(parts)
