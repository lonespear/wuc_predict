"""WUC predictor — loads a fine-tuned classifier from a configurable path.

Set WUC_MODEL_PATH to point at a local checkpoint directory (e.g. produced by
train_hierarchical.py). Defaults to the legacy HF model for backward compat.

The active model expects text formatted as:
    "<discrepancy> [SEP] <corrective_action> [SEP] <wce_narrative> [SEP] <how_mal> [SEP] <action_taken>"
(only non-empty fields, each uppercased) matching prepare_data.py's training
format. Use build_input_text() to construct it; predict_discrepancy() /
predict_top_k() accept the already-built string.
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

# Lookups for the human-readable response.
# Primary source: codes.json. Fallback: kc135_wuc_lookup_dictionary.csv (ships
# with the repo) — covers WUCs the model predicts that aren't in codes.json so
# the UI shows a real description instead of "Unknown Definition".
with open("codes.json", "r") as f:
    wuc_defs = json.load(f)
try:
    import pandas as _pd

    _lookup = _pd.read_csv("kc135_wuc_lookup_dictionary.csv")
    _cols = list(_lookup.columns)
    _code_col = "wuc_code" if "wuc_code" in _cols else _cols[0]
    _desc_col = "description" if "description" in _cols else _cols[1]
    _csv_defs = dict(
        zip(_lookup[_code_col].astype(str), _lookup[_desc_col].astype(str))
    )
    # codes.json wins; CSV fills the gaps.
    wuc_defs = {**_csv_defs, **wuc_defs}
except FileNotFoundError:
    pass
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
    definition = wuc_defs.get(wuc, f"(no dictionary entry for {wuc})")
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
            "definition": wuc_defs.get(wuc, f"(no dictionary entry for {wuc})"),
            "system": main_system.get(wuc[:2], "Unknown Main System"),
            "confidence": float(p) * 100.0,
        })
    return results


def build_input_text(
    discrepancy,
    corrective_action="",
    wce_narrative="",
    how_mal="",
    action_taken="",
) -> str:
    """Combine the five training text fields into the [SEP]-joined, UPPERCASE
    format used at training time (see prepare_data.py TEXT_FIELDS).

    Field order: Discrepancy, Corrective Action, WCE Narrative, How Mal,
    Action Taken. Only non-empty fields (after strip) are included; each part
    is uppercased before joining with " [SEP] ", because the training text is
    maintenance-report style (all caps, terse, technical). Returns a single
    string.
    """
    parts = []
    for value in (discrepancy, corrective_action, wce_narrative, how_mal, action_taken):
        if isinstance(value, str) and value.strip():
            parts.append(value.strip().upper())
    return " [SEP] ".join(parts)
