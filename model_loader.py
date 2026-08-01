"""WUC predictor — loads a fine-tuned classifier from a configurable path.

WUC_MODEL_PATH is REQUIRED and must point at a local checkpoint directory
(e.g. produced by train_hierarchical.py). Importing without it raises. There
is deliberately no default: the old fallback to `jonday/wuc-model` loaded a
different label space and returned confident wrong answers. That repo was
deleted from Hugging Face on 2026-07-31.

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
MODEL_PATH = os.environ.get("WUC_MODEL_PATH")
if not MODEL_PATH:
    # Previously this defaulted to "jonday/wuc-model" — a different checkpoint
    # with a different label space (1727 vs 1251 classes) whose config carries
    # only HF's placeholder LABEL_0..LABEL_N in id2label. It loaded without
    # complaint and returned strings like "LABEL_847" as WUCs, or plausible
    # wrong codes. Failing loudly beats serving the wrong model silently.
    raise RuntimeError(
        "WUC_MODEL_PATH is not set. Point it at the deployed checkpoint:\n"
        "    export WUC_MODEL_PATH=./wuc-model-hier\n"
        "Set it explicitly to a legacy model if that is genuinely what you want."
    )

# MUST match train_hierarchical.py / train_fresh.py MAX_LEN. Without an
# explicit max_length the tokenizer falls back to tokenizer.model_max_length
# (8192 for ModernBERT), feeding the classifier sequences up to 64x longer
# than anything it saw in training. Silent accuracy loss on long write-ups.
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
# Streamlit launches without touching the device, so inference silently ran on
# CPU. _model_device() below keeps input tensors on whatever device the model
# actually lives on, so this is purely a speed fix — predictions are identical.
model.to("cuda" if torch.cuda.is_available() else "cpu")
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

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
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

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
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

    Values are coerced with str() rather than guarded by isinstance(str),
    matching prepare_data.py's `if pd.notna(v) and str(v).strip()`. The old
    isinstance guard was fine for Streamlit — widgets always return str — but
    silently DROPPED short code columns like How Mal and Action Taken when a
    caller passed them from a pandas frame that parsed them as numeric,
    building shorter text than training used.
    """
    parts = []
    for value in (discrepancy, corrective_action, wce_narrative, how_mal, action_taken):
        if value is None:
            continue
        text = str(value).strip()
        # "nan" is what str() makes of a pandas NaN; prepare_data.py excluded
        # those via pd.notna(). A literal NAN write-up is not a real WUC input.
        if not text or text.lower() == "nan":
            continue
        parts.append(text.upper())
    return " [SEP] ".join(parts)
