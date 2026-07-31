"""Score the checkpoint under both pooling modes to confirm a train/serve mismatch.

train_hierarchical.py pools by hand (line ~93):

    cls = encoder_out.last_hidden_state[:, 0, :]
    return self.wuc_model.head(cls)

i.e. CLS pooling. But it saves via `model.wuc_model.save_pretrained()`, and the
resulting config carries `classifier_pooling` from the base checkpoint. If that
value is "mean", then every downstream consumer —
AutoModelForSequenceClassification, and therefore model_loader.py,
batch_predict.py, compare_models.py and Tab 1 — mean-pools across tokens
before `head`, feeding the classifier a representation it never trained on.

No error is raised. The predictions stay plausible; they are just worse.

This script changes nothing on disk. It loads the checkpoint twice, overriding
`classifier_pooling` in memory, and reports top-1 on the held-out test split
each way.

Usage:
    export WUC_MODEL_PATH=./wuc-model-hier
    python training/check_pooling.py [N_ROWS]     # default 3000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

MODEL_PATH = os.environ.get("WUC_MODEL_PATH", "./wuc-model-hier")
TEST_PATH = REPO_ROOT / "data_splits" / "test.parquet"
MAX_LEN = 128  # matches train_hierarchical.py
BATCH = 64


def score(pooling: str, texts: list[str], truth: pd.Series, device: torch.device) -> float:
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    cfg.classifier_pooling = pooling
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=cfg)
    model.to(device).eval()
    id2 = {int(k): str(v) for k, v in model.config.id2label.items()}

    preds: list[str] = []
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(texts[i:i + BATCH], return_tensors="pt", truncation=True,
                        padding=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds += [id2.get(int(j), "?").strip().upper() for j in logits.argmax(-1).tolist()]
        print(f"  {pooling}: {min(i + BATCH, len(texts)):,}/{len(texts):,}", end="\r", flush=True)
    print()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return accuracy_score(truth, preds)


if not TEST_PATH.exists():
    sys.exit(f"ERROR: {TEST_PATH} not found — run training/prepare_data.py first")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
test = pd.read_parquet(TEST_PATH, columns=["text", "Corrected WUC"]).head(n)
texts = test["text"].astype(str).tolist()
truth = test["Corrected WUC"].astype(str).str.strip().str.upper()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved = AutoConfig.from_pretrained(MODEL_PATH).classifier_pooling
print(f"Checkpoint : {MODEL_PATH}")
print(f"Saved config classifier_pooling = {saved!r}")
print(f"Scoring {len(test):,} held-out test rows under each mode on {device}\n")

results = {p: score(p, texts, truth, device) for p in ("mean", "cls")}

print()
for pooling, acc in results.items():
    marker = "  <- what is currently served" if pooling == saved else ""
    print(f"  classifier_pooling={pooling:<5}  top-1 = {acc:.4f}{marker}")

delta = results["cls"] - results["mean"]
print()
if abs(delta) < 0.01:
    print("No meaningful difference — pooling is not the cause. Look elsewhere.")
elif delta > 0:
    print(f"CLS pooling is {delta:.4f} better, matching how train_hierarchical.py")
    print("pooled. The saved config is wrong; set classifier_pooling to 'cls'.")
else:
    print(f"mean pooling is {-delta:.4f} better, which contradicts the training")
    print("code. Do not change anything — investigate before acting.")
