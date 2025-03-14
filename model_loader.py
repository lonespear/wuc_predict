from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

# Load your model and tokenizer (modify based on your model)
MODEL_PATH = "jonday/wuc-model"  # Update this path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

with open("wuc_mapping.json", "r") as file:
    wuc_mapping = json.load(file)
with open("codes.json", "r") as file:
    wuc_defs = json.load(file)
with open("main_system.json", "r") as file:
    main_system = json.load(file)
index_to_wuc = {v: k for k, v in wuc_mapping.items()}

# Ensure model is in evaluation mode
model.eval()

def predict_discrepancy(text, method = 1):
    """Predict the work user code based on input discrepancy text"""

    if not isinstance(text, str) or not text.strip():
        return "Invalid input"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    confidence = probabilities[0, predicted_class].item() * 100

    # Convert model output (index) back to WUC
    wuc = index_to_wuc.get(predicted_class, "Unknown WUC")
    definition = wuc_defs.get(wuc, "Unknown Definition")
    system = main_system.get(wuc[:2], "Unknown Main System")
    if method == 1:
        return f"{wuc}: {system}, {definition} (Confidence: {confidence:.2f}%)"
    else:
        return wuc, definition, system, confidence
