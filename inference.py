# inference.py
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch.nn.functional as F

MODEL_DIR = "saved_model/pt_model"
TOKENIZER_DIR = "saved_model/tokenizer"
MAX_LEN = 128
LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# Load model & tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

try:
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(
        f"Failed to load model/tokenizer from {MODEL_DIR} and {TOKENIZER_DIR}. "
        f"Ensure you've trained the model and saved it before running inference.\nError: {e}"
    )

def predict_sentiment(texts):
    """texts: list of strings
    returns: list of dicts {label, score, probs}
    """
    if isinstance(texts, str):
        texts = [texts]

    encodings = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

    results = []
    for i, txt in enumerate(texts):
        label = LABELS[int(preds[i])]
        score = float(probs[i][preds[i]])
        results.append({"text": txt, "label": label, "score": score, "probs": probs[i].tolist()})

    return results

if __name__ == "__main__":
    sample = [
        "Company reports strong revenue growth.",
        "Market jitters cause share fall due to regulation concerns."
    ]
    print(predict_sentiment(sample))
