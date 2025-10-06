import sys, json
import pandas as pd
from joblib import load

MODEL_PATH = "artifacts/model_rf_pipeline.joblib"
COLS = ["description", "category", "probability", "impact"]

if len(sys.argv) != 2:
    raise SystemExit("Usage: python -m src.predict <path_to_card.json>")

with open(sys.argv[1], "r", encoding="utf-8") as f:
    rec = json.load(f)

for k in COLS:
    if k not in rec:
        raise SystemExit(f"Missing key: {k}")

df = pd.DataFrame([{c: rec[c] for c in COLS}], columns=COLS)

clf = load(MODEL_PATH)
print(clf.predict(df)[0])  # Low | Medium | High
