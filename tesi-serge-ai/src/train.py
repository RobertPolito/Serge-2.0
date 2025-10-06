import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("dataset/risks.json")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42

def load_dataset():
    data = json.loads(Path(DATA_PATH).read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    # Verifiche minime
    needed = {"description","category","probability","impact","severity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne nel dataset: {missing}")
    return df

def build_pipeline():
    text_col = "description"
    cat_col  = "category"
    num_cols = ["probability","impact"]

    preproc = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1,2), min_df=1), text_col),
            ("cat",  OneHotEncoder(handle_unknown="ignore"), [cat_col]),
            ("num",  "passthrough", num_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preproc", preproc),
        ("clf", clf),
    ])
    return pipe

def main():
    df = load_dataset()
    X = df[["description","category","probability","impact"]]
    y = df["severity"]

    pipe = build_pipeline()

    # CV stratificata 5-fold con seed fisso
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    f1_scores  = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)

    metrics = {
        "cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "f1_macro_mean": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "folds_accuracy": [float(v) for v in acc_scores],
        "folds_f1_macro": [float(v) for v in f1_scores],
        "n_samples": int(len(df)),
        "classes": sorted(list(map(str, set(y)))),
    }

    # Addestramento finale su tutto il dataset e salvataggio
    pipe.fit(X, y)
    joblib.dump(pipe, ARTIFACTS_DIR / "model_rf_pipeline.joblib")
    Path(ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[OK] Training completato.")
    print(f"Accuracy (media±std): {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
    print(f"F1-macro (media±std): {metrics['f1_macro_mean']:.3f} ± {metrics['f1_macro_std']:.3f}")
    print(f"Modello salvato in: {ARTIFACTS_DIR / 'model_rf_pipeline.joblib'}")
    print(f"Metriche salvate in: {ARTIFACTS_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()

