import json
from pathlib import Path
import pandas as pd

DATA_PATH = Path("dataset/risks.json")

def main():
    data = json.loads(Path(DATA_PATH).read_text(encoding="utf-8"))
    df = pd.DataFrame(data)

    # Conteggio per categoria
    cat_counts = df["category"].value_counts().to_dict()
    # Conteggio per severity
    sev_counts = df["severity"].value_counts().to_dict()
    # Tabella combinata categoria x severity
    table = pd.crosstab(df["category"], df["severity"])

    print("Distribuzione per categoria:")
    for k,v in cat_counts.items():
        print(f"  {k}: {v}")

    print("\nDistribuzione per gravità:")
    for k,v in sev_counts.items():
        print(f"  {k}: {v}")

    print("\nTabella categoria x gravità:")
    print(table)

if __name__ == "__main__":
    main()
