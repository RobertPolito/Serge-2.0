import json
import sys
from jsonschema import validate, ValidationError

def main():
    try:
        with open("schemas/card.schema.json", "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open("dataset/risks.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(f"[ERR] File non trovato: {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERR] JSON non valido: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("[ERR] Il dataset deve essere una lista di record.")
        sys.exit(1)

    for i, rec in enumerate(data, 1):
        try:
            validate(instance=rec, schema=schema)
        except ValidationError as e:
            print(f"[ERR] Record {i} (id={rec.get('id')}): {e.message}")
            sys.exit(1)

    print(f"[OK] Dataset valido: {len(data)} record.")

if __name__ == "__main__":
    main()
