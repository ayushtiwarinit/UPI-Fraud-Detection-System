# scripts/predict_single.py
import argparse, json, pandas as pd
from joblib import load
from preprocess import split_X_y

def predict_row(model_path, json_path):
    model = load(model_path)
    row = json.loads(open(json_path).read())
    df = pd.DataFrame([row])
    X, _, _ = split_X_y(df)
    prob = model.predict_proba(X)[:,1][0]
    return float(prob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--json_path", required=True)
    args = parser.parse_args()
    p = predict_row(args.model_path, args.json_path)
    print(f"Fraud probability: {p:.4f}")
