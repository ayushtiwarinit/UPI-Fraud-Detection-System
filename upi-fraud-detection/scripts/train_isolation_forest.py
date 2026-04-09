# scripts/train_isolation_forest.py
import argparse, os, json, warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from preprocess import split_X_y, CATEGORICAL_COLS, NUMERIC_COLS

warnings.filterwarnings("ignore")

def build_preprocessor():
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    ct = ColumnTransformer([
        ("cat", ohe, CATEGORICAL_COLS),
        ("num", scaler, NUMERIC_COLS + ["hour","dayofweek","is_weekend","is_night",
                                        "amount_z_user","since_last_min","txns_last_hour","UserID"]),
    ])
    return ct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="data/upi_transactions.csv")
    parser.add_argument("--models_dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    df = pd.read_csv(args.data_csv)
    X, y, _ = split_X_y(df)

    pre = build_preprocessor()
    pipe = Pipeline([
        ("pre", pre),
        ("iforest", IsolationForest(n_estimators=300, contamination=0.02, random_state=42))
    ])
    pipe.fit(X)  # unsupervised
    path = os.path.join(args.models_dir, "anomaly_iforest.joblib")
    dump(pipe, path)
    print(f"Saved IsolationForest to {path}")
