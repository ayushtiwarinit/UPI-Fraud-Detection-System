# scripts/train_model.py
import argparse, os, warnings, json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from preprocess import split_X_y, CATEGORICAL_COLS, NUMERIC_COLS

warnings.filterwarnings("ignore")

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def build_preprocessor():
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    ct = ColumnTransformer([
        ("cat", ohe, CATEGORICAL_COLS),
        ("num", scaler, NUMERIC_COLS + ["hour","dayofweek","is_weekend","is_night",
                                        "amount_z_user","since_last_min","txns_last_hour","UserID"]),
    ])
    return ct

def main(args):
    os.makedirs(args.models_dir, exist_ok=True)
    df = pd.read_csv(args.data_csv)
    X, y, _ = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = build_preprocessor()

    models = {}
    reports = {}

    # 1) Logistic Regression with class_weight
    logreg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None))
    ])
    logreg.fit(X_train, y_train)
    prob = logreg.predict_proba(X_test)[:,1]
    reports["logreg"] = evaluate(y_test, prob)
    models["logreg"] = logreg

    # 2) Random Forest with SMOTE
    rf = ImbPipeline(steps=[
        ("pre", pre),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, class_weight=None
        ))
    ])
    rf.fit(X_train, y_train)
    prob = rf.predict_proba(X_test)[:,1]
    reports["random_forest"] = evaluate(y_test, prob)
    models["random_forest"] = rf

    # 3) XGBoost with scale_pos_weight
    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
    xgb = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            scale_pos_weight=float(pos_weight)
        ))
    ])
    xgb.fit(X_train, y_train)
    prob = xgb.predict_proba(X_test)[:,1]
    reports["xgboost"] = evaluate(y_test, prob)
    models["xgboost"] = xgb

    print("Evaluation (on holdout):")
    for k,v in reports.items():
        print(k, "=>", {m: round(s,4) for m,s in v.items()})

    # choose best by roc_auc
    best_name = max(reports, key=lambda k: reports[k]["roc_auc"])
    best_model = models[best_name]
    model_path = os.path.join(args.models_dir, f"best_model__{best_name}.joblib")
    dump(best_model, model_path)
    print(f"Saved best model: {best_name} -> {model_path}")

    with open(os.path.join(args.models_dir, "metrics.json"), "w") as f:
        json.dump(reports, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="data/upi_transactions.csv")
    parser.add_argument("--models_dir", default="models")
    args = parser.parse_args()
    main(args)
