# scripts/preprocess.py
import pandas as pd
import numpy as np

CATEGORICAL_COLS = ["Merchant", "Location", "TransactionType", "DeviceType"]
NUMERIC_COLS = ["Amount"]
ID_COLS = ["TransactionID", "UserID"]
TARGET_COL = "IsFraud"
TIME_COL = "Time"

def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["hour"] = df[TIME_COL].dt.hour
    df["dayofweek"] = df[TIME_COL].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 6)).astype(int)

    df["user_amount_mean"] = df.groupby("UserID")["Amount"].transform("mean")
    df["user_amount_std"] = df.groupby("UserID")["Amount"].transform("std").fillna(0.0)
    df["amount_z_user"] = (df["Amount"] - df["user_amount_mean"]) / (df["user_amount_std"].replace(0, 1.0))

    df = df.sort_values(["UserID", TIME_COL])
    df["prev_time"] = df.groupby("UserID")[TIME_COL].shift(1)
    df["since_last_min"] = (df[TIME_COL] - df["prev_time"]).dt.total_seconds() / 60.0
    df["since_last_min"] = df["since_last_min"].fillna(1e6)

    def _last_hour_counts(g):
        times = g[TIME_COL].values
        out = np.zeros(len(g), dtype=int)
        j = 0
        for i in range(len(g)):
            while j < i and (times[i] - times[j]).astype("timedelta64[s]").astype(int) > 3600:
                j += 1
            out[i] = i - j
        return pd.Series(out, index=g.index, dtype=int)

    df["txns_last_hour"] = df.groupby("UserID", group_keys=False).apply(_last_hour_counts)

    df["seen_merchant_before"] = df.groupby(["UserID", "Merchant"]).cumcount()
    df["is_new_merchant"] = (df["seen_merchant_before"] == 0).astype(int)
    df.drop(columns=["prev_time", "seen_merchant_before"], inplace=True)
    return df

def feature_columns():
    return (
        NUMERIC_COLS +
        CATEGORICAL_COLS +
        ["hour", "dayofweek", "is_weekend", "is_night",
         "amount_z_user", "since_last_min", "txns_last_hour", "UserID"]
    )

def split_X_y(df: pd.DataFrame):
    df = add_behavioral_features(df)
    feats = feature_columns()
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    X = df[feats].copy()
    return X, y, df

if __name__ == "__main__":
    raw = pd.read_csv("data/upi_transactions.csv")
    X, y, df_all = split_X_y(raw)
    print("X shape:", X.shape, "| y:", None if y is None else y.shape)
    print("Columns:", list(X.columns))
