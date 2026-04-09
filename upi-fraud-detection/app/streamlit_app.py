import streamlit as st
import pandas as pd
from joblib import load
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from preprocess import split_X_y, add_behavioral_features

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

st.title("UPI Fraud Detection System 🛡️")
st.markdown("Batch predictions from CSV and single-transaction testing.")

# Model loader
default_model = None
models = sorted(Path("models").glob("best_model__*.joblib"))
if models:
    default_model = str(models[0])
model_path = st.text_input("Path to trained model (.joblib)", value=default_model or "models/best_model__xgboost.joblib")

@st.cache_resource(show_spinner=False)
def load_model(path):
    return load(path)

if model_path and Path(model_path).exists():
    model = load_model(model_path)
    st.success(f"Loaded model: {Path(model_path).name}")
else:
    st.warning("Train a model first (python scripts/train_model.py)")

tab1, tab2 = st.tabs(["📄 Batch CSV", "🧪 Single Transaction"])

with tab1:
    st.subheader("Upload CSV")
    st.caption("CSV must include: TransactionID, UserID, Amount, Merchant, Location, Time, TransactionType, DeviceType")
    file = st.file_uploader("Choose CSV", type=["csv"])
    if file and model_path and Path(model_path).exists():
        df = pd.read_csv(file)
        X, _, df_engineered = split_X_y(df)
        probs = model.predict_proba(X)[:,1]
        df_out = df.copy()
        df_out["FraudProbability"] = probs
        df_out["IsFraud_Pred"] = (df_out["FraudProbability"] >= 0.5).astype(int)
        st.dataframe(df_out.head(50))
        st.download_button("Download predictions CSV", df_out.to_csv(index=False), file_name="predictions.csv")
        # quick alerts
        alerts = df_out[df_out["IsFraud_Pred"]==1][["TransactionID","UserID","Amount","Merchant","Location","FraudProbability"]]
        if not alerts.empty:
            st.error(f"ALERT: {len(alerts)} suspected frauds flagged")
            st.dataframe(alerts.sort_values("FraudProbability", ascending=False).head(100))

with tab2:
    st.subheader("Single Transaction Test")
    with st.form("single_txn"):
        col1, col2, col3 = st.columns(3)
        with col1:
            userid = st.number_input("UserID", min_value=1, value=123)
            amount = st.number_input("Amount (INR)", min_value=1.0, value=999.0, step=1.0)
            merchant = st.text_input("Merchant", value="UnknownQR")
        with col2:
            location = st.selectbox("Location", ["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai","Kolkata","Pune","Ahmedabad","Jaipur","Surat","Lucknow","Indore"], index=0)
            txn_type = st.selectbox("TransactionType", ["P2P","P2M","QR","PullRequest","AutoPay","Refund"], index=2)
            device = st.selectbox("DeviceType", ["Android","iOS","Rooted","Emulator"], index=0)
        with col3:
            time_str = st.text_input("Time (YYYY-MM-DD HH:MM:SS)", value="2025-09-25 23:45:00")
            txn_id = st.text_input("TransactionID", value="manual-test-1")

        submitted = st.form_submit_button("Predict")
        if submitted and model_path and Path(model_path).exists():
            row = {
                "TransactionID": txn_id,
                "UserID": int(userid),
                "Amount": float(amount),
                "Merchant": merchant,
                "Location": location,
                "Time": time_str,
                "TransactionType": txn_type,
                "DeviceType": device
            }
            df = pd.DataFrame([row])
            X, _, _ = split_X_y(df)
            prob = load_model(model_path).predict_proba(X)[:,1][0]
            st.metric("Fraud Probability", f"{prob:.2%}")
            st.write("Input:", row)
