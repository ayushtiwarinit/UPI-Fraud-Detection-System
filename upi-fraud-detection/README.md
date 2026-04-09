# UPI Fraud Detection System (Python + ML + Streamlit)

## Abstract
End-to-end project to detect fraudulent UPI transactions using classical ML. Includes synthetic data generation, EDA, supervised models (LogReg, RF, XGBoost), optional anomaly detection (Isolation Forest), a Streamlit web app for batch & single predictions, and packaging with `joblib` models.

## Features
- Synthetic UPI dataset with realistic fraud patterns
- Behavioral features (night-time, velocity, user amount z-score, new-merchant)
- Class imbalance handling (SMOTE / class weights)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Saves best model via `joblib`
- **Streamlit app** for CSV upload & single-transaction testing
- Optional: Isolation Forest anomaly scoring
- Alert table for suspected frauds

## Tech Stack
Python, pandas, numpy, scikit-learn, imbalanced-learn, xgboost, matplotlib, seaborn, streamlit, joblib

## Project Structure
```
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── upi_transactions.csv            # generated sample (5K rows)
├── models/                             # will contain saved models
├── notebooks/
│   └── eda_quickstart.py               # quick EDA plots -> notebooks/*.png
├── scripts/
│   ├── generate_synthetic_upi.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── train_isolation_forest.py
│   └── predict_single.py
├── requirements.txt
└── README.md
```

## Setup
```bash
# 1) Create venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt
```

## Data
A 5,000-row sample is provided at `data/upi_transactions.csv`.
Regenerate with:
```bash
python scripts/generate_synthetic_upi.py
```

## Training
```bash
python scripts/train_model.py --data_csv data/upi_transactions.csv --models_dir models
# metrics written to models/metrics.json
# best model saved as models/best_model__<name>.joblib
```

### Optional: Anomaly detector
```bash
python scripts/train_isolation_forest.py --data_csv data/upi_transactions.csv --models_dir models
```

## CLI Prediction (Single JSON)
Create `sample.json`:
```json
{
  "TransactionID": "manual-1",
  "UserID": 123,
  "Amount": 999.0,
  "Merchant": "UnknownQR",
  "Location": "Mumbai",
  "Time": "2025-09-25 23:45:00",
  "TransactionType": "QR",
  "DeviceType": "Android"
}
```
Run:
```bash
python scripts/predict_single.py --model_path models/best_model__xgboost.joblib --json_path sample.json
```

## Streamlit App
```bash
streamlit run app/streamlit_app.py
```
- Upload CSV with required columns to get per-row predictions.
- Use the form to test a single transaction and view probability.

## Screenshots (placeholders)
- notebooks/class_balance.png
- notebooks/amount_distribution.png
- notebooks/fraud_by_hour.png

## Future Improvements
- Device fingerprint & SIM change features
- Graph features (merchant-user networks)
- Autoencoder for deep anomaly detection
- Threshold tuning per segment to control FPR
- SHAP-based explanations in the app
```

