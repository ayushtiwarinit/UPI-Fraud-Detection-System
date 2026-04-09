# scripts/generate_synthetic_upi.py
import os
import uuid
import math
import random
from collections import deque, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Surat": (21.1702, 72.8311),
    "Lucknow": (26.8467, 80.9462),
    "Indore": (22.7196, 75.8577),
}

MERCHANTS = [
    "Amazon", "Flipkart", "Zomato", "Swiggy", "IRCTC", "Uber", "Ola",
    "PaytmMall", "BigBazaar", "Airtel", "Jio", "ElectricityBoard", "SmallShop", "UnknownQR"
]

TXN_TYPES = ["P2P", "P2M", "QR", "PullRequest", "AutoPay", "Refund"]
DEVICE_TYPES = ["Android", "iOS", "Rooted", "Emulator"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    import math
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def generate_upi_dataset(
    n_rows=20000,
    n_users=5000,
    start_date="2025-06-01",
    end_date="2025-09-30",
    base_fraud_rate=0.018,
    save_path="data/upi_transactions.csv"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    span_seconds = int((end_dt - start_dt).total_seconds())

    users = np.arange(1, n_users + 1)
    user_home_city = np.random.choice(list(CITY_COORDS.keys()), size=n_users, replace=True)

    user_amount_mu = np.random.uniform(7.5, 9.0, size=n_users)
    user_amount_sigma = np.random.uniform(0.35, 0.70, size=n_users)

    device_weights = np.array([0.68, 0.29, 0.02, 0.01])

    last_txn_time = {}
    last_city = {}
    sliding_windows = defaultdict(lambda: deque())
    known_merchants = defaultdict(set)

    rows = []
    sampled_users = np.random.choice(users, size=n_rows, replace=True)
    span_seconds = max(1, span_seconds)
    sampled_seconds = np.random.randint(0, span_seconds, size=n_rows)
    start_dt = datetime.fromisoformat(start_date)
    sampled_times = [start_dt + timedelta(seconds=int(s)) for s in sampled_seconds]

    cities = list(CITY_COORDS.keys())
    order = np.argsort(sampled_seconds)

    for idx in order:
        uid = int(sampled_users[idx])
        t = sampled_times[idx]

        city = np.random.choice([user_home_city[uid - 1]] + cities, p=[0.7] + [0.3/len(cities)]*len(cities))
        device = np.random.choice(DEVICE_TYPES, p=[0.68,0.29,0.02,0.01])
        txn_type = np.random.choice(TXN_TYPES, p=[0.35,0.35,0.12,0.06,0.05,0.07])
        merchant = np.random.choice(MERCHANTS, p=[
            0.17, 0.12, 0.10, 0.09, 0.04, 0.07, 0.05, 0.04, 0.05, 0.06, 0.05, 0.05, 0.05, 0.06
        ])

        mu = user_amount_mu[uid - 1]
        sigma = user_amount_sigma[uid - 1]
        amount = float(np.random.lognormal(mean=mu, sigma=sigma))
        amount = max(1.0, min(amount, 200000.0))

        p = base_fraud_rate
        if t.hour >= 23 or t.hour <= 6: p += 0.020
        if txn_type == "PullRequest": p += 0.060
        elif txn_type == "QR": p += 0.020
        elif txn_type == "P2P": p += 0.010
        elif txn_type == "AutoPay": p -= 0.006
        elif txn_type == "Refund": p -= 0.010
        if device in ["Rooted","Emulator"]: p += 0.050
        if merchant not in known_merchants[uid]: p += 0.020

        since_last_min = None
        if uid in last_txn_time:
            diff = (t - last_txn_time[uid]).total_seconds()/60.0
            since_last_min = diff
            if diff < 2.0: p += 0.050

        dq = sliding_windows[uid]
        while dq and (t - dq[0]).total_seconds() > 3600:
            dq.popleft()
        last_hour_count = len(dq)
        if last_hour_count >= 5: p += 0.040
        dq.append(t)

        approx_user_mean = math.exp(mu + (sigma**2)/2.0)
        if amount > 3.0*approx_user_mean: p += 0.050
        elif amount > 1.8*approx_user_mean: p += 0.025

        if uid in last_city and since_last_min is not None and since_last_min <= 60:
            lat1, lon1 = CITY_COORDS[last_city[uid]]
            lat2, lon2 = CITY_COORDS[city]
            dist_km = haversine_km(lat1, lon1, lat2, lon2)
            if dist_km > 800: p += 0.080

        p = float(np.clip(p, 0.0, 0.95))
        is_fraud = int(np.random.rand() < p)

        last_txn_time[uid] = t
        last_city[uid] = city
        known_merchants[uid].add(merchant)

        rows.append({
            "TransactionID": str(uuid.uuid4()),
            "UserID": uid,
            "Amount": round(amount, 2),
            "Merchant": merchant,
            "Location": city,
            "Time": t.isoformat(sep=" "),
            "TransactionType": txn_type,
            "DeviceType": device,
            "IsFraud": is_fraud
        })

    df = pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)
    fraud_rate = df["IsFraud"].mean()
    print(f"Generated: {len(df):,} rows | Users: {n_users:,} | Fraud rate: {fraud_rate:.3%}")
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    return df

if __name__ == "__main__":
    generate_upi_dataset(
        n_rows=5000,
        n_users=1500,
        start_date="2025-06-01",
        end_date="2025-09-30",
        base_fraud_rate=0.018,
        save_path="data/upi_transactions.csv"
    )
