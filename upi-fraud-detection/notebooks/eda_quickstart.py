# notebooks/eda_quickstart.py
# Quick EDA (run: python notebooks/eda_quickstart.py)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scripts.preprocess import add_behavioral_features

df = pd.read_csv("data/upi_transactions.csv")
df = add_behavioral_features(df)

print(df.head())
print("\nClass balance:\n", df["IsFraud"].value_counts(normalize=True))

plt.figure()
sns.countplot(x="IsFraud", data=df)
plt.title("Class Balance")
plt.savefig("notebooks/class_balance.png", bbox_inches="tight")

plt.figure()
sns.histplot(df["Amount"], bins=60, kde=True)
plt.title("Amount Distribution")
plt.savefig("notebooks/amount_distribution.png", bbox_inches="tight")

plt.figure()
sns.barplot(x="hour", y="IsFraud", data=df, estimator=lambda x: sum(x)/len(x))
plt.title("Fraud Rate by Hour")
plt.savefig("notebooks/fraud_by_hour.png", bbox_inches="tight")

print("Saved EDA plots to notebooks/*.png")
