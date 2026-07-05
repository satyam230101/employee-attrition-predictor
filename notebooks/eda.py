"""
==============================================================
  AI-Powered Employee Attrition Prediction System
  EDA + Model Evaluation Jupyter Notebook (Python script)
==============================================================
Run this as a standalone script or copy it to a .ipynb.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ─── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# ─── Target distribution ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["Attrition"].value_counts().plot(kind="bar", ax=axes[0], color=["#3fb950", "#f85149"])
axes[0].set_title("Attrition Distribution")

df.groupby("Attrition")["Age"].hist(bins=20, alpha=0.7, ax=axes[1],
                                      color=["#3fb950", "#f85149"], label=["Stay","Leave"])
axes[1].set_title("Age Distribution by Attrition")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "screenshots", "eda_overview.png"), dpi=150)
plt.close()

print("[EDA] Charts saved to screenshots/")
