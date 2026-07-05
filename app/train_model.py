"""
==============================================================
  AI-Powered Employee Attrition Prediction System
  Model Training Script
==============================================================
Author: Your Name
Description: Trains Logistic Regression, Random Forest, and
             XGBoost models on the IBM HR Attrition Dataset
             and saves the best model with its preprocessor.
==============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH    = os.path.join(MODEL_DIR, "model_metadata.pkl")

# ─── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Columns to drop (no predictive value or data-leakage risk)
COLS_TO_DROP = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# Categorical columns that will be label-encoded
CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at:\n  {path}\n"
            "Please download the IBM HR Attrition dataset from Kaggle and place it in the 'data/' folder."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    """
    Clean and encode the dataset.
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names
    """
    df = df.copy()

    # Drop useless columns
    df.drop(columns=COLS_TO_DROP, inplace=True, errors="ignore")

    # Encode target variable
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Handle missing values (fill with mode for categoricals, median for numerics)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # Label-encode categorical columns
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Separate features and target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    feature_names = X.columns.tolist()

    # Train/test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_names)

    print(f"[INFO] Train size: {X_train_scaled.shape[0]}  |  Test size: {X_test_scaled.shape[0]}")
    print(f"[INFO] Class distribution (train) — 0: {(y_train==0).sum()}  |  1: {(y_train==1).sum()}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_models():
    """Return a dictionary of {name: model_instance} to compare."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=5,           # handle class imbalance
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train a model and return its evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred),         4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0),        4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_pred_prob),     4),
    }

    # Cross-validation ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    metrics["CV-AUC (mean)"] = round(cv_scores.mean(), 4)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def train_all_models(X_train, X_test, y_train, y_test):
    """Train all models, compare them, return best model and its name."""
    models      = get_models()
    results     = {}
    trained     = {}

    for name, model in models.items():
        metrics         = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results[name]   = metrics
        trained[name]   = model

    # Compare using ROC-AUC as primary metric
    best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
    print(f"\n[RESULT] Best Model → {best_name}  (ROC-AUC: {results[best_name]['ROC-AUC']})")

    return trained[best_name], best_name, results


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artifacts(model, scaler, label_encoders, feature_names, best_name, results):
    """Persist model, scaler, encoders, and metadata to disk."""
    joblib.dump(model,          MODEL_PATH)
    joblib.dump(scaler,         SCALER_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)

    metadata = {
        "model_name":    best_name,
        "feature_names": feature_names,
        "metrics":       results[best_name],
        "all_results":   results,
        "categorical_cols": CATEGORICAL_COLS,
    }
    joblib.dump(metadata, META_PATH)

    print(f"\n[SAVED] Model      → {MODEL_PATH}")
    print(f"[SAVED] Scaler     → {SCALER_PATH}")
    print(f"[SAVED] Encoders   → {ENCODER_PATH}")
    print(f"[SAVED] Metadata   → {META_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI-Powered Employee Attrition Prediction System")
    print("  Model Training Pipeline")
    print("="*60)

    # Step 1 — Load data
    df = load_data(DATASET_PATH)

    # Step 2 — Preprocess
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess(df)

    # Step 3 — Train & evaluate all models
    best_model, best_name, results = train_all_models(X_train, X_test, y_train, y_test)

    # Step 4 — Save everything
    save_artifacts(best_model, scaler, label_encoders, feature_names, best_name, results)

    print("\n[DONE] Training complete. Run the Streamlit app with:")
    print("       streamlit run app/streamlit_app.py\n")
