"""
==============================================================
  AI-Powered Employee Attrition Prediction System
  Bonus: FastAPI REST API
==============================================================
Description: Provides a REST endpoint to get predictions
             programmatically, enabling integration with
             other systems (e.g., HRMS, dashboards).

Run with:
    uvicorn app.api:app --reload --port 8000

Documentation:
    http://localhost:8000/docs
==============================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(BASE_DIR, "model")

MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
META_PATH    = os.path.join(MODEL_DIR, "model_metadata.pkl")

# ─── Load artifacts ───────────────────────────────────────────────────────────
try:
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    meta     = joblib.load(META_PATH)
    FEATURE_NAMES = meta["feature_names"]
    MODEL_READY   = True
except Exception:
    MODEL_READY   = False
    FEATURE_NAMES = []

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="REST API for the AI-Powered Employee Attrition Prediction System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas ───────────────────────────────────────────────
class EmployeeInput(BaseModel):
    Age: int = Field(..., ge=18, le=65, example=35)
    BusinessTravel: str = Field(..., example="Travel_Rarely")
    DailyRate: int = Field(..., example=800)
    Department: str = Field(..., example="Research & Development")
    DistanceFromHome: int = Field(..., ge=1, le=29, example=5)
    Education: int = Field(..., ge=1, le=5, example=3)
    EducationField: str = Field(..., example="Life Sciences")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, example=3)
    Gender: str = Field(..., example="Male")
    HourlyRate: int = Field(..., example=65)
    JobInvolvement: int = Field(3, ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5, example=2)
    JobRole: str = Field(..., example="Research Scientist")
    JobSatisfaction: int = Field(..., ge=1, le=4, example=3)
    MaritalStatus: str = Field(..., example="Single")
    MonthlyIncome: int = Field(..., example=5000)
    MonthlyRate: int = Field(..., example=14000)
    NumCompaniesWorked: int = Field(..., ge=0, le=9, example=2)
    OverTime: str = Field(..., example="No")
    PercentSalaryHike: int = Field(..., ge=11, le=25, example=15)
    PerformanceRating: int = Field(..., ge=1, le=4, example=3)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, example=3)
    StockOptionLevel: int = Field(..., ge=0, le=3, example=1)
    TotalWorkingYears: int = Field(..., example=10)
    TrainingTimesLastYear: int = Field(..., ge=0, le=6, example=3)
    WorkLifeBalance: int = Field(..., ge=1, le=4, example=3)
    YearsAtCompany: int = Field(..., example=5)
    YearsInCurrentRole: int = Field(..., example=3)
    YearsSinceLastPromotion: int = Field(..., example=1)
    YearsWithCurrManager: int = Field(..., example=3)


class PredictionResponse(BaseModel):
    prediction: str
    will_leave: bool
    leave_probability_pct: float
    stay_probability_pct: float
    risk_level: str
    model_used: str


# ─── Helpers ──────────────────────────────────────────────────────────────────
def encode_and_predict(raw: dict):
    df = pd.DataFrame([raw])
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df[col].values[0])
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = le.transform([le.classes_[0]])

    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)
    X_scaled = scaler.transform(df)
    pred     = model.predict(X_scaled)[0]
    prob     = model.predict_proba(X_scaled)[0]
    return int(pred), float(prob[1])


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_ready": MODEL_READY, "message": "Employee Attrition API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"model_ready": MODEL_READY, "model_name": meta.get("model_name", "unknown") if MODEL_READY else None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(employee: EmployeeInput):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    raw   = employee.dict()
    pred, leave_prob = encode_and_predict(raw)

    leave_pct = round(leave_prob * 100, 1)
    stay_pct  = round((1 - leave_prob) * 100, 1)

    if leave_pct < 30:
        risk = "Low"
    elif leave_pct < 60:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionResponse(
        prediction="Leave" if pred == 1 else "Stay",
        will_leave=bool(pred == 1),
        leave_probability_pct=leave_pct,
        stay_probability_pct=stay_pct,
        risk_level=risk,
        model_used=meta.get("model_name", "unknown"),
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_name":    meta.get("model_name"),
        "metrics":       meta.get("metrics"),
        "feature_count": len(FEATURE_NAMES),
        "features":      FEATURE_NAMES,
    }
