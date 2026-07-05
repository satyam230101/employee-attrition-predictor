"""
==============================================================
  AI-Powered Employee Attrition Prediction System
  Streamlit Web Application
==============================================================
Author: Your Name
Description: Full-featured Streamlit UI with:
  - Manual prediction form
  - Bulk CSV upload
  - Interactive analytics dashboard
  - Explainable AI (feature importance)
==============================================================
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(BASE_DIR, "model")
DATA_DIR     = os.path.join(BASE_DIR, "data")

MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
META_PATH    = os.path.join(MODEL_DIR, "model_metadata.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ─── Streamlit Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Dark Premium Theme) ─────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1117; }
    section[data-testid="stSidebar"] { background: #161b22 !important; }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1c2230 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
        transition: transform .2s;
    }
    .metric-card:hover { transform: translateY(-4px); border-color: #58a6ff; }
    .metric-title  { font-size: 13px; color: #8b949e; font-weight: 500; letter-spacing: .8px; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value  { font-size: 36px; font-weight: 700; color: #e6edf3; }
    .metric-sub    { font-size: 12px; color: #58a6ff; margin-top: 4px; }

    /* ── Prediction result cards ── */
    .result-card-leave {
        background: linear-gradient(135deg, #3d1a1a 0%, #2d1515 100%);
        border: 2px solid #f85149;
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .result-card-stay {
        background: linear-gradient(135deg, #1a3d2e 0%, #152d23 100%);
        border: 2px solid #3fb950;
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    }
    .result-emoji { font-size: 64px; display: block; margin-bottom: 16px; }
    .result-label { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
    .result-prob  { font-size: 18px; color: #8b949e; }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(248,81,73,.4); }
        50%       { box-shadow: 0 0 20px 8px rgba(248,81,73,.15); }
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 22px; font-weight: 700; color: #e6edf3;
        border-left: 4px solid #58a6ff;
        padding-left: 12px; margin-bottom: 20px;
    }

    /* ── Feature importance bar ── */
    .feat-bar-wrap { margin: 6px 0; }
    .feat-label    { font-size: 13px; color: #8b949e; margin-bottom: 3px; }
    .feat-bar-bg   { background: #21262d; border-radius: 6px; height: 12px; }
    .feat-bar-fill { height: 12px; border-radius: 6px;
                     background: linear-gradient(90deg, #58a6ff, #3fb950); }

    /* ── DataFrame style ── */
    .dataframe th { background: #161b22 !important; color: #58a6ff !important; }
    .dataframe td { color: #e6edf3 !important; }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, encoders, and metadata (cached)."""
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    meta     = joblib.load(META_PATH)
    return model, scaler, encoders, meta


@st.cache_data
def load_dataset():
    """Load the original IBM HR dataset for dashboard analytics."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None


def models_available():
    return all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, META_PATH])


def encode_input(raw_input: dict, encoders: dict, feature_names: list, scaler) -> np.ndarray:
    """
    Encode a single raw input dict into a scaled numpy array
    matching the model's expected feature order.
    """
    df = pd.DataFrame([raw_input])

    # Apply label encoders
    for col, le in encoders.items():
        if col in df.columns:
            val = df[col].astype(str).values[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = le.transform([le.classes_[0]])  # fallback

    # Reorder to training feature order
    df = df.reindex(columns=feature_names, fill_value=0)

    return scaler.transform(df)


def predict_attrition(X_scaled, model):
    """Return prediction label and probability."""
    pred      = model.predict(X_scaled)[0]
    prob      = model.predict_proba(X_scaled)[0]
    leave_prob = round(float(prob[1]) * 100, 1)
    stay_prob  = round(float(prob[0]) * 100, 1)
    return pred, leave_prob, stay_prob


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from tree-based models."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    df["Importance_pct"] = (df["Importance"] / df["Importance"].sum() * 100).round(2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏢 Attrition Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔮  Predict Employee", "📂  Bulk CSV Upload", "📊  Analytics Dashboard", "🧠  Model Info"],
        index=0
    )

    st.markdown("---")
    st.markdown("### Model Status")
    if models_available():
        meta = joblib.load(META_PATH)
        st.success(f"✅ {meta['model_name']}")
        m = meta["metrics"]
        st.markdown(f"""
        | Metric | Score |
        |---|---|
        | Accuracy  | {m['Accuracy']} |
        | F1-Score  | {m['F1-Score']} |
        | ROC-AUC   | {m['ROC-AUC']} |
        """)
    else:
        st.error("⚠️ No trained model found.\nRun `python app/train_model.py` first.")

    st.markdown("---")
    st.caption("AI-Powered HR Analytics • v1.0")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown('<h1 style="color:#e6edf3;font-size:40px;font-weight:800;">🏢 AI-Powered Employee Attrition Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:18px;">Predict, explain, and analyze employee attrition using machine learning.</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-title">Models Trained</div>
            <div class="metric-value">3</div>
            <div class="metric-sub">LR · RF · XGBoost</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-title">Dataset Rows</div>
            <div class="metric-value">1,470</div>
            <div class="metric-sub">IBM HR Analytics</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-title">Features Used</div>
            <div class="metric-value">30+</div>
            <div class="metric-sub">After preprocessing</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-title">Target</div>
            <div class="metric-value">Attrition</div>
            <div class="metric-sub">Binary classification</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Features")

    features = [
        ("🔮", "Predict Employee", "Enter employee details and get instant attrition prediction with confidence score."),
        ("📂", "Bulk CSV Upload", "Upload a CSV of employees and download predictions for all of them at once."),
        ("📊", "Analytics Dashboard", "Explore interactive charts: attrition by department, age, salary, overtime, and more."),
        ("🧠", "Explainable AI", "Understand which factors drive the model's predictions using feature importance."),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <div style="font-size:32px;margin-bottom:8px;">{icon}</div>
                <div style="font-size:17px;font-weight:600;color:#e6edf3;margin-bottom:6px;">{title}</div>
                <div style="font-size:14px;color:#8b949e;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT EMPLOYEE
# ══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown('<div class="section-header">🔮 Predict Employee Attrition</div>', unsafe_allow_html=True)

    if not models_available():
        st.error("⚠️ No trained model found. Please run `python app/train_model.py` first.")
        st.stop()

    model, scaler, encoders, meta = load_model_artifacts()
    feature_names = meta["feature_names"]

    st.markdown("Fill in the employee details below and click **Predict** to get the result.")

    with st.form("prediction_form"):
        st.markdown("#### 👤 Personal Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            Age              = st.slider("Age", 18, 65, 35)
            Gender           = st.selectbox("Gender", ["Male", "Female"])
            MaritalStatus    = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        with c2:
            Education        = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                            format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
            EducationField   = st.selectbox("Education Field", ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"])
            NumCompaniesWorked = st.slider("Companies Worked At", 0, 9, 2)
        with c3:
            DistanceFromHome = st.slider("Distance From Home (km)", 1, 29, 5)
            EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1,2,3,4],
                format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1,2,3,4],
                format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])

        st.markdown("#### 💼 Job Details")
        c4, c5, c6 = st.columns(3)
        with c4:
            Department       = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            JobRole          = st.selectbox("Job Role", [
                "Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director",
                "Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"
            ])
            JobLevel         = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        with c5:
            JobSatisfaction  = st.selectbox("Job Satisfaction", [1,2,3,4],
                format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            BusinessTravel   = st.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
            OverTime         = st.selectbox("Overtime", ["Yes", "No"])
        with c6:
            YearsAtCompany      = st.slider("Years at Company", 0, 40, 5)
            YearsInCurrentRole  = st.slider("Years in Current Role", 0, 18, 3)
            YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
            YearsWithCurrManager    = st.slider("Years With Current Manager", 0, 17, 3)

        st.markdown("#### 💰 Compensation & Work-Life")
        c7, c8, c9 = st.columns(3)
        with c7:
            MonthlyIncome    = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=500)
            MonthlyRate      = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, value=14000, step=500)
            DailyRate        = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800, step=50)
        with c8:
            HourlyRate       = st.slider("Hourly Rate ($)", 30, 100, 65)
            PercentSalaryHike= st.slider("Salary Hike (%)", 11, 25, 15)
            StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        with c9:
            WorkLifeBalance  = st.selectbox("Work-Life Balance", [1,2,3,4],
                format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x])
            TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 3)
            TotalWorkingYears     = st.slider("Total Working Years", 0, 40, 10)
            PerformanceRating     = st.selectbox("Performance Rating", [1,2,3,4])

        submitted = st.form_submit_button("🔮  Predict Attrition", use_container_width=True)

    if submitted:
        raw_input = {
            "Age": Age, "BusinessTravel": BusinessTravel, "DailyRate": DailyRate,
            "Department": Department, "DistanceFromHome": DistanceFromHome, "Education": Education,
            "EducationField": EducationField, "EnvironmentSatisfaction": EnvironmentSatisfaction,
            "Gender": Gender, "HourlyRate": HourlyRate, "JobInvolvement": 3,
            "JobLevel": JobLevel, "JobRole": JobRole, "JobSatisfaction": JobSatisfaction,
            "MaritalStatus": MaritalStatus, "MonthlyIncome": MonthlyIncome, "MonthlyRate": MonthlyRate,
            "NumCompaniesWorked": NumCompaniesWorked, "OverTime": OverTime,
            "PercentSalaryHike": PercentSalaryHike, "PerformanceRating": PerformanceRating,
            "RelationshipSatisfaction": RelationshipSatisfaction, "StockOptionLevel": StockOptionLevel,
            "TotalWorkingYears": TotalWorkingYears, "TrainingTimesLastYear": TrainingTimesLastYear,
            "WorkLifeBalance": WorkLifeBalance, "YearsAtCompany": YearsAtCompany,
            "YearsInCurrentRole": YearsInCurrentRole,
            "YearsSinceLastPromotion": YearsSinceLastPromotion,
            "YearsWithCurrManager": YearsWithCurrManager,
        }

        X_scaled = encode_input(raw_input, encoders, feature_names, scaler)
        pred, leave_prob, stay_prob = predict_attrition(X_scaled, model)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        col_res, col_feat = st.columns([1, 1])

        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="result-card-leave">
                    <span class="result-emoji">⚠️</span>
                    <div class="result-label" style="color:#f85149;">HIGH ATTRITION RISK</div>
                    <div class="result-prob">Probability of Leaving: <b style="color:#f85149;">{leave_prob}%</b></div>
                    <div class="result-prob">Probability of Staying: <b style="color:#3fb950;">{stay_prob}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-stay">
                    <span class="result-emoji">✅</span>
                    <div class="result-label" style="color:#3fb950;">LOW ATTRITION RISK</div>
                    <div class="result-prob">Probability of Staying: <b style="color:#3fb950;">{stay_prob}%</b></div>
                    <div class="result-prob">Probability of Leaving: <b style="color:#f85149;">{leave_prob}%</b></div>
                </div>
                """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=leave_prob,
                title={"text": "Attrition Risk %", "font": {"size": 16, "color": "#e6edf3"}},
                number={"suffix": "%", "font": {"size": 32, "color": "#e6edf3"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar":  {"color": "#f85149" if leave_prob > 50 else "#3fb950"},
                    "bgcolor": "#21262d",
                    "steps": [
                        {"range": [0, 30],   "color": "#0d2318"},
                        {"range": [30, 60],  "color": "#2d2a00"},
                        {"range": [60, 100], "color": "#3d0f0f"},
                    ],
                    "threshold": {"line": {"color": "#f0e68c", "width": 3}, "thickness": 0.8, "value": leave_prob},
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#e6edf3", height=250, margin=dict(l=20, r=20, t=40, b=0)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_feat:
            st.markdown("#### 🧠 Top Factors Driving Prediction")
            feat_df = get_feature_importance(model, feature_names)
            if not feat_df.empty:
                top10 = feat_df.head(10)
                for _, row in top10.iterrows():
                    pct = row["Importance_pct"]
                    width = round(pct / top10["Importance_pct"].max() * 100, 1)
                    st.markdown(f"""
                    <div class="feat-bar-wrap">
                        <div class="feat-label">{row['Feature']}  <span style="color:#58a6ff;">{pct}%</span></div>
                        <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{width}%;"></div></div>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BULK CSV UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif "Bulk" in page or "CSV" in page:
    st.markdown('<div class="section-header">📂 Bulk CSV Prediction</div>', unsafe_allow_html=True)

    if not models_available():
        st.error("⚠️ No trained model found. Please run `python app/train_model.py` first.")
        st.stop()

    model, scaler, encoders, meta = load_model_artifacts()
    feature_names = meta["feature_names"]
    cat_cols = meta.get("categorical_cols", [])

    st.info("📎 Upload a CSV file with employee data. The file should have the same columns as the training dataset.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded:** {df_upload.shape[0]} rows × {df_upload.shape[1]} columns")
        st.dataframe(df_upload.head(5), use_container_width=True)

        if st.button("🚀 Run Predictions", use_container_width=True):
            df_proc = df_upload.copy()

            # Drop target if present
            df_proc.drop(columns=["Attrition"], errors="ignore", inplace=True)

            # Fill missing values
            for col in df_proc.columns:
                if df_proc[col].dtype == "object":
                    df_proc[col].fillna(df_proc[col].mode()[0] if not df_proc[col].mode().empty else "Unknown", inplace=True)
                else:
                    df_proc[col].fillna(df_proc[col].median(), inplace=True)

            # Label-encode categoricals
            for col, le in encoders.items():
                if col in df_proc.columns:
                    df_proc[col] = df_proc[col].astype(str).apply(
                        lambda v: le.transform([v])[0] if v in le.classes_ else le.transform([le.classes_[0]])[0]
                    )

            # Align columns
            df_feat = df_proc.reindex(columns=feature_names, fill_value=0)

            # Scale
            X_scaled = scaler.transform(df_feat)

            # Predict
            preds      = model.predict(X_scaled)
            probs      = model.predict_proba(X_scaled)[:, 1]

            df_result = df_upload.copy()
            df_result["Predicted_Attrition"] = ["Yes" if p == 1 else "No" for p in preds]
            df_result["Leave_Probability_%"] = (probs * 100).round(1)
            df_result["Risk_Level"] = pd.cut(
                probs * 100,
                bins=[0, 30, 60, 100],
                labels=["🟢 Low", "🟡 Medium", "🔴 High"]
            )

            st.success(f"✅ Predictions complete! {(preds==1).sum()} employees predicted to leave.")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Employees", df_result.shape[0])
            with col_b:
                st.metric("Predicted to Leave", int((preds==1).sum()), delta=f"{(preds==1).mean()*100:.1f}%")
            with col_c:
                st.metric("Avg Leave Probability", f"{(probs*100).mean():.1f}%")

            # Risk distribution pie
            risk_counts = df_result["Risk_Level"].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_sequence=["#3fb950", "#d29922", "#f85149"],
                hole=0.5
            )
            fig_pie.update_layout(paper_bgcolor="#0d1117", font_color="#e6edf3", title_font_size=16)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.dataframe(df_result[["Predicted_Attrition", "Leave_Probability_%", "Risk_Level"]
                                   + [c for c in df_result.columns if c not in ["Predicted_Attrition","Leave_Probability_%","Risk_Level"]]
                                   ].head(30), use_container_width=True)

            # Download
            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download Predictions CSV",
                data=csv_bytes,
                file_name="attrition_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page or "Dashboard" in page:
    st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Please place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the `data/` folder.")
        st.stop()

    # Top KPI row
    total        = len(df)
    left         = (df["Attrition"] == "Yes").sum()
    stayed       = total - left
    attrition_rt = round(left / total * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Employees</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="metric-card"><div class="metric-title">Left Company</div><div class="metric-value" style="color:#f85149;">{left}</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="metric-card"><div class="metric-title">Retained</div><div class="metric-value" style="color:#3fb950;">{stayed}</div></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="metric-card"><div class="metric-title">Attrition Rate</div><div class="metric-value">{attrition_rt}%</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    COLORS = {"Yes": "#f85149", "No": "#3fb950"}
    LAYOUT = dict(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#e6edf3",
                  title_font_size=15, legend_bgcolor="#161b22", legend_borderwidth=0)

    # Row 1: Attrition by Department | by Job Role
    c1, c2 = st.columns(2)
    with c1:
        dept_df = df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        fig1 = px.bar(dept_df, x="Department", y="Count", color="Attrition",
                      barmode="group", title="Attrition by Department",
                      color_discrete_map=COLORS)
        fig1.update_layout(**LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        role_left = df[df["Attrition"]=="Yes"]["JobRole"].value_counts().reset_index()
        role_left.columns = ["JobRole", "Count"]
        fig2 = px.bar(role_left, x="Count", y="JobRole", orientation="h",
                      title="Attrition Count by Job Role",
                      color="Count", color_continuous_scale="Reds")
        fig2.update_layout(**LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Age distribution | Salary vs Attrition
    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(df, x="Age", color="Attrition",
                            title="Attrition by Age Group", nbins=20,
                            color_discrete_map=COLORS, barmode="overlay", opacity=0.75)
        fig3.update_layout(**LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.box(df, x="Attrition", y="MonthlyIncome",
                      title="Monthly Income vs Attrition",
                      color="Attrition", color_discrete_map=COLORS)
        fig4.update_layout(**LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Overtime | Work-Life Balance
    c5, c6 = st.columns(2)
    with c5:
        ot_df = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
        fig5 = px.bar(ot_df, x="OverTime", y="Count", color="Attrition",
                      title="Overtime vs Attrition",
                      color_discrete_map=COLORS, barmode="group")
        fig5.update_layout(**LAYOUT)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        wlb_df = df.groupby(["WorkLifeBalance", "Attrition"]).size().reset_index(name="Count")
        fig6 = px.bar(wlb_df, x="WorkLifeBalance", y="Count", color="Attrition",
                      title="Work-Life Balance vs Attrition",
                      color_discrete_map=COLORS, barmode="group",
                      labels={"WorkLifeBalance": "WLB Score (1=Bad, 4=Best)"})
        fig6.update_layout(**LAYOUT)
        st.plotly_chart(fig6, use_container_width=True)

    # Row 4: Satisfaction heatmap | Distance from home
    c7, c8 = st.columns(2)
    with c7:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        corr = df[num_cols].corr()
        import plotly.figure_factory as ff
        fig7 = ff.create_annotated_heatmap(
            z=corr.values.round(2),
            x=num_cols, y=num_cols,
            colorscale="Blues", showscale=True, annotation_text=None
        )
        fig7.update_layout(title="Correlation Heatmap (Numeric Features)", **LAYOUT, height=450)
        st.plotly_chart(fig7, use_container_width=True)

    with c8:
        fig8 = px.violin(df, x="Attrition", y="DistanceFromHome",
                         title="Distance From Home vs Attrition",
                         color="Attrition", color_discrete_map=COLORS, box=True, points="outliers")
        fig8.update_layout(**LAYOUT)
        st.plotly_chart(fig8, use_container_width=True)

    # Marital status
    ms_df = df.groupby(["MaritalStatus","Attrition"]).size().reset_index(name="Count")
    fig9 = px.sunburst(ms_df, path=["MaritalStatus","Attrition"], values="Count",
                       title="Attrition by Marital Status",
                       color_discrete_map={"Yes":"#f85149","No":"#3fb950"})
    fig9.update_layout(**LAYOUT)
    st.plotly_chart(fig9, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown('<div class="section-header">🧠 Model Information</div>', unsafe_allow_html=True)

    if not models_available():
        st.error("⚠️ No trained model found. Please run `python app/train_model.py` first.")
        st.stop()

    model, scaler, encoders, meta = load_model_artifacts()
    feature_names = meta["feature_names"]
    all_results   = meta.get("all_results", {})

    st.markdown(f"### Best Model: **{meta['model_name']}**")

    # Metrics comparison table
    if all_results:
        compare_df = pd.DataFrame(all_results).T.reset_index()
        compare_df.columns = ["Model"] + list(compare_df.columns[1:])
        st.markdown("#### 📊 Model Comparison")
        st.dataframe(compare_df.style.highlight_max(axis=0, color="#1a3d2e"), use_container_width=True)

        # Radar chart
        categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        fig_radar = go.Figure()
        palette = ["#58a6ff", "#3fb950", "#f0e68c"]
        for i, (mname, mvals) in enumerate(all_results.items()):
            vals = [mvals.get(c, 0) for c in categories]
            vals += [vals[0]]  # close radar
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                fill="toself", name=mname,
                line_color=palette[i % len(palette)]
            ))
        fig_radar.update_layout(
            polar=dict(bgcolor="#161b22",
                       radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d"),
                       angularaxis=dict(gridcolor="#30363d")),
            paper_bgcolor="#0d1117", font_color="#e6edf3",
            title="Model Comparison (Radar Chart)", title_font_size=16,
            legend_bgcolor="#161b22"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Feature importance
    feat_df = get_feature_importance(model, feature_names)
    if not feat_df.empty:
        st.markdown("#### 🏆 Top 15 Most Important Features")
        fig_feat = px.bar(feat_df.head(15), x="Importance_pct", y="Feature",
                          orientation="h", title="Feature Importances (%)",
                          color="Importance_pct", color_continuous_scale="Blues")
        fig_feat.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_feat, use_container_width=True)
