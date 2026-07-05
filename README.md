# 🏢 AI-Powered Employee Attrition Prediction System

> A production-style machine learning web application that predicts whether an employee is likely to leave a company — complete with an interactive analytics dashboard, bulk CSV predictions, and Explainable AI.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## 🚀 Live Demo

> **[🔗 [Click here to view the live app on Streamlit Cloud](https://employee-attrition-predictor-beetvurojwcdjgpbj85gus.streamlit.app/)](#)**
> *(Replace with your deployed URL after deployment)*

---

## 📸 Screenshots

| Home | Prediction | Dashboard |
|------|-----------|-----------|
| ![Home](screenshots/home.png) | ![Predict](screenshots/predict.png) | ![Dashboard](screenshots/dashboard.png) |

---

## 🎯 Project Overview

Employee attrition is one of the most costly challenges for organizations. This system uses machine learning to:

- **Predict** whether an employee will leave the company
- **Explain** which factors drive that prediction
- **Analyze** historical attrition trends through an interactive dashboard
- **Scale** to bulk predictions via CSV upload
- **Integrate** with other systems via a REST API (FastAPI bonus)

Built using the **IBM HR Analytics Employee Attrition Dataset** (1,470 records, 35 features).

---

## ✨ Features

### 🔮 1. Employee Attrition Prediction
- Input any employee's details via a clean form
- Get instant **Stay / Leave** prediction
- See animated **risk gauge chart** with exact probability

### 🧠 2. Explainable AI
- Top-10 feature importance visualization
- Understand exactly *why* the model makes a prediction
- Powered by tree-based feature importances

### 📊 3. Interactive Analytics Dashboard
- Attrition by **Department**, **Job Role**, **Age**
- **Monthly Income** box plot vs. attrition
- **Overtime** analysis
- **Work-Life Balance** breakdown
- Correlation heatmap of all numeric features
- Marital status sunburst chart

### 📂 4. Bulk CSV Upload
- Upload a CSV of multiple employees
- Download predictions with **Risk Level** (Low / Medium / High)
- Visual pie chart of risk distribution

### 🤖 5. REST API (Bonus)
- FastAPI endpoint at `POST /predict`
- Auto-generated Swagger docs at `/docs`
- Suitable for integration with HRMS systems

### 🐳 6. Docker Support (Bonus)
- Single `docker build` + `docker run` to deploy anywhere

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **ML Models** | Logistic Regression, Random Forest, XGBoost |
| **Data** | Pandas, NumPy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Web App** | Streamlit |
| **REST API** | FastAPI + Uvicorn |
| **Model Persistence** | Joblib |
| **Deployment** | Streamlit Cloud |
| **Containerization** | Docker |
| **Version Control** | Git + GitHub |

---

## 📁 Project Structure

```
employee-attrition-predictor/
│
├── data/                          # Dataset (not committed to Git)
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/
│   └── eda.py                     # Exploratory Data Analysis script
│
├── model/                         # Saved model artifacts (after training)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── model_metadata.pkl
│
├── app/
│   ├── streamlit_app.py           # 🎯 Main Streamlit web application
│   ├── train_model.py             # 🤖 Model training pipeline
│   └── api.py                     # 🔌 FastAPI REST API (bonus)
│
├── screenshots/                   # App screenshots for README
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker containerization
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- pip
- VS Code (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-predictor.git
cd employee-attrition-predictor
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (VS Code Terminal / PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset
1. Go to [IBM HR Analytics Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv`
3. Place it inside the `data/` folder

### Step 5: Train the Model
```bash
python app/train_model.py
```
This will:
- Train Logistic Regression, Random Forest, and XGBoost
- Compare them and pick the best one
- Save model artifacts to `model/`

### Step 6: Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Open your browser at **http://localhost:8501**

### Step 7 (Optional): Run the FastAPI REST API
```bash
uvicorn app.api:app --reload --port 8000
```
- API docs: **http://localhost:8000/docs**
- Health check: **http://localhost:8000/health**

---

## 🐳 Docker

```bash
# Build image
docker build -t attrition-predictor .

# Run container
docker run -p 8501:8501 attrition-predictor

# Open browser at http://localhost:8501
```

---

## ☁️ Deployment on Streamlit Cloud

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "feat: initial commit — AI Employee Attrition Predictor"
git remote add origin https://github.com/YOUR_USERNAME/employee-attrition-predictor.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and main file (`app/streamlit_app.py`)
5. Click **"Deploy"**
6. Your app will be live at `https://YOUR_APP_NAME.streamlit.app`

> **Note:** You must commit the trained `model/*.pkl` files to Git (or use Git LFS) for the deployed app to work. Remove `model/*.pkl` from `.gitignore` before deploying.

---

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.87 | ~0.65 | ~0.55 | ~0.59 | ~0.84 |
| Random Forest | ~0.86 | ~0.68 | ~0.50 | ~0.57 | ~0.83 |
| **XGBoost** ✅ | **~0.88** | **~0.70** | **~0.58** | **~0.63** | **~0.87** |

*Exact values depend on your training run.*

---

## 📡 API Usage

```bash
# POST /predict
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 35,
       "BusinessTravel": "Travel_Rarely",
       "DailyRate": 800,
       "Department": "Research & Development",
       "DistanceFromHome": 5,
       "Education": 3,
       "EducationField": "Life Sciences",
       "EnvironmentSatisfaction": 3,
       "Gender": "Male",
       "HourlyRate": 65,
       "JobLevel": 2,
       "JobRole": "Research Scientist",
       "JobSatisfaction": 3,
       "MaritalStatus": "Single",
       "MonthlyIncome": 5000,
       "MonthlyRate": 14000,
       "NumCompaniesWorked": 2,
       "OverTime": "No",
       "PercentSalaryHike": 15,
       "PerformanceRating": 3,
       "RelationshipSatisfaction": 3,
       "StockOptionLevel": 1,
       "TotalWorkingYears": 10,
       "TrainingTimesLastYear": 3,
       "WorkLifeBalance": 3,
       "YearsAtCompany": 5,
       "YearsInCurrentRole": 3,
       "YearsSinceLastPromotion": 1,
       "YearsWithCurrManager": 3
     }'
```

**Response:**
```json
{
  "prediction": "Stay",
  "will_leave": false,
  "leave_probability_pct": 18.5,
  "stay_probability_pct": 81.5,
  "risk_level": "Low",
  "model_used": "XGBoost"
}
```

---

## 🔮 Future Improvements

- [ ] SHAP waterfall plots per prediction
- [ ] Real-time model monitoring (data drift detection)
- [ ] Authentication & role-based access control
- [ ] Database integration (PostgreSQL) for prediction logging
- [ ] CI/CD pipeline with GitHub Actions
- [ ] A/B testing between model versions
- [ ] Email alerts for high-risk employees
- [ ] Streamlit multi-page app refactor

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙋‍♂️ Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

---

## ⭐ If you found this helpful, please give it a star!
