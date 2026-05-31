# Customer Churn Prediction with Explainable AI (XAI)

An end-to-end Machine Learning web application designed to predict customer churn risk and provide interpretable, transparent model decisions using Explainable AI (XAI) frameworks. This project bridges the gap between predictive accuracy and business interpretability.

## 🚀 Project Overview
Predicting *why* a customer is likely to leave is just as crucial as predicting *if* they will leave. This application uses a machine learning classifier to identify high-risk customers and leverages XAI techniques to highlight the exact feature contributions driving each individual prediction.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Wrangling & ML:** Frameworks like Pandas, NumPy, Scikit-learn
* **Explainable AI:** SHAP (SHapley Additive exPlanations) / LIME
* **Deployment/UI:** Streamlit (or Flask/FastAPI if applicable)

## 📊 Key Insights & XAI Features
* **Global Explanations:** Identifies overall feature importance across the entire dataset (e.g., how factors like `tenure`, `contract_type`, and `monthly_charges` impact overall churn).
* **Local Explanations:** Generates force plots or waterfall charts for individual customers, allowing business analysts to see exactly why a specific user received a high churn risk score.

## 🏃‍♂️ How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/RANUSINGH-S/customer-churn-xai-app.git](https://github.com/RANUSINGH-S/customer-churn-xai-app.git)
cd customer-churn-xai-app
