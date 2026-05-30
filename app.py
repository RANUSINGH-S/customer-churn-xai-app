import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Set up page configurations
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard (with Explainable AI)")
st.write("Input customer demographics and service details to evaluate risk and view SHAP interpretability insights.")

# Load models safely using caching
@st.cache_resource
def load_model(model_name):
    # Map the dropdown selection to the correct pickle file
    if model_name == "Random Forest":
        filename = 'rf_model.pkl'
    elif model_name == "XGBoost":
        filename = 'xgb_model.pkl'
    else:
        filename = 'lr_model.pkl'
        
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Sidebar for Model Selection
st.sidebar.header("🧠 Model Configurations")
# Added XGBoost to the model choices to match resume statement
selected_model_name = st.sidebar.selectbox("Choose Prediction Model", ["Random Forest", "XGBoost", "Logistic Regression"])

try:
    model = load_model(selected_model_name)
except FileNotFoundError:
    st.error(f"Could not find the saved file for {selected_model_name}. Please make sure you have generated and saved '{selected_model_name.lower().replace(' ', '_')}_model.pkl' via your training script.")
    st.stop()

# Layout layout split into 3 columns for organized user inputs
st.subheader("👤 Enter Customer Details")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# Financial inputs section placed horizontally below
st.markdown("---")
f1, f2 = st.columns(2)
with f1:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.5)
with f2:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=1.0)

# Reconstruct the user inputs into a DataFrame format matching the training features
input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}])

st.markdown("---")

# Prediction Execution Action Button
if st.button("Analyze Churn Risk", type="primary"):
    # Generate prediction and probability scores
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    churn_probability = probabilities[1] * 100

    # Display Results
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Churning!**")
        st.metric(label="Churn Probability", value=f"{churn_probability:.2f}%")
        st.progress(int(churn_probability))
    else:
        st.success(f"✅ **Low Risk: Customer likely to stay.**")
        st.metric(label="Churn Probability", value=f"{churn_probability:.2f}%")
        st.progress(int(churn_probability))
        
    # --- NEW: EXPLAINABLE AI (SHAP) INTEGRATION BLOCK ---
    st.markdown("---")
    st.subheader("🕵️‍♂️ Explainable AI (XAI) - Feature Attribution Breakdown")
    
    # SHAP TreeExplainer works cleanly on Random Forest and XGBoost tree models
    if selected_model_name in ["Random Forest", "XGBoost"]:
        with st.spinner("Computing feature attribution SHAP values..."):
            try:
                # 1. Isolate the internal steps of your scikit-learn Pipeline
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']
                
                # 2. Vectorize/encode the individual UI input row using the pipeline's encoder
                transformed_input = preprocessor.transform(input_data)
                
                # 3. Reconstruct feature names matching the transformed array dimensions
                encoded_feature_names = preprocessor.get_feature_names_out()
                transformed_df = pd.DataFrame(transformed_input, columns=encoded_feature_names)
                
                # 4. Generate SHAP explanations using TreeExplainer
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer(transformed_df)
                
                # 5. Handle shape adjustments if the model outputs log-odds array shapes differently
                if len(shap_values.shape) == 3:  # Multi-class output edge case
                    shap_obj = shap_values[0, :, 1]
                else:
                    shap_obj = shap_values[0]

                # 6. Build the waterfall plot and render it inside Streamlit
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_obj, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("ℹ️ **How to read this plot:** Red bars increase the probability of customer churn, while blue bars decrease it. Longer bars indicate features that had a higher impact on this specific prediction.")
                
            except Exception as e:
                st.warning(f"Could not render SHAP plot. Ensure your saved model structure matches the training pipeline pipeline. Error: {e}")
    else:
        # Logistic Regression uses coefficients rather than tree leaves, so we flag it accordingly
        st.info("SHAP visual explanations are configured for Tree-based models (Random Forest & XGBoost). Select one of those models in the sidebar to see real-time feature attributions.")