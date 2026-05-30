import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier  # <-- NEW: Imported XGBoost

# 1. Load the dataset
df = pd.read_csv('customer_churn_data.csv')

# Drop customerID as it's just an identifier
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# 2. Handle missing values/data types
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Map target variable 'Churn' to 0 and 1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Split Features and Target
X = df.drop(columns=['Churn'])
y = df['Churn']

# 3. Identify column types
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
categorical_cols = [col for col in X.columns if col not in numerical_cols]

# 4. Create preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# 5. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train and save Random Forest Pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
rf_pipeline.fit(X_train, y_train)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_pipeline, f)

# 7. Train and save Logistic Regression Pipeline
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
lr_pipeline.fit(X_train, y_train)

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)

# 8. --- NEW: Train and save XGBoost Pipeline ---
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))])
xgb_pipeline.fit(X_train, y_train)

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_pipeline, f)

print("Preprocessing and all three models (RF, LR, XGB) trained and saved successfully!")