# train_script.py

import pandas as pd
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# Load the cleaned data
df = pd.read_csv("data/clean.csv")

# Define features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Identify numerical and categorical columns
numerical_features = ["RAM", "Screen_Width", "Screen_Height"]
categorical_features = ["Manufacturer", "CPU", "GPU", "Operating System"]

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Fit model
pipeline.fit(X, y)

# Save model and features
joblib.dump(pipeline, "laptop_price_predictor.pkl")
with open("feature_columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)

# Add check points for model version control
import os
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model_v1.pkl")
