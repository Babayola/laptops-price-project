import joblib
import pandas as pd
import sys

# Get the model path from command-line argument
model_path = sys.argv[sys.argv.index("--model") + 1]

# Load the model
model = joblib.load(model_path)

# Example: Load test data
# For this example, you can reuse data/clean.csv or create a small sample
df = pd.read_csv("data/clean.csv")

# Drop target column if it exists
if "Price" in df.columns:
    df = df.drop(columns=["Price"])

# Make predictions
predictions = model.predict(df)

# Save predictions to file
pd.DataFrame(predictions, columns=["Predicted_Price"]).to_csv("predictions.csv", index=False)

print("✅ Predictions saved to predictions.csv")
