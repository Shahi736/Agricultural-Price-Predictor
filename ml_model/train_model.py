import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("agri_data.csv")

# Convert text columns to numeric
df["crop"] = df["crop"].astype("category").cat.codes
df["region"] = df["region"].astype("category").cat.codes

X = df[["crop", "region"]]
y = df["price"]

# Better model than LinearRegression
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "price_model.pkl")

print("Model trained successfully with CSV dataset!")