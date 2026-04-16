import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("Working directory:", os.getcwd())

# ── Load data ─────────────────────────────────────────────────────────────
file_path = "cleaned_data.csv"
if not os.path.exists(file_path):
    print("ERROR: cleaned_data.csv not found!")
    print("Run data_clean.py first, then run this file.")
    exit()

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Rename columns to standard names
df.rename(columns={
    "STATE":        "state",
    "Market Name":  "market",
    "Commodity":    "crop",
    "Modal_Price":  "price",
    "Price Date":   "date"
}, inplace=True)

# ── Process date → month ──────────────────────────────────────────────────
if "date" in df.columns and "month" not in df.columns:
    df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["month"] = df["date"].dt.month

# ── Keep required columns ─────────────────────────────────────────────────
df = df[["crop", "state", "market", "month", "price"]].dropna()
df = df[df["price"] > 0]

# ── Standardize text (Title Case — MUST match api.py) ────────────────────
df["crop"]   = df["crop"].str.strip().str.title()
df["state"]  = df["state"].str.strip().str.title()
df["market"] = df["market"].str.strip().str.title()

print(f"Records: {len(df)}")
print(f"Crops:   {sorted(df['crop'].unique().tolist())}")
print(f"States:  {sorted(df['state'].unique().tolist())}")

# ── Encode ────────────────────────────────────────────────────────────────
crop_encoder   = LabelEncoder()
state_encoder  = LabelEncoder()
market_encoder = LabelEncoder()

df["crop_enc"]   = crop_encoder.fit_transform(df["crop"])
df["state_enc"]  = state_encoder.fit_transform(df["state"])
df["market_enc"] = market_encoder.fit_transform(df["market"])

# ── Train ─────────────────────────────────────────────────────────────────
X = df[["crop_enc", "state_enc", "market_enc", "month"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────
preds = model.predict(X_test)
print(f"MAE : Rs. {round(mean_absolute_error(y_test, preds))}")
print(f"R2  : {round(r2_score(y_test, preds), 3)}")

# ── Save ──────────────────────────────────────────────────────────────────
joblib.dump(model,          "price_model.pkl")
joblib.dump(crop_encoder,   "crop_encoder.pkl")
joblib.dump(state_encoder,  "state_encoder.pkl")
joblib.dump(market_encoder, "market_encoder.pkl")

print("Model and encoders saved successfully!")
print("Now run: python api.py")