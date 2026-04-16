from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# LOAD MODEL + ENCODERS
# -----------------------------
model = joblib.load("price_model.pkl")
crop_encoder   = joblib.load("crop_encoder.pkl")
state_encoder  = joblib.load("state_encoder.pkl")
market_encoder = joblib.load("market_encoder.pkl")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("cleaned_data.csv")
df.columns = df.columns.str.strip()

df.rename(columns={
    "Commodity": "crop",
    "STATE": "state",
    "Market Name": "market",
    "Modal_Price": "price"
}, inplace=True)

print("✅ Dataset loaded:", len(df), "records")
print("🌾 Crops:", list(crop_encoder.classes_))

# -----------------------------
# SAFE ENCODER
# -----------------------------
def safe_transform(encoder, value):
    value = value.strip().title()
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return None

# -----------------------------
# ROOT CHECK
# -----------------------------
@app.route('/')
def home():
    return "✅ Flask API running"

# -----------------------------
# ✅ PREDICT ROUTE (IMPORTANT)
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        crop   = data.get('crop')
        state  = data.get('state')
        market = data.get('market')
        month  = int(data.get('month'))

        # Encode
        crop_enc   = safe_transform(crop_encoder, crop)
        state_enc  = safe_transform(state_encoder, state)
        market_enc = safe_transform(market_encoder, market)

        if None in [crop_enc, state_enc, market_enc]:
            return jsonify({
                "error": "Invalid input. Not found in dataset"
            })

        # Predict
        pred = model.predict([[crop_enc, state_enc, market_enc, month]])

        return jsonify({
            "predicted_price": round(float(pred[0]), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# ✅ DATA ROUTE (IMPORTANT)
# -----------------------------
@app.route('/data', methods=['GET'])
def get_data():
    try:
        return jsonify(
            df[["crop", "state", "market", "price"]].head(100).to_dict(orient="records")
        )
    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == '__main__':
    app.run(port=5001, debug=True)