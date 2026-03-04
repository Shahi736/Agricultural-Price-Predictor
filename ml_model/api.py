from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    crop = data.get("crop", "").strip().lower()
    region = data.get("region", "").strip().lower()

    df = pd.read_csv("agri_data.csv")

    result = df[(df["crop"] == crop) & (df["region"] == region)]

    if result.empty:
        return jsonify({"predicted_price": "No data found"})

    return jsonify({"predicted_price": int(result.iloc[0]["price"])})

@app.route("/data", methods=["GET"])
def get_data():
    df = pd.read_csv("agri_data.csv")
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)