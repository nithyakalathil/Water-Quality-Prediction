from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define model directory
MODEL_DIR = r"E:\model"

# Load trained models with error handling
try:
    dt_model_path = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
    knn_model_path = os.path.join(MODEL_DIR, "knn_model.pkl")

    dt_model = joblib.load(dt_model_path)
    knn_model = joblib.load(knn_model_path)

    print("✅ Models loaded successfully!")

except FileNotFoundError:
    print("❌ Model files not found. Check model directory paths.")
    dt_model = None
    knn_model = None

# Function to fetch TDS from ThingSpeak
def fetch_tds():
    THINGSPEAK_CHANNEL_ID = "YOUR_CHANNEL_ID"
    THINGSPEAK_FIELD_ID = "YOUR_FIELD_ID"
    THINGSPEAK_API_KEY = "YOUR_READ_API_KEY"

    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/fields/{THINGSPEAK_FIELD_ID}/last.json?api_key={THINGSPEAK_API_KEY}"

    try:
        response = requests.get(url, timeout=5)  # Set timeout for reliability
        response.raise_for_status()
        data = response.json()

        if "field1" in data and data["field1"]:
            try:
                return float(data["field1"])
            except ValueError:
                print("⚠️ Invalid TDS value received, setting to 0")
                return 0  # Default to 0 if parsing fails
        else:
            print("⚠️ Missing field1 in API response, setting TDS to 0")
            return 0  # Default to 0 if field is missing

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching TDS: {e}, setting TDS to 0")
        return 0  # Default to 0 if request fails

@app.route('/')
def home():
    return jsonify({"message": "Welcome to Water Quality Prediction API"}), 200

@app.route('/predict', methods=['POST'])
def predict_water_quality():
    try:
        if dt_model is None or knn_model is None:
            return jsonify({"error": "Models not loaded"}), 500

        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Fetch TDS from ThingSpeak
        tds_value = fetch_tds()
        data['Solids'] = tds_value

        # Required features for prediction
        required_fields = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                           "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

        # Validate input fields
        for field in required_fields:
            if field not in data or not isinstance(data[field], (int, float)) or np.isnan(data[field]):
                return jsonify({"error": f"Missing or invalid value for {field}"}), 400

        # Extract features for model input
        input_features = np.array([[data[field] for field in required_fields]])

        # Make predictions
        dt_prediction = dt_model.predict(input_features)
        knn_prediction = knn_model.predict(input_features)

        response = {
            "Decision_Tree_Prediction": int(dt_prediction[0]),
            "KNN_Prediction": int(knn_prediction[0])
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
