from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Fix: use absolute path to find model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "solar_model.pkl"))

@app.route('/')
def home():
    return render_template('map.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hour = datetime.now().hour
    month = datetime.now().month
    features = pd.DataFrame([{
        'Temperature': data['Temperature'],
        'Pressure': data['Pressure'],
        'Humidity': data['Humidity'],
        'WindDirection(Degrees)': data['WindDirection'],
        'Speed': data['Speed'],
        'hour': hour,
        'month': month
    }])
    prediction = model.predict(features)
    radiation = round(float(prediction[0]), 2)
    solar_energy = round(radiation * 5.5, 2)
    return jsonify({
        "predicted_radiation": radiation,
        "solar_energy": solar_energy
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)