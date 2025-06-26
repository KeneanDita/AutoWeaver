from flask import Flask, render_template, request, jsonify
import os
import json
import re
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ========= STEP 1: Load Category and Product Info =========
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'Models')

with open(os.path.join(DATA_DIR, 'product_list.json'), 'r', encoding='utf-8') as f:
    product_map = json.load(f)

categories = list(product_map.keys())


# ========= STEP 2: Utility Functions =========

def clean_name_for_lstm(name):
    """Convert names like 'Cheese (Ayib)' to 'Cheese_Ayib'"""
    name = name.strip()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', '_', name)     # Replace spaces with underscores
    return name

def load_pickle_model(filepath):
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"[ERROR] Failed to load pickle file '{filepath}': {e}")
        return None

def load_h5_model(filepath):
    try:
        return load_model(filepath)
    except Exception as e:
        print(f"[ERROR] Failed to load H5 model '{filepath}': {e}")
        return None

def load_models(product):
    product_clean = clean_name_for_lstm(product)

    paths = {
        'arima':  os.path.join(MODELS_DIR, f"{product_clean}_arima_model.pkl"),
        'rf':     os.path.join(MODELS_DIR, f"{product_clean}_rf_model.pkl"),
        'scaler': os.path.join(MODELS_DIR, f"{product_clean}_scaler.pkl"),
        'lstm':   os.path.join(MODELS_DIR, f"{product_clean}_lstm_model.h5"),
    }

    models = {}
    if os.path.exists(paths['arima']):
        models['arima'] = load_pickle_model(paths['arima'])
    if os.path.exists(paths['rf']):
        models['rf'] = load_pickle_model(paths['rf'])
    if os.path.exists(paths['scaler']):
        models['scaler'] = load_pickle_model(paths['scaler'])
    if os.path.exists(paths['lstm']):
        models['lstm'] = load_h5_model(paths['lstm'])

    return models


# ========= STEP 3: Routes =========

@app.route('/')
def index():
    return render_template('index.html', categories=categories, product_map=product_map)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        product = request.form['product']
        features = request.form['features']

        # Convert input to array
        feature_array = np.array([float(x) for x in features.split(',')]).reshape(1, -1)

        models = load_models(product)
        if not models:
            return jsonify({'error': f"No models found for '{product}'."}), 404

        predictions = {}

        # LSTM
        if models.get('scaler') and models.get('lstm'):
            try:
                scaled = models['scaler'].transform(feature_array)
                lstm_input = scaled.reshape((1, scaled.shape[1], 1))
                lstm_pred = models['lstm'].predict(lstm_input)[0][0]
                predictions['LSTM Prediction'] = float(lstm_pred)
            except Exception as e:
                print(f"[LSTM ERROR] {e}")

        # Random Forest
        if models.get('rf'):
            try:
                rf_pred = models['rf'].predict(feature_array)[0]
                predictions['Random Forest Prediction'] = float(rf_pred)
            except Exception as e:
                print(f"[RF ERROR] {e}")

        # ARIMA
        if models.get('arima'):
            try:
                arima_pred = models['arima'].forecast(steps=1)[0]
                predictions['ARIMA Forecast'] = float(arima_pred)
            except Exception as e:
                print(f"[ARIMA ERROR] {e}")

        if not predictions:
            return jsonify({'error': f"No usable models available for '{product}'."}), 500

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ========= STEP 4: Run App =========

if __name__ == '__main__':
    app.run(debug=True)
