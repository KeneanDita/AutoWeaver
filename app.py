from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import re
import json
import joblib


app = Flask(__name__)

# ========= STEP 1: Load Product List and Categories =========
with open('data/product_list.json', 'r', encoding='utf-8') as f:
    product_map = json.load(f)

categories = list(product_map.keys())

@app.route('/')
def index():
    product_list = []

    # Flatten product list from product_map
    for cat_products in product_map.values():
        product_list.extend(cat_products)

    return render_template('index.html', products=product_list)


# ========= STEP 2: Helper Functions =========

def clean_name_for_lstm(name):
    """
    Converts 'Cheese (Ayib)' → 'Cheese_Ayib' to match model filenames
    """
    import re
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
        print(f"[ERROR] Failed to load Keras model '{filepath}': {e}")
        return None


# === Main Unified Loader ===

def load_models(product):
    product_clean = clean_name_for_lstm(product)
    model_dir = "Models/"

    file_paths = {
        'arima': os.path.join(model_dir, f"{product_clean}_arima_model.pkl"),
        'rf': os.path.join(model_dir, f"{product_clean}_rf_model.pkl"),
        'lstm': os.path.join(model_dir, f"{product_clean}_lstm_model.h5"),
        'scaler': os.path.join(model_dir, f"{product_clean}_scaler.pkl")
    }

    models = {}

    # Load .pkl files using joblib
    for key in ['arima', 'rf', 'scaler']:
        try:
            models[key] = joblib.load(file_paths[key])
        except Exception:
            print(f"[WARNING] '{key}' model missing or corrupted for '{product}'")

    # Load .h5 model using Keras
    try:
        models['lstm'] = load_model(file_paths['lstm'])
    except Exception:
        print(f"[WARNING] 'lstm' model missing or corrupted for '{product}'")

    return models  # Could be partial




# ========= STEP 3: Flask Routes =========

@app.route('/predict', methods=['POST'])
def predict():
    try:
        product = request.form['product']
        features = request.form['features']

        # Parse input features
        feature_array = np.array([float(x) for x in features.split(',')]).reshape(1, -1)

        # Load any available models
        models = load_models(product)
        if not models:
            return jsonify({'error': f"No models found for '{product}'."}), 404

        predictions = {}

        # LSTM prediction (if LSTM and Scaler exist)
        if 'lstm' in models and 'scaler' in models:
            scaled = models['scaler'].transform(feature_array)
            lstm_input = scaled.reshape((1, scaled.shape[1], 1))
            lstm_pred = models['lstm'].predict(lstm_input)[0][0]
            predictions['LSTM Prediction'] = float(lstm_pred)

        # Random Forest prediction
        if 'rf' in models:
            rf_pred = models['rf'].predict(feature_array)[0]
            predictions['Random Forest Prediction'] = float(rf_pred)

        # ARIMA prediction
        if 'arima' in models:
            arima_pred = models['arima'].forecast(steps=1)[0]
            predictions['ARIMA Forecast'] = float(arima_pred)

        if not predictions:
            return jsonify({'error': f"No usable models found for '{product}'."}), 500

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========= STEP 4: Run Server =========

if __name__ == '__main__':
    app.run(debug=True)