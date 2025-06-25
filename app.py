from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load all models and scalers
model_dir = 'models'
models = {}
scalers = {}

for i in range(1, 241):  # Assuming 240 models
    model_name = f'model_{i}'
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    scaler_path = os.path.join(model_dir, f'{model_name}_scaler.pkl')

    try:
        models[model_name] = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scalers[model_name] = pickle.load(f)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Example input fields (should match your form)
        input_data = request.form.getlist('feature')  # e.g., 10 features
        model_id = request.form.get('model_id')       # e.g., '1' to '240'

        model_key = f'model_{model_id}'

        if model_key not in models or model_key not in scalers:
            return jsonify({'error': f'Model {model_id} not found'}), 404

        input_array = np.array(input_data, dtype=float).reshape(1, -1)
        scaled_input = scalers[model_key].transform(input_array)
        scaled_input = scaled_input.reshape((1, scaled_input.shape[1], 1))  # LSTM shape

        prediction = models[model_key].predict(scaled_input)
        prediction = prediction.flatten()[0]

        return jsonify({'prediction': float(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
