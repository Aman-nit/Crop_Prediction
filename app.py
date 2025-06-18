from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd
import joblib


# Load model, scaler, and label encoder
model = joblib.load(open('crop_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))
le = joblib.load(open('label_encoder.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Crop_predict', methods = ['POST'])
def Crop_prediction():
    sample_input = request.get_json()
    # Input sample and column names
    #sample_input = [90, 42, 43, 20.87, 82.00, 6.5, 202.93]
    column_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    sample_input = [sample_input[col] for col in column_names]
    # Create DataFrame
    sample_df = pd.DataFrame([sample_input], columns=column_names)

    # Scale and rewrap in DataFrame to preserve feature names
    scaled_array = scaler.transform(sample_df)
    scaled_df = pd.DataFrame(scaled_array, columns=column_names)


    #    Predict (now with correct feature names)
    prediction = model.predict(scaled_df)
    predicted_crop = le.inverse_transform(prediction)[0]
    return jsonify({'crop': predicted_crop})



if __name__ == '__main__':
    app.run(debug=True , host= '0.0.0.0')

