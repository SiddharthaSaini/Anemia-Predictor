from flask import Flask, render_template, request
import numpy as np
import pickle
import warnings
import os
import sklearn
import joblib

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model and scaler safely
model = None
scaler = None

# Load model
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("model.pkl not found!")

# Load scaler
if os.path.exists(scaler_path):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading scaler: {e}")
else:
    print("scaler.pkl not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('predict.html', result="Model or Scaler not loaded properly.", reasons=[], advice="Please contact the administrator.")

    try:
        gender = int(request.form['gender'])
        hemoglobin = float(request.form['hemoglobin'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])
        mcv = float(request.form['mcv'])

        input_data = np.array([[gender, hemoglobin, mch, mchc, mcv]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]

        if prediction == 1:
            result = "You may have Anemic Disease."
            reasons = []
            if mch < 27.5:
                reasons.append(f"Your MCH ({mch} pg) is below normal range.")
            if mchc < 33.4:
                reasons.append(f"Your MCHC ({mchc} g/dL) is below normal range.")
            if hemoglobin < 11.6:
                reasons.append(f"Your Hemoglobin ({hemoglobin} g/dL) is below normal range.")
            if mcv < 80:
                reasons.append(f"Your MCV ({mcv} fL) is below normal range.")
            advice = "Please consult with a healthcare provider for proper diagnosis."
        else:
            result = "You may not have Anemia. You are healthy."
            reasons = []
            advice = "Keep maintaining a healthy lifestyle!"

        return render_template('predict.html', result=result, reasons=reasons, advice=advice)

    except Exception as e:
        return render_template('predict.html', result="Error in prediction.", reasons=[], advice=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
