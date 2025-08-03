from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Scale Time and Amount (assume they are first two)
        features[:, [0, 1]] = scaler.transform(features[:, [0, 1]])

        # Predict
        prediction = model.predict(features)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"

        return render_template('index.html', prediction_text=f'Result: {result}')
    except:
        return render_template('index.html', prediction_text='❌ Invalid Input!')

if __name__ == '__main__':
    app.run(debug=True)
