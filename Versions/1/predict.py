from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('heart_disease_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            float(data['heart_rate']),
            float(data['systolic_bp']),
            float(data['diastolic_bp']),
            float(data['blood_sugar'])
        ]
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        result = "Heart Disease Risk" if prediction[0] == 1 else "Normal"
        confidence = float(np.max(probability[0]))
        
        return jsonify({
            'prediction': result,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
