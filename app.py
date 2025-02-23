from flask import Flask, request, jsonify, render_template
from HeartUlcerModel import HealthPredictionModel
from flask_cors import CORS
import logging
from waitress import serve

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the model
model = HealthPredictionModel()
try:
    logger.info("Loading saved model...")
    model.load_saved_model()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise Exception("Pre-trained model not found. Please ensure 'health_prediction_model.joblib' exists.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logger.info(f"Received prediction request with data: {data}")
        
        # Validate required fields
        required_fields = ['age', 'sex', 'pain_level', 'nausea', 'sweating', 
                         'chest_pressure', 'breath_shortness', 'stomach_pain',
                         'stomach_acidity', 'weight_loss', 'vomiting_blood', 
                         'diearea', 'stress_level']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f'Missing required fields: {missing_fields}'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Convert string values to appropriate types
        processed_data = {}
        for field in required_fields:
            try:
                if field in ['age', 'sex', 'nausea', 'sweating', 'chest_pressure', 
                           'breath_shortness', 'stomach_pain', 'weight_loss', 
                           'vomiting_blood', 'diearea']:
                    processed_data[field] = int(data[field])
                else:
                    processed_data[field] = float(data[field])
            except ValueError as e:
                error_msg = f'Invalid value for field {field}: {data[field]}'
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
            
        result = model.predict(processed_data)
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 400

# def run_production_server():
#     logger.info("Starting production server on http://localhost:8000")
#     serve(app, host='0.0.0.0', port=8000)

def run_development_server():
    logger.info("Starting development server on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)

if __name__ == '__main__':
#     # Use development server for debugging
     run_development_server()
#     # For production, uncomment the following line and comment out the above line
#     run_production_server()
