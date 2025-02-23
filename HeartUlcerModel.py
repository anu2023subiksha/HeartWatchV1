import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class HealthPredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = [
            'age', 'sex', 'pain_level', 'nausea', 'sweating', 
            'chest_pressure', 'breath_shortness', 'stomach_pain',
            'stomach_acidity', 'weight_loss', 'vomiting_blood', 
            'diearea', 'stress_level'
        ]
        # Define normal ranges for features
        self.normal_ranges = {
            'pain_level': (0, 1),  # 0-1 is normal, 2-3 is concerning
            'stomach_acidity': (0, 2),  # 0-2 is normal
            'stress_level': (0, 4),  # 0-4 is normal
            'nausea': 0,
            'sweating': 0,
            'chest_pressure': 0,
            'breath_shortness': 0,
            'stomach_pain': 0,
            'weight_loss': 0,
            'vomiting_blood': 0,
            'diearea': 0,
            'black_stool': 0
        }
        
    def is_condition_normal(self, data):
        """Check if the symptoms indicate a normal condition"""
        if isinstance(data, dict):
            # Check numerical ranges
            if (float(data['pain_level']) > self.normal_ranges['pain_level'][1] or
                float(data['stomach_acidity']) > self.normal_ranges['stomach_acidity'][1] or
                float(data['stress_level']) > self.normal_ranges['stress_level'][1]):
                return False
            
            # Check binary symptoms
            binary_symptoms = ['nausea', 'sweating', 'chest_pressure', 'breath_shortness',
                             'stomach_pain', 'weight_loss', 'vomiting_blood', 'diearea']
            
            # If any major symptom is present, it's not normal
            symptom_count = sum(int(data[symptom]) for symptom in binary_symptoms)
            if symptom_count > 1:  # Allow at most one mild symptom for normal condition
                return False
                
            return True
        return False

    def check_for_outliers(self, data):
        """Check and adjust outliers in the input data"""
        if isinstance(data, dict):
            cleaned_data = data.copy()
            
            # Clamp numerical values to valid ranges
            cleaned_data['pain_level'] = max(0, min(float(cleaned_data['pain_level']), 3))
            cleaned_data['stomach_acidity'] = max(0, min(float(cleaned_data['stomach_acidity']), 5))
            cleaned_data['stress_level'] = max(0, min(float(cleaned_data['stress_level']), 10))
            
            # Ensure binary values are 0 or 1
            binary_fields = ['sex', 'nausea', 'sweating', 'chest_pressure', 'breath_shortness',
                           'stomach_pain', 'weight_loss', 'vomiting_blood', 'diearea']
            
            for field in binary_fields:
                cleaned_data[field] = 1 if int(cleaned_data[field]) >= 1 else 0
                
            # Ensure age is within reasonable range (0-120)
            cleaned_data['age'] = max(0, min(int(cleaned_data['age']), 120))
            
            return cleaned_data
        return data
        
    def prepare_combined_dataset(self):
        """Prepare a combined dataset from heart attack, ulcer, and additional heart attack data"""
        # Load heart attack data
        heart_data = pd.read_csv('heart.csv')
        additional_heart_data = pd.read_csv('HeartAttackDataSet.csv')
        # Map heart attack features to common features
        heart_features = pd.DataFrame({
            'age': heart_data['age'],
            'sex': heart_data['sex'],
            'pain_level': heart_data['cp'],
            'nausea': 0,  # Default values for heart attack cases
            'sweating': heart_data['exang'],
            'chest_pressure': 1,
            'breath_shortness': (heart_data['thalach'] > 150).astype(int),
            'stomach_pain': 0,
            'stomach_acidity': 0,
            'weight_loss': 0,
            'vomiting_blood': 0,
            'diearea': 0,
            'stress_level': heart_data['oldpeak']
        })
        heart_features['condition'] = 'heart_attack'

        # Map additional heart attack features to common features
        additional_heart_features = pd.DataFrame({
            'age': additional_heart_data['Age'],
            'sex': additional_heart_data['Gender'],
            'pain_level': additional_heart_data['Heart rate'],  # Assuming heart rate as a proxy for pain level
            'nausea': 0,
            'sweating': (additional_heart_data['Systolic blood pressure'] > 140).astype(int),  # Assuming high blood pressure as sweating
            'chest_pressure': 1,
            'breath_shortness': (additional_heart_data['Diastolic blood pressure'] > 90).astype(int),  # Assuming high diastolic pressure as breath shortness
            'stomach_pain': 0,
            'stomach_acidity': 0,
            'weight_loss': 0,
            'vomiting_blood': 0,
            'diearea': 0,
            'stress_level': additional_heart_data['Blood sugar']  # Assuming blood sugar as a proxy for stress level
        })
        additional_heart_features['condition'] = 'heart_attack'

        # Load ulcer data
        ulcer_data = pd.read_csv('ulcer_dataset.csv')
        # Map ulcer features to common features
        ulcer_features = pd.DataFrame({
            'age': ulcer_data['age'],
            'sex': ulcer_data['sex'],
            'pain_level': ulcer_data['pain_level'],
            'nausea': ulcer_data['nausea'],
            'sweating': 0,  # Default values for ulcer cases
            'chest_pressure': 0,
            'breath_shortness': 0,
            'stomach_pain': 1,
            'stomach_acidity': ulcer_data['stomach_acidity'],
            'weight_loss': ulcer_data['weight_loss'],
            'vomiting_blood': ulcer_data['vomiting_blood'],
            'diearea': ulcer_data['black_stool'],
            'stress_level': ulcer_data['stress_level']
        })
        ulcer_features['condition'] = 'ulcer'

        # Add synthetic normal cases
        normal_cases = pd.DataFrame([
            {
                'age': np.random.randint(18, 80),
                'sex': np.random.randint(0, 2),
                'pain_level': np.random.uniform(0, 1),
                'nausea': 0,
                'sweating': 0,
                'chest_pressure': 0,
                'breath_shortness': 0,
                'stomach_pain': 0,
                'stomach_acidity': np.random.uniform(0, 2),
                'weight_loss': 0,
                'vomiting_blood': 0,
                'diearea': 0,
                'stress_level': np.random.uniform(0, 4),
                'condition': 'normal'
            } for _ in range(100)  # Add 100 normal cases
        ])

        # Combine datasets
        self.data = pd.concat([heart_features, additional_heart_features, ulcer_features, normal_cases], ignore_index=True)
        
        # Prepare features and target
        self.X = self.data[self.features]
        self.y = self.data['condition']
        
    def preprocess_data(self):
        """Preprocess the data including handling missing values and scaling"""
        # Handle missing values
        self.X = self.X.fillna(self.X.mean())
        
        # Scale the features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        
    def train_model(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, self.X_scaled, self.y, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
    def evaluate_model(self):
        """Evaluate the model and print detailed metrics"""
        y_pred = self.model.predict(self.X_test)
        
        print("\nModel Evaluation Metrics:")
        print("-------------------------")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
    def predict(self, input_data):
        """Make predictions on new data"""
        # Clean and check for outliers
        cleaned_data = self.check_for_outliers(input_data)
        
        # Check if condition is normal
        if self.is_condition_normal(cleaned_data):
            return {
                'prediction': 'normal',
                'probability': 0.9,  # High confidence for normal condition
                'message': 'Your symptoms appear to be normal. However, if symptoms persist, please consult a healthcare professional.'
            }
        
        # Proceed with model prediction for non-normal cases
        if isinstance(cleaned_data, dict):
            cleaned_data = pd.DataFrame([cleaned_data])
        
        # Scale the input data
        scaled_data = self.scaler.transform(cleaned_data[self.features])
        prediction = self.model.predict(scaled_data)
        probabilities = self.model.predict_proba(scaled_data)
        
        # Get the predicted condition and its probability
        condition = prediction[0]
        prob_dict = dict(zip(self.model.classes_, probabilities[0]))
        
        message = ""
        if condition == 'heart_attack':
            message = "Warning: Symptoms suggest possible heart attack. Seek immediate medical attention!"
        elif condition == 'ulcer':
            message = "Warning: Symptoms suggest possible ulcer. Please consult a healthcare professional."
        
        return {
            'prediction': condition,
            'probability': prob_dict[condition],
            'message': message
        }
    
    def save_model(self):
        """Save the trained model and scaler"""
        model_filename = 'health_prediction_model.joblib'
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }, model_filename)
        print(f"\nModel saved as {model_filename}")
        
    def load_saved_model(self):
        """Load a previously saved model"""
        model_filename = 'health_prediction_model.joblib'
        saved_data = joblib.load(model_filename)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.features = saved_data['features']

def train_and_save_model():
    """Train and save the combined prediction model"""
    print("Training Health Prediction Model...")
    model = HealthPredictionModel()
    model.prepare_combined_dataset()
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.save_model()

if __name__ == "__main__":
    train_and_save_model()
