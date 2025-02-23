import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load heart attack dataset
heart_data = pd.read_csv('HeartAttackDataSet.csv')

# Select relevant features
X_heart = heart_data[['Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar']]
y_heart = heart_data['Result'].map({'positive': 1, 'negative': 0})

# Drop any rows with missing values
X_heart = X_heart.dropna()
y_heart = y_heart[X_heart.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Model evaluation
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizations
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
})
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.sort_values('importance', ascending=False))
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()

# Save the model and scaler
joblib.dump(rf_model, 'heart_disease_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("\nModel and scaler have been saved successfully!")
