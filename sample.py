import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset
from io import StringIO

data = """Breath_Shortness,Sweating,Nausea,Pressure,Heartbeat,Hiccups,H/C
1,0,0,112,92,1,1
0,0,1,173,70,0,1
1,1,0,86,144,1,0
1,0,0,139,148,1,1
0,1,0,160,119,1,0
1,1,0,177,83,1,1
0,1,1,97,148,0,1
1,0,1,168,90,1,0
1,0,1,160,152,0,0
0,1,0,156,81,1,1
1,0,0,97,93,0,0
0,1,0,156,88,1,0
0,0,0,133,137,0,0
1,0,0,167,103,0,1
1,0,0,166,128,1,0
0,0,1,155,120,1,1
1,0,1,85,79,1,0
0,0,0,131,132,1,0
1,1,1,97,121,1,1
0,1,1,171,133,0,1
0,0,1,88,90,1,1
1,0,0,165,87,1,0
0,0,1,109,143,0,0
1,1,1,129,90,1,1
0,1,0,91,102,1,1
1,0,0,111,83,0,1
0,0,0,146,124,0,1
1,0,1,129,147,1,0
1,1,1,108,68,0,1
0,0,1,138,64,0,0
1,0,1,154,160,0,1
1,1,1,107,152,0,0
1,0,1,112,118,1,0
1,1,1,124,114,0,0
1,0,0,118,63,0,1
1,1,0,148,125,0,0
1,0,0,124,108,0,0
1,0,1,177,133,1,0
0,0,0,83,125,0,1
0,0,0,142,159,1,1
0,0,0,117,136,1,0
1,0,1,127,142,0,0
1,1,1,167,139,0,1
1,0,1,84,113,0,1
1,0,0,157,83,0,0
1,0,1,162,160,1,0
1,1,0,86,90,0,0
0,0,1,89,137,1,1
1,1,1,156,125,1,1
1,1,0,88,110,0,0
0,1,1 ,104,80,1,1
0,0,1,137,137,1,1
0,1,1,176,136,0,0
1,1,0,132,133,1,0
1,0,1,108,125,0,0
0,0,1,87,121,1,1
1,0,0,130,137,0,0
0,1,0,180,69,1,0
1,0,1,122,107,1,0
0,1,1,161,157,0,0
0,0,0,124,67,0,1
0,0,0,127,104,0,1
1,1,1,146,96,1,0
0,1,1,165,136,1,0
1,1,0,110,138,1,1
1,0,0,122,129,0,0
0,0,1,133,94,1,0
0,0,0,117,148,1,0
1,1,1,167,145,0,1
0,1,0,82,86,0,0
0,0,1,136,64,0,1
1,0,0,134,113,1,1
0,0,1,109,106,0,0
0,0,1,87,154,0,1
1,1,1,86,119,1,0
1,0,1,129,145,1,0
1,0,0,134,147,0,1
0,0,1,131,116,0,0
1,0,1,113,128,0,1
1,1,1,155,125,1,1
1,1,0,91,109,0,0
1,1,0,167,155,0,1
0,1,0,147,82,0,1
0,1,0,128,64,1,0
1,0,0,154,67,0,1
0,1,1,171,78,1,0
1,0,0,140,114,1,0
0,1,1,171,70,0,0
1,0,1,140,120,1,0
0,0,0,122,126,1,1
0,0,1,80,157,1,1
1,0,1,119,83,0,1
1,1,0,117,80,1,0
0,1,0,92,86,1,0
1,0,0,138,76,1,0
0,1,1,131,64,0,0
1,0,0,86,145,0,0
0,1,0,114,136,0,1
0,0,1,150,157,1,0
1,1,0,134,98,1,0"""

df = pd.read_csv(StringIO(data))
X = df[['Breath_Shortness', 'Sweating', 'Nausea', 'Pressure', 'Heartbeat', 'Hiccups']]
y = df['H/C']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
def predict_heart_condition():
    print("Enter the following details:")
    breath_shortness = int(input("Breath Shortness (0 or 1): "))
    sweating = int(input("Sweating (0 or 1): "))
    nausea = int(input("Nausea (0 or 1): "))
    pressure = float(input("Pressure: "))
    heartbeat = float(input("Heartbeat: "))
    hiccups = int(input("Hiccups (0 or 1): "))
    user_input = np.array([[breath_shortness, sweating, nausea, pressure, heartbeat, hiccups]])
    user_input_scaled = scaler.transform(user_input)
    prediction_prob = model.predict_proba(user_input_scaled)[0][1]
    if prediction_prob > 0.5:
        print("Prediction: Ulcer")
    else:
        print("Prediction: Heart Attack")
    print(f"Confidence: {prediction_prob:.2f}")
predict_heart_condition()
