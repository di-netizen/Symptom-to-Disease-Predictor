import pandas as pd
import joblib

# Load the trained model
model = joblib.load("symptom_rf_model.pkl")

# New patient input (1 = has symptom, 0 = no symptom)
new_patient = pd.DataFrame([{
    'fever': 1,
    'cough': 1,
    'fatigue': 0,
    'headache': 0,
    'nausea': 0,
    'rash': 0,
    'shortness_of_breath': 1,
    'joint_pain': 0,
    'sore_throat': 0,
    'diarrhea': 0
}])

# Make prediction
prediction = model.predict(new_patient)[0]

# Label mapping (reverse)
label_map = {
    0: "COVID-19",
    1: "Common Cold",
    2: "Dengue",
    3: "Flu",
    4: "Measles"
}
disease = label_map[prediction]

print("🧪 Predicted Disease:", disease)

