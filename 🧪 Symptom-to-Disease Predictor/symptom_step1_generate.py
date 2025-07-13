import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

n = 1000
symptoms = [
    'fever', 'cough', 'fatigue', 'headache', 'nausea',
    'rash', 'shortness_of_breath', 'joint_pain', 'sore_throat', 'diarrhea'
]

# Simulate binary symptom presence (0 or 1)
data = {symptom: np.random.choice([0, 1], size=n, p=[0.7, 0.3]) for symptom in symptoms}

# Assign diseases based on some symptom logic
diseases = []
for i in range(n):
    s = {symptoms[j]: data[symptoms[j]][i] for j in range(len(symptoms))}
    # Simple rules for simulation:
    if s['fever'] and s['cough'] and s['shortness_of_breath']:
        diseases.append('COVID-19')
    elif s['headache'] and s['nausea'] and s['rash']:
        diseases.append('Measles')
    elif s['fatigue'] and s['joint_pain'] and s['diarrhea']:
        diseases.append('Dengue')
    elif s['sore_throat'] and s['cough']:
        diseases.append('Flu')
    else:
        diseases.append('Common Cold')

data['disease'] = diseases

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("symptom_disease_data.csv", index=False)
print("✅ Generated 'symptom_disease_data.csv' with shape:", df.shape)
print(df.head())
print("\nDisease distribution:\n", df['disease'].value_counts())


