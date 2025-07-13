# 🩺 Symptom-to-Disease Predictor

This machine learning project predicts the most likely disease based on a user’s reported symptoms. It uses a Random Forest classifier trained on synthetic data to simulate real-life health conditions.

---

## 📌 Features

- ✅ Generates a dataset of 10 common symptoms across 5 diseases  
- 🧹 Cleans and encodes the data using LabelEncoder  
- 🧠 Trains a Random Forest Classifier  
- 📊 Evaluates with confusion matrix & classification report  
- 🔮 Predicts disease for new patient inputs

---

## 🗃️ Dataset Overview

Each row in the dataset represents a patient with symptoms like:
- `fever`, `cough`, `fatigue`, `headache`, `nausea`,  
- `rash`, `shortness_of_breath`, `joint_pain`, `sore_throat`, `diarrhea`

The model predicts one of the following diseases:
- **COVID-19**
- **Flu**
- **Dengue**
- **Measles**
- **Common Cold**

---

## 🧪 How It Works

1. **Synthetic data generation** using symptom logic  
2. **Data encoding** and preprocessing  
3. **Model training** with Random Forest  
4. **Evaluation** using accuracy, confusion matrix, and classification report  
5. **Prediction** from new user symptom input  

---

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run training and evaluation
python train_model.py

# Predict disease for a new patient
python predict_new_patient.py

symptom-to-disease-predictor/
│
├── symptom_disease_data.csv               # Raw synthetic data
├── symptom_disease_data_cleaned.csv       # Cleaned + encoded data
├── symptom_rf_model.pkl                   # Trained Random Forest model
├── train_model.py                         # Model training + evaluation
├── predict_new_patient.py                 # Symptom-based prediction
├── requirements.txt                       # Project dependencies
├── LICENSE                                # Open-source license
└── README.md                              # You're here!

