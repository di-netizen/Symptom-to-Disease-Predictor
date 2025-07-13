import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
df = pd.read_csv("symptom_disease_data.csv")

# 2. Check for missing values
print("🔍 Missing values per column:\n", df.isnull().sum())

# 3. Encode the target disease labels
le = LabelEncoder()
df['disease_label'] = le.fit_transform(df['disease'])
print("\n🔠 Disease classes and their labels:")
for cls, lbl in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {cls} → {lbl}")

# 4. Drop the original 'disease' column
df_encoded = df.drop('disease', axis=1)

# 5. Save the cleaned & encoded data
df_encoded.to_csv("symptom_disease_data_cleaned.csv", index=False)
print("\n✅ Saved cleaned and encoded data as 'symptom_disease_data_cleaned.csv'")
print(df_encoded.head())
