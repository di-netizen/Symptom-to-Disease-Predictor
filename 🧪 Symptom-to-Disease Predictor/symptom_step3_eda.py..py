import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load encoded data
df = pd.read_csv("symptom_disease_data_cleaned.csv")

# Disease label mapping (optional: reverse lookup)
label_map = {
    0: "COVID-19",
    1: "Common Cold",
    2: "Dengue",
    3: "Flu",
    4: "Measles"
}

# 1. Disease distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="disease_label", data=df, palette="Set2")
plt.title("Disease Class Distribution")
plt.xlabel("Disease Label")
plt.ylabel("Count")
plt.xticks(ticks=list(label_map.keys()), labels=list(label_map.values()), rotation=45)
plt.tight_layout()
plt.show()

# 2. Symptom frequency (across all patients)
symptom_cols = df.drop(columns=["disease_label"]).columns
symptom_counts = df[symptom_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=symptom_counts.index, y=symptom_counts.values, palette="viridis")
plt.title("Symptom Frequency Across All Records")
plt.ylabel("Occurrences")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Symptom correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[symptom_cols].corr(), annot=True, cmap="coolwarm", square=True)
plt.title("Symptom Correlation Heatmap")
plt.tight_layout()
plt.show()

