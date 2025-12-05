import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("improved_disease_dataset.csv")
print(df.head())

# Basic EDA

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nClass Distribution:\n")
print(df['disease'].value_counts())

plt.figure(figsize=(12,4))
sns.countplot(y=df['disease'])
plt.title("Disease Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.drop("disease", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Symptoms")
plt.show()

# Split features and target

X = df.drop("disease", axis=1)
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree Classifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
print("\nDecision Tree Accuracy:", dt_acc)
print("\nDecision Tree Report:\n")
print(classification_report(y_test, dt_pred))

dt_cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(10,6))
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Random Forest Classifier

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
print("\nRandom Forest Accuracy:", rf_acc)
print("\nRandom Forest Report:\n")
print(classification_report(y_test, rf_pred))

rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(10,6))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance (Random Forest)

feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Symptoms")
plt.show()

# Accuracy comparison graph

plt.figure(figsize=(7,5))
plt.bar(["Decision Tree", "Random Forest"], [dt_acc, rf_acc])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Prediction Example 1 (List Input)

sample1 = pd.DataFrame(
    [[1,0,1,0,1,0,0,1,0,1]],
    columns=X.columns
)

pred1 = rf_model.predict(sample1)
print("Predicted Disease (Sample 1):", pred1[0])

# Prediction Example 2 (Dictionary Input)

sample2 = {
    "fever": 1,
    "headache": 0,
    "nausea": 1,
    "vomiting": 0,
    "fatigue": 1,
    "joint_pain": 0,
    "skin_rash": 0,
    "cough": 1,
    "weight_loss": 0,
    "yellow_eyes": 1
}

sample_df = pd.DataFrame([sample2])
prediction2 = rf_model.predict(sample_df)
print("Predicted Disease (Sample 2):", prediction2[0])

# Final Evaluation Summary

print("\n===== Evaluation Summary =====")
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

print("\nDecision Tree Confusion Matrix:\n", dt_cm)
print("\nRandom Forest Confusion Matrix:\n", rf_cm)
