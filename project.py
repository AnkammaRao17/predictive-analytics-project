import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("improved_disease_dataset.csv")

# Features and target
X = df.drop("disease", axis=1)
y = df["disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Test predictions
y_pred = model.predict(X_test)

print("Model Used: Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature importance plot
feature_imp = pd.Series(model.feature_importances_, index=X.columns)
feature_imp = feature_imp.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Train vs Test accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

plt.figure(figsize=(7,5))
plt.bar(["Train Accuracy", "Test Accuracy"], [train_acc, test_acc])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Fixed prediction input (no warning)
sample_input = pd.DataFrame(
    [[1,0,1,0,1,0,0,1,0,1]],
    columns=X.columns
)

prediction = model.predict(sample_input)
print("Predicted Disease:", prediction[0])

