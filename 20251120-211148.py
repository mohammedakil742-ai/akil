# -----------------------------------------------
# Logistic Regression Binary Classification
# -----------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# 1. Load a sample binary classification dataset
# (Breast Cancer dataset from sklearn)
# ------------------------------------------------
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# ------------------------------------------------
# 2. Train/Test Split + Standardization
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# 3. Fit Logistic Regression Model
# ------------------------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# ------------------------------------------------
# 4. Evaluation Metrics
# ------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability for class 1

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------------------------
# 5. ROC Curve
# ------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ------------------------------------------------
# 6. Threshold Tuning Demo
# ------------------------------------------------
threshold = 0.4  # try values like 0.3, 0.5, 0.7
y_pred_threshold = (y_proba >= threshold).astype(int)

print(f"Accuracy @ threshold {threshold}:", accuracy_score(y_test, y_pred_threshold))
print(f"Precision @ threshold {threshold}:", precision_score(y_test, y_pred_threshold))
print(f"Recall @ threshold {threshold}:", recall_score(y_test, y_pred_threshold))

# ------------------------------------------------
# Sigmoid Explanation (Optional Print)
# ------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("\nExample of Sigmoid Output:")
print(sigmoid(np.array([-2, 0, 2])))
