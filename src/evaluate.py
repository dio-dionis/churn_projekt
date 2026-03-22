import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

# load data
df = pd.read_csv("data/internet_service_churn.csv")

if "id" in df.columns:
    df = df.drop("id", axis=1)

X = df.drop("churn", axis=1)
y = df["churn"]

# split (ВАЖЛИВО: той самий random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# load models
models = {
    "LogisticRegression": joblib.load("models/model_LogisticRegression.pkl"),
    "RandomForest": joblib.load("models/model_RandomForest.pkl")
}

accuracy_list = []
f1_list = []

# оцінка
for name, model in models.items():
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    accuracy_list.append(acc)
    f1_list.append(f1)

    print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

# 📊 BAR CHART
plt.figure()
plt.bar(models.keys(), accuracy_list)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

plt.figure()
plt.bar(models.keys(), f1_list)
plt.title("Model F1 Score Comparison")
plt.ylabel("F1 Score")
plt.show()

# 📈 ROC CURVE
plt.figure()

for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1])
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()