import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. load data
df = pd.read_csv("data/internet_service_churn.csv")

# 2. clean
if "id" in df.columns:
    df = df.drop("id", axis=1)

# 3. split target
X = df.drop("churn", axis=1)
y = df["churn"]

# 4. split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. pipelines (🔥 ГОЛОВНЕ)
models = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestClassifier(n_estimators=100))
    ])
}

results = {}

# 6. training
for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[name] = (acc, f1)

    # save pipeline (🔥 важливо)
    joblib.dump(model, f"models/model_{name}.pkl")

# 7. result
print("Results:")
for k, v in results.items():
    print(f"{k}: Accuracy={v[0]:.4f}, F1={v[1]:.4f}")