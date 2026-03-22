import streamlit as st
import joblib
import pandas as pd

st.title("📊 Churn Prediction App")

model = joblib.load("models/model_RandomForest.pkl")

# 🔥 ВАЖЛИВО: беремо фічі з моделі
feature_names = model.feature_names_in_

st.write("Введіть дані:")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

# predict
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])  # ← DataFrame, не numpy!

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Результат:")

    if prediction == 1:
        st.error(f"❌ Висока ймовірність відтоку ({probability:.2f})")
    else:
        st.success(f"✅ Низька ймовірність відтоку ({probability:.2f})")