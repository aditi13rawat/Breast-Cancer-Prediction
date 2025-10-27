import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("🔬 Breast Cancer Prediction App")

st.write("Upload a dataset or use the built-in Breast Cancer dataset to predict malignant or benign tumors.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    from sklearn.datasets import load_breast_cancer
    cancer_data = load_breast_cancer()
    data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    data['target'] = cancer_data.target

st.write("### Dataset Preview")
st.dataframe(data.head())

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"✅ Model Accuracy: {accuracy*100:.2f}%")

st.write("### Make a Custom Prediction")
input_data = []
for feature in X.columns:
    val = st.number_input(feature, value=float(X[feature].mean()))
    input_data.append(val)

if st.button("Predict"):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    st.success("Prediction: **Malignant**" if prediction[0] == 0 else "Prediction: **Benign**")
