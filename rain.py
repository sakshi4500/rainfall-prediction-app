import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Rainfall Prediction App",
    page_icon="🌧",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🌧 Rainfall Prediction Web Application")

st.markdown(
    "<hr style='border:2px solid #00C2A8;'>",
    unsafe_allow_html=True
)

st.write(
    "This Machine Learning app predicts whether it will rain tomorrow "
    "using a Random Forest Classifier."
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("rainfall.csv")
    df = df.dropna()
    return df

df = load_data()

# ---------------- DATA PREPARATION ----------------
X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL TRAINING ----------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open("rainfall_model.pkl", "wb") as file:
    pickle.dump(model, file)

# ---------------- MODEL EVALUATION ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.success(f"Model Accuracy: {accuracy:.2f}")

st.markdown(
    "<hr style='border:2px solid #1f77b4;'>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📝 Enter Weather Details")

input_data = []

for col in X.columns:
    value = st.sidebar.number_input(
        label=col,
        min_value=float(X[col].min()),
        max_value=float(X[col].max()),
        value=float(X[col].mean())
    )
    input_data.append(value)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Rainfall"):

    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0]

    st.subheader("🔎 Prediction Result")

    if prediction == 1:
        st.error("🌧 It WILL Rain Tomorrow")
    else:
        st.success("☀ It will NOT Rain Tomorrow")

    st.info(f"Rain Probability: {probability[1]:.2f}")

st.markdown(
    "<hr style='border:2px solid #00C2A8;'>",
    unsafe_allow_html=True
)

# ---------------- DATA PREVIEW ----------------
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)