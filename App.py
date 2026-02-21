import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Rainfall Prediction", page_icon="🌧")

st.title("🌧 Rainfall Prediction App")

# Load data
df = pd.read_csv("rainfall.csv")
df = df.dropna()

X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

# Train model inside app
model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.header("Input Weather Parameters")

input_data = []
for col in X.columns:
    val = st.sidebar.number_input(col, float(X[col].mean()))
    input_data.append(val)

if st.button("Predict"):
    prediction = model.predict([input_data])[0]

    if prediction == 1:
        st.error("🌧 It WILL Rain Tomorrow")
    else:
        st.success("☀ It will NOT Rain Tomorrow")