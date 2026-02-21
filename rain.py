import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧",
    layout="wide"
)




col1, col2 = st.columns(2)

with col1:
    humidity = st.number_input("Humidity")

with col2:
    wind_speed = st.number_input("Wind Speed")



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
        from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
import matplotlib.pyplot as plt

st.subheader("Feature Importance")

importance = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(X.columns, importance)
st.pyplot(fig)
prob = model.predict_proba([input_data])[0][1]
st.write(f"Rain Probability: {prob*100:.2f}%")