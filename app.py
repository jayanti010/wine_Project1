# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("🍷 Wine Quality Prediction using KNN")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/wineQualityReds.csv"
    return pd.read_csv(url)

df = load_data()
st.write("### Wine Dataset", df.head())

X = df.drop("quality", axis=1)
y = df["quality"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sidebar - K selection
k = st.sidebar.slider("Select value of K for KNN", 1, 20, 5)

# Model training
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.sidebar.metric("Model Accuracy", f"{accuracy:.2f}")

st.write("## Predict Your Own Wine Quality")

# User input
input_data = {}
for feature in df.columns[:-1]:
    input_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Wine Quality: **{prediction[0]}**")
