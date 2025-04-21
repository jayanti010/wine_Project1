import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Title
st.title("Wine Quality Classification using KNN 🍷")

# Load data
@st.cache_data
def load_data():
    red = pd.read_csv("D:\\wine\\winequality-red.csv", sep=';')
    white = pd.read_csv("D:\\wine\\winequality-white.csv", sep=';')
    red['type'] = 'red'
    white['type'] = 'white'
    return pd.concat([red, white], ignore_index=True)

df = load_data()

# Filter wine type
wine_type = st.selectbox("Select wine type", ['red', 'white'])
filtered_df = df[df['type'] == wine_type]

# Display raw data
if st.checkbox("Show raw data"):
    st.write(filtered_df.head())

# Features and target
X = filtered_df.drop(columns=['quality', 'type'])
y = filtered_df['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K selection
k = st.slider("Select number of neighbors (K)", 1, 20, 5)

# Train model
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show results
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Try your own prediction
st.subheader("Try predicting a wine quality")
input_data = {}
for col in X.columns:
    input_data[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
input_df = pd.DataFrame([input_data])
predicted_quality = model.predict(input_df)[0]
st.success(f"Predicted Quality: {predicted_quality}")
