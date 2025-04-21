import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wine Quality Classifier", layout="centered")

st.title("🍷 Wine Quality Classifier with KNN")

@st.cache_data
def load_data():
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    red = pd.read_csv(red_url, sep=';')
    white = pd.read_csv(white_url, sep=';')
    red['type'] = 'red'
    white['type'] = 'white'
    return pd.concat([red, white], ignore_index=True)

df = load_data()

st.subheader("📊 Raw Data")
if st.checkbox("Show dataset"):
    st.dataframe(df.head())

# Choose wine type
wine_type = st.selectbox("Select Wine Type", ["red", "white"])
filtered_df = df[df['type'] == wine_type]

# Prepare features and target
X = filtered_df.drop(columns=["quality", "type"])
y = filtered_df["quality"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
test_size = st.slider("Test set size (%)", 10, 50, 30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size / 100, random_state=42)

# KNN Classifier
k = st.slider("Choose value for K (neighbors)", 1, 15, 5)
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Performance")
st.write(f"Accuracy: **{accuracy:.2f}**")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("📉 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
st.pyplot(fig)
