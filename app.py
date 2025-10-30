# app.py

import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load model (or retrain if needed)
@st.cache_resource
def load_model():
    df = pd.read_csv('learning_behavior_data.csv')
    X = df.drop('Learning_Effect', axis=1)
    y = df['Learning_Effect']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

st.title("📊 Learning Effect Predictor")
st.write("Upload a CSV file with 25 features (Likert scale 1–5) to predict learning effect.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    if input_df.shape[1] != 25:
        st.error("CSV must have exactly 25 features.")
    else:
        predictions = model.predict(input_df)
        st.write("### Predictions:")
        st.write(predictions)