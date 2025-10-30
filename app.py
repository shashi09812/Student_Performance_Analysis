import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(page_title="Learning Effect Predictor", layout="centered")

# Load and train model
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("learning_behavior_data.csv")
    except FileNotFoundError:
        st.error("❌ Dataset file 'learning_behavior_data.csv' not found in your repo.")
        return None

    # Try to detect target column
    possible_targets = ['Learning_Effect', 'learning_effect', 'Effect', 'Score']
    target_col = next((col for col in df.columns if col in possible_targets), None)

    if not target_col:
        st.error("❌ Target column not found. Expected one of: " + ", ".join(possible_targets))
        return None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if X.shape[1] != 25:
        st.warning(f"⚠️ Training data has {X.shape[1]} features. Expected 25 for prediction consistency.")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

# Load model and feature names
result = load_model()
if result:
    model, feature_names = result

    st.title("📊 Learning Effect Predictor")
    st.write("Upload a CSV file with **exactly 25 features** (Likert scale 1–5) to predict learning effect.")

    uploaded_file = st.file_uploader("📁 Choose a CSV file", type="csv")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)

        # Validate feature count and names
        if input_df.shape[1] != 25:
            st.error(f"❌ Uploaded file has {input_df.shape[1]} columns. Expected 25.")
        elif set(input_df.columns) != set(feature_names):
            st.warning("⚠️ Column names do not match training data. Predictions may be inaccurate.")
            st.write("Expected columns:", feature_names)
            st.write("Uploaded columns:", input_df.columns.tolist())
        else:
            predictions = model.predict(input_df)
            st.success("✅ Predictions generated successfully!")
            st.dataframe(pd.DataFrame(predictions, columns=["Predicted Learning Effect"]))
else:
    st.stop()
