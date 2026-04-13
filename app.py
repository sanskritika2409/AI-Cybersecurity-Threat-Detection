import streamlit as st
import pandas as pd
import joblib
from src.preprocessing import preprocess_data

st.title("🛡️ AI Cybersecurity Threat Detection System")

# Load model
model = joblib.load("models/cyber_model.pkl")

# Upload file
uploaded_file = st.file_uploader("Upload Network Data CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("📊 Uploaded Data:")
    st.write(data.head())

    if st.button("Detect Threats"):
        # 🔥 IMPORTANT FIX
        X, _ = preprocess_data(data)

        predictions = model.predict(X)

        results = ["⚠️ Threat" if p == 1 else "✅ Normal" for p in predictions]

        data["Prediction"] = results

        st.write("🔍 Results:")
        st.write(data)

        st.write(f"🚨 Threats Detected: {results.count('⚠️ Threat')}")
        st.write(f"✅ Normal Traffic: {results.count('✅ Normal')}")