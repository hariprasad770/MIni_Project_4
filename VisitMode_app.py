import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load sample data for dropdown options
df = pd.read_csv('C:/Users/DEEPADHARSHINI/OneDrive/Desktop/Tourism/MergedTourismData.csv')

# Load pre-trained model and encoders
model = joblib.load("visit_xgb.pkl")
ohe = joblib.load("visit_ohe.pkl")
te = joblib.load("visit_te.pkl")
scaler = joblib.load("visit_scaler.pkl")
label_encoder = joblib.load("visit_label.pkl")

# Define user inputs
def user_input_features():
    st.title("🚦 Visit Mode Prediction App")

    VisitYear = st.number_input("📅 Visit Year", min_value=2000, max_value=2030, value=2023)
    VisitMonth = st.number_input("📆 Visit Month", min_value=1, max_value=12, value=5)

    AttractionId = st.number_input("🏛️ Attraction ID", min_value=1, value=100)
    ContinentId = st.selectbox("🌍 Continent ID", sorted(df["ContinentId"].dropna().unique()))
    RegionId = st.number_input("📍 Region ID", min_value=1, value=10)

    Attraction = st.selectbox("🏞️ Attraction", sorted(df["Attraction"].dropna().unique()))
    AttractionType = st.selectbox("🎡 Attraction Type", sorted(df["AttractionType"].dropna().unique()))
    AttractionTypeId = st.number_input("🎟️ Attraction Type ID", min_value=1, value=1)
    Rating = st.slider("⭐ Rating", 1.0, 5.0, step=0.1, value=4.0)

    data = {
        'VisitYear': [VisitYear],
        'VisitMonth': [VisitMonth],
        'AttractionId': [AttractionId],
        'ContinentId': [ContinentId],
        'RegionId': [RegionId],
        'Attraction': [Attraction],
        'AttractionType': [AttractionType],
        'AttractionTypeId': [AttractionTypeId],
        'Rating': [Rating]
    }
    return pd.DataFrame(data)

# Get input from user
input_df = user_input_features()

# Prediction logic
if st.button("🔍 Predict Visit Mode"):
    try:
        # --- Target encode 'Attraction'
        input_df["Attraction"] = te.transform(input_df[["Attraction"]])

        # --- One-hot encode 'AttractionType'
        ohe_encoded = ohe.transform(input_df[["AttractionType"]])
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(["AttractionType"]))

        # --- Drop original 'AttractionType'
        input_df = input_df.drop(columns=["AttractionType"])

        # --- Combine numerical and encoded features
        final_input = pd.concat([input_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

        # --- Apply scaler
        final_input_scaled = scaler.transform(final_input)

        # --- Predict
        prediction = model.predict(final_input_scaled)
        result = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🚗 Predicted Visit Mode: **{result}**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
