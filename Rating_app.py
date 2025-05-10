import streamlit as st
import pandas as pd
import numpy as np
import joblib

df=pd.read_csv('C:/Users/DEEPADHARSHINI/OneDrive/Desktop/Tourism/MergedTourismData.csv')

# Load model and preprocessing objects
model = joblib.load("xgboost_model.pkl")
ohe = joblib.load("ohe_encoder.pkl")  # OneHotEncoder
te = joblib.load("target_encoder.pkl")  # TargetEncoder
scaler = joblib.load("scaler.pkl")  # StandardScaler or similar

# Define the features used in encoding
en_features = ['VisitModeName', 'AttractionType']

# Collect input from user
def user_input_features():
    st.title("🎯 Tourism Rating Prediction App")

    VisitYear = st.number_input("📅 Visit Year", min_value=2000, max_value=2025, value=2023)
    VisitMonth = st.number_input("📆 Visit Month", min_value=1, max_value=12, value=5)

    VisitModes = df["VisitModeName"].dropna().unique().tolist()
    VisitMode = st.selectbox("🚗 Visit Mode", sorted(VisitModes))

    Attractions = df["Attraction"].dropna().unique().tolist()
    Attraction = st.selectbox("🏛️ Attraction", sorted(Attractions))

    AttractionTypes = df["AttractionType"].dropna().unique().tolist()
    AttractionType = st.selectbox("🎡 Attraction Type", sorted(AttractionTypes))

    CountryId = st.number_input("🌍 Country ID", min_value=1, value=1)
    RegionId = st.number_input("📍 Region ID", min_value=1, value=1)

    # Return input as DataFrame
    data = {
        'VisitYear': [VisitYear],
        'VisitMonth': [VisitMonth],
        'VisitModeName': [VisitMode],
        'Attraction': [Attraction],
        'AttractionType': [AttractionType],
        'CountryId': [CountryId],
        'RegionId': [RegionId],
    }
    return pd.DataFrame(data)

# Get input
input_df = user_input_features()

# Predict when button is clicked
if st.button("🔍 Predict Rating"):
    try:
        # One-hot encode VisitMode and AttractionType
        ohe_encoded = ohe.transform(input_df[en_features])
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(en_features))

        # Target encode Attraction
        input_df['Attraction'] = te.transform(input_df['Attraction'])

        # Drop original categorical columns
        input_df.drop(columns=en_features, inplace=True)

        # Combine with encoded columns
        final_input = pd.concat([input_df, ohe_df], axis=1)

        # Scale the data
        final_input_scaled = scaler.transform(final_input)

        # Predict using model
        prediction = model.predict(final_input_scaled)

        # Show result
        st.success(f"🌟 Predicted Rating: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
