import streamlit as st
import pandas as pd
import joblib


# LOAD PICKLE FILE

model_data = joblib.load("final_crop_yield_model.pkl")
model = model_data["model"]
columns = model_data["columns"]

st.title("🌾 Crop Yield Prediction App")

st.write("Enter the crop details below to predict yield")


# INPUT FIELDS (NUMERIC)

soil_ph = st.number_input("Soil pH", 3.0, 9.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 3000.0, 500.0)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
land_area = st.number_input("Land Area (acres)", 0.1, 50.0, 5.0)
fertilizer_qty = st.number_input("Fertilizer Quantity (kg)", 0.0, 500.0, 50.0)
pesticide = st.number_input("Pesticide Used (liters)", 0.0, 50.0, 2.0)


# CREATE INPUT DATAFRAME

input_data = {
    "Soil_pH": soil_ph,
    "Rainfall_mm": rainfall,
    "Temperature_C": temperature,
    "Humidity_Percent": humidity,
    "Land_Area_Acres": land_area,
    "Fertilizer_Quantity_kg": fertilizer_qty,
    "Pesticide_Used_Liters": pesticide
}

input_df = pd.DataFrame([input_data])

# Match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# PREDICT BUTTON

if st.button("Predict Yield"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌾 Predicted Crop Yield: {prediction:.2f} tonnes")
