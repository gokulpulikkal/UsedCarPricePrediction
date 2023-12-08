import pandas as pd
import numpy as np
import pickle
import streamlit as st
import options

# Load the pickled model
model_file = "ModelForTestingFromNet.pkl"
with open(model_file, "rb") as f:
    model = pickle.load(f)

# Create a function to predict the price
def predict_price(year, cylinder, odometer, age):
    company = "Toyota"
    car_model = "Corolla"
    year = 2020
    fuel_type = "Petrol"
    driven = 50000

    # Create a pandas DataFrame with the dummy input
    dummy_input = pd.DataFrame({
        "name": [car_model],
        "company": [company],
        "year": [year],
        "kms_driven": [driven],
        "fuel_type": [fuel_type],
    })
    # Make a prediction using the model
    prediction = model.predict(dummy_input)
    return prediction

# Create a Streamlit app
st.title("Used Car Price Prediction Model")

# Collect user input
year = st.number_input("Year", min_value=1980, max_value=2023)
cylinder = st.number_input("Number of cylinders", min_value=3, max_value=8)
odometer = st.number_input("Odometer reading (miles)", min_value=0)
age = st.number_input("Age of car (years)", min_value=0)
manufacturer = st.selectbox("Please select Manufacturer", index=None,options=options.manufacturer_options, placeholder="Select Manufacturer")
region = st.selectbox("Please select Region",options=options.regions_options, index=None, placeholder="Select Region")
# Make the default condition as uncharted
condition = st.selectbox("Please select Condition of your vehicle",options=options.vehicle_condition_options, index=None, placeholder="Vehicle condition")
fuel = st.selectbox("Please select fuel type of your vehicle",options=options.fuel_options, index=None, placeholder="Fuel type")
title_status = st.selectbox("Please select title status of your vehicle",options=options.title_status_options, index=None, placeholder="Title Status")
transmission = st.selectbox("Please select transmission type of your vehicle",options=options.transmission_options, index=None, placeholder="Transmission")
# Make the default condition as uncharted
vehicle_size = st.selectbox("Please select size type of your vehicle",options=options.vehicle_sizes, index=None, placeholder="Vehicle size")
vehicle_color = st.selectbox("Please select color of your vehicle",options=options.colors, index=None, placeholder="Vehicle color")
months_ago_number = st.number_input("When many months ago this got posted", min_value=0)
# Select car model
selected_model = st.selectbox("Choose car model", options.models, index=None, placeholder="Vehicle Model")
# Select car drive
selected_model = st.selectbox("Choose car drive type", options.drive_options, index=None, placeholder="Vehicle Drive type")
# Select car type
selected_type = st.selectbox("Choose car type", options.type, index=None, placeholder="Vehicle type")
# Select state
selected_state = st.selectbox("Choose state", options.state, index=None, placeholder="Select State")
description = st.text_area("Describe your vehicle condition")

# Make the prediction
if st.button("Predict Price"):
    predicted_price = predict_price(year, cylinder, odometer, age)
    st.write(f"Predicted price: ${predicted_price:.2f}")

# Display information about the model
st.markdown("**Model details:**")
st.write(f"- Programming language: Python")
st.write(f"- Model format: Pickle")
st.write(f"- Input features: year, cylinder, odometer, age")
