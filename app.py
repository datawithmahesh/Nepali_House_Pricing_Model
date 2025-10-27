# Nepali Housing Price Prediction 

import streamlit as st
import pandas as pd
import pickle as pk

# Page Config
st.set_page_config(page_title="Nepali Housing Price Prediction", page_icon="🏠", layout="wide")

# Load Models
model = pk.load(open('linear_model.pickle','rb'))
rf_model = pk.load(open('RandomForest_model.pickle','rb'))
gb_model = pk.load(open('GradientBoosting_model.pickle','rb'))

with open('scaler.pickle', 'rb') as f:
    scaler = pk.load(f)

# Header Section
st.markdown("""
    <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px'>
        <h1 style='color: white; text-align: center;'>🏠 Nepali Housing Price Prediction System</h1>
        <p style='color: white; text-align: center;'>Predict house prices based on selected features</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input Section
st.subheader("Enter House Details")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        floor = st.number_input("Number of Floors", 0, 15, 1)
        bedroom = st.number_input("Number of Bedrooms", 0, 50, 1)
        bathroom = st.number_input("Number of Bathrooms", 0, 20, 1)
    
    with col2:
        land = st.number_input("Land Area (aana)", 0, 50, 5)
        road = st.number_input("Road Access (feet)", 0, 200, 10)
        age = st.number_input("Age of the Building", 0, 50, 1)
    
    with col3:
        parking = st.number_input("Car Parking", 0, 15, 0)
        district = st.selectbox("District", ["Bhaktapur", "Chitwan", "Kaski", "Kathmandu", "Lalitpur"])

# One-hot encoding for district
bhaktapur = 1 if district == "Bhaktapur" else 0
chitwan = 1 if district == "Chitwan" else 0
kaski = 1 if district == "Kaski" else 0
kathmandu = 1 if district == "Kathmandu" else 0
lalitpur = 1 if district == "Lalitpur" else 0

# Create input DataFrame
input_df = pd.DataFrame({
    'FLOOR':[floor],
    'BEDROOM':[bedroom],
    'BATHROOM':[bathroom],
    'Land_in_aana':[land],
    'road_access_in_feet':[road],
    'AGE':[age],
    'car_parking':[parking],
    'bhaktapur':[bhaktapur],
    'chitwan':[chitwan],
    'kaski':[kaski],
    'kathmandu':[kathmandu],
    'lalitpur':[lalitpur]
})

# ----------------------------
# Scale numeric features
# ----------------------------
numeric_cols = ['FLOOR','BEDROOM','BATHROOM','Land_in_aana','road_access_in_feet','AGE','car_parking']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Show input data
st.subheader("Input Data Preview")
st.dataframe(input_df.style.set_properties(**{'background-color': '#f0f0f0', 'color': 'black', 'border-color': 'black'}))

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
if st.button("Predict Price"):
    # Predictions from all models
    linear_pred = float(model.predict(input_df)[0])
    rf_pred = float(rf_model.predict(input_df)[0])
    gb_pred = float(gb_model.predict(input_df)[0])

    # Result Section
    st.markdown("""
        <div style='background-color: #FFEB3B; padding: 20px; border-radius: 10px'>
            <h2 style='text-align: center;'>💰 Predicted House Prices (in Crore)</h2>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"{linear_pred:.2f} Cr")
    col2.metric("Random Forest", f"{rf_pred:.2f} Cr")
    col3.metric("Gradient Boosting", f"{gb_pred:.2f} Cr")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This model is developed by Mahesh Thapa</p>", unsafe_allow_html=True)
