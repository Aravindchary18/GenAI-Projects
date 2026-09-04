import streamlit as st 
import joblib 
import numpy as np 
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "house_model.pkl"))

st.title("🏡House Price Prediction🏘️")

Area=st.number_input("Area(GiLivArea)",min_value=0,max_value=10000)
Bedroom=st.number_input("Bedrooms(BedroomAbrGr)",min_value=1,max_value=7,step=1)
Year=st.number_input("Year Built",min_value=1000,max_value=2025)
Quality=st.number_input("Overall Quality(1-10)",min_value=1,max_value=10)
Bathrooms=st.number_input("Full Bathrooms",min_value=0,max_value=10)
Garage=st.number_input("Garage Cars",min_value=0,max_value=10)

if st.button("Predict"):
    input_data=np.array([[Area,Bedroom,Year,Quality,Bathrooms,Garage]])
    prediction=model.predict(input_data)
    st.success(f"Predicted Price:{prediction[0]:.2f}")


