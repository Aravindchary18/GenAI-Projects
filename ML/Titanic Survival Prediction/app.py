import streamlit as st
import joblib 
import numpy as np

model = joblib.load(r"C:\Users\Aravind Nannabattuni\OneDrive\Desktop\aiml_projects\Titanic Survival Prediction\model.pkl")

st.title("Titanic Survival Prediction")

Pclass=st.selectbox("passenger class",[1,2,3])
Sex=st.selectbox("Sex",["Male","Female"])
Age=st.slider("Age",0,100,20)
Fare=st.number_input("Fare",value=50)

Sex = 0 if Sex=="Male" else 1

if st.button("predict"):
   input_data=np.array([[Pclass,Sex,Age,Fare]])
   prediction=model.predict(input_data)
   if prediction[0]==1:
      st.success("Survived")
   else:
      st.error("Did Not Survive")
      