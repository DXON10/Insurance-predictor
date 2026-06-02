import streamlit as st
import pickle
import numpy as np

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter your details below to predict your annual insurance cost.")

# User inputs
age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ["Female", "Male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker?", ["No", "Yes"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Encode inputs
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0
region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0

if st.button("Predict Insurance Cost"):
    features = np.array([[age, sex_val, bmi, children, smoker_val,
                          region_northwest, region_southeast, region_southwest]])
    prediction = model.predict(features)[0]
    st.success(f"💰 Estimated Annual Insurance Cost: **${prediction:,.2f}**")