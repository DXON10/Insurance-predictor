import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# Train model directly from CSV
df = pd.read_csv('insurance.csv')
df_model = df.copy()
df_model['sex'] = LabelEncoder().fit_transform(df_model['sex'])
df_model['smoker'] = LabelEncoder().fit_transform(df_model['smoker'])
df_model = pd.get_dummies(df_model, columns=['region'], drop_first=True)

X = df_model.drop('charges', axis=1)
y = df_model['charges']

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter your details below to predict your annual insurance cost.")

age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ["Female", "Male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker?", ["No", "Yes"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

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
