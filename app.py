import streamlit as st
import pickle
import numpy as np

# Load your model file
with open('boston_housing_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🏡 Boston House Price Predictor")
st.write("Enter the property features below to predict the median house value.")

col1, col2 = st.columns(2)

with col1:
    crim = st.number_input("Per capita crime rate (CRIM)", value=0.1)
    zn = st.number_input("Residential land zoned for large lots (ZN)", value=0.0)
    indus = st.number_input("Proportion of non-retail business acres (INDUS)", value=5.0)
    chas = st.selectbox("Charles River adjacency (CHAS)", [0, 1])
    nox = st.number_input("Nitric oxides concentration (NOX)", value=0.5)
    rm = st.number_input("Average number of rooms per dwelling (RM)", value=6.5)
    age = st.number_input("Proportion of owner-occupied units built prior to 1940 (AGE)", value=50.0)

with col2:
    dis = st.number_input("Weighted distances to employment centres (DIS)", value=4.0)
    rad = st.number_input("Index of accessibility to radial highways (RAD)", value=4.0)
    tax = st.number_input("Full-value property-tax rate per $10,000 (TAX)", value=300.0)
    ptratio = st.number_input("Pupil-teacher ratio by town (PTRATIO)", value=15.0)
    b = st.number_input("Black population proportion index (B)", value=390.0)
    lstat = st.number_input("% lower status of the population (LSTAT)", value=10.0)

if st.button("Predict House Price"):
    input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    prediction = model.predict(input_data)
    predicted_price = prediction[0] * 1000
    st.success(f"Estimated House Price: ${predicted_price:,.2f}")