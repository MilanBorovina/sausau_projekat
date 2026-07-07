from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="BMW Sales Prediction", layout="centered")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "bmw_sales_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("BMW Sales Prediction")
st.write("Predikcija klase prodaje BMW automobila pomocu klasicnog sklearn modela.")

col1, col2 = st.columns(2)

with col1:
    car_model = st.selectbox("Model", ["3 Series", "5 Series", "7 Series", "M3", "M5", "X1", "X3", "X5", "X6", "i3", "i8"])
    year = st.number_input("Year", min_value=2010, max_value=2026, value=2021)
    region = st.selectbox("Region", ["Africa", "Asia", "Europe", "Middle East", "North America", "South America"])
    color = st.selectbox("Color", ["Black", "Blue", "Grey", "Red", "Silver", "White"])
    fuel_type = st.selectbox("Fuel type", ["Diesel", "Electric", "Hybrid", "Petrol"])

with col2:
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    engine_size = st.number_input("Engine size L", min_value=1.0, max_value=5.0, value=3.0)
    mileage = st.number_input("Mileage KM", min_value=0, max_value=300000, value=45000)
    price = st.number_input("Price USD", min_value=10000, max_value=150000, value=65000)

if st.button("Predict"):
    input_data = pd.DataFrame([
        {
            "Model": car_model,
            "Year": year,
            "Region": region,
            "Color": color,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Engine_Size_L": engine_size,
            "Mileage_KM": mileage,
            "Price_USD": price,
        }
    ])

    prediction = model.predict(input_data)[0]
    st.subheader("Result")
    st.dataframe(input_data)
    st.success(f"Prediction: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = pd.DataFrame({
            "class": model.classes_,
            "probability": model.predict_proba(input_data)[0],
        })
        st.write("Prediction probabilities")
        st.dataframe(probabilities)
