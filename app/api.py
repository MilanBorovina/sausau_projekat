from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="BMW Sales Classification API",
    description="API for predicting BMW sales classification using a saved sklearn pipeline.",
    version="1.0",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "bmw_sales_model.joblib"
model = joblib.load(MODEL_PATH)


class BMWInput(BaseModel):
    Model: Literal["3 Series", "5 Series", "7 Series", "M3", "M5", "X1", "X3", "X5", "X6", "i3", "i8"]
    Year: int
    Region: Literal["Africa", "Asia", "Europe", "Middle East", "North America", "South America"]
    Color: Literal["Black", "Blue", "Grey", "Red", "Silver", "White"]
    Fuel_Type: Literal["Diesel", "Electric", "Hybrid", "Petrol"]
    Transmission: Literal["Automatic", "Manual"]
    Engine_Size_L: float
    Mileage_KM: float
    Price_USD: float


@app.get("/")
def home():
    return {"message": "BMW Sales Classification API is running."}


@app.post("/predict")
def predict(car: BMWInput):
    input_data = pd.DataFrame([car.dict()])
    prediction = model.predict(input_data)[0]
    response = {"prediction": str(prediction)}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        response["probabilities"] = {
            str(class_label): round(float(probability), 3)
            for class_label, probability in zip(model.classes_, probabilities)
        }
    return response


# Pokretanje:
# uvicorn app.api:app --reload
# Dokumentacija:
# http://127.0.0.1:8000/docs
