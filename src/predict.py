from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "bmw_sales_model.joblib"


model = joblib.load(MODEL_PATH)

new_car = pd.DataFrame([
    {
        "Model": "X5",
        "Year": 2021,
        "Region": "Europe",
        "Color": "Black",
        "Fuel_Type": "Diesel",
        "Transmission": "Automatic",
        "Engine_Size_L": 3.0,
        "Mileage_KM": 45000,
        "Price_USD": 65000,
    }
])

prediction = model.predict(new_car)[0]

print("Ulazni podaci")
print(new_car)
print("\nPredikcija modela")
print(prediction)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(new_car)[0]
    classes = model.classes_
    print("\nVerovatnoce po klasama")
    for class_label, probability in zip(classes, probabilities):
        print(f"{class_label}: {probability:.3f}")
