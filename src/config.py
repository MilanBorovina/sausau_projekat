from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "BMW_Car_Sales_Classification.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

MODEL_PATH = MODEL_DIR / "bmw_sales_model.joblib"

TARGET_COLUMN = "Sales_Classification"
LEAKAGE_COLUMN = "Sales_Volume"

CATEGORICAL_FEATURES = ["Model", "Region", "Color", "Fuel_Type", "Transmission"]
NUMERIC_FEATURES = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
CORRELATION_NUMERIC_FEATURES = NUMERIC_FEATURES + [LEAKAGE_COLUMN]

RANDOM_STATE = 42
