import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    LEAKAGE_COLUMN,
    MODEL_FEATURES,
    TARGET_COLUMN,
)


def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


def clean_dataset(dataset):
    dataset = dataset.copy()
    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()
    return dataset


def print_data_quality(dataset):
    print("Prvih nekoliko redova")
    print(dataset.head())

    print("\nDimenzije skupa")
    print(dataset.shape)

    print("\nNedostajuce vrednosti")
    print(dataset.isna().sum())

    print("\nBroj duplikata")
    print(dataset.duplicated().sum())

    print("\nRaspodela ciljnih klasa")
    print(dataset[TARGET_COLUMN].value_counts())


def split_features_target(dataset):
    x = dataset[MODEL_FEATURES]
    y = dataset[TARGET_COLUMN]
    return x, y


def analyze_sales_volume_leakage(dataset):
    result = {}
    for class_name, group in dataset.groupby(TARGET_COLUMN):
        result[class_name] = {
            "count": len(group),
            "min_sales_volume": group[LEAKAGE_COLUMN].min(),
            "max_sales_volume": group[LEAKAGE_COLUMN].max(),
            "mean_sales_volume": group[LEAKAGE_COLUMN].mean(),
        }

    rule_prediction = dataset[LEAKAGE_COLUMN].apply(lambda value: "High" if value >= 7000 else "Low")
    accuracy = (rule_prediction == dataset[TARGET_COLUMN]).mean()

    leakage_table = pd.DataFrame(result).T
    leakage_table.loc["threshold_rule", "rule"] = "High if Sales_Volume >= 7000 else Low"
    leakage_table.loc["threshold_rule", "accuracy"] = accuracy
    return leakage_table


def category_class_percentages(dataset):
    rows = []
    for column in CATEGORICAL_FEATURES:
        table = pd.crosstab(dataset[column], dataset[TARGET_COLUMN])
        for value, row in table.iterrows():
            high_count = row.get("High", 0)
            low_count = row.get("Low", 0)
            total = high_count + low_count
            rows.append({
                "Attribute": column,
                "Value": value,
                "Total": total,
                "High count": high_count,
                "Low count": low_count,
                "High percent": round(high_count / total * 100, 2),
                "Low percent": round(low_count / total * 100, 2),
            })
    return pd.DataFrame(rows)


def anomaly_table(dataset):
    checks = [
        ("Year van opsega 1980-2026", (dataset["Year"] < 1980) | (dataset["Year"] > 2026)),
        ("Engine_Size_L van opsega 0-8", (dataset["Engine_Size_L"] <= 0) | (dataset["Engine_Size_L"] > 8)),
        ("Mileage_KM negativna ili preko 500000", (dataset["Mileage_KM"] < 0) | (dataset["Mileage_KM"] > 500000)),
        ("Price_USD negativna ili preko 250000", (dataset["Price_USD"] <= 0) | (dataset["Price_USD"] > 250000)),
        ("Sales_Volume negativan", dataset["Sales_Volume"] < 0),
    ]
    return pd.DataFrame([
        {"Anomaly check": name, "Number of rows": int(mask.sum())}
        for name, mask in checks
    ])
