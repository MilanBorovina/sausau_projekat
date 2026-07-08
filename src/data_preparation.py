import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    LEAKAGE_COLUMN,
    MODEL_FEATURES,
    TARGET_COLUMN,
)

# ucitava dataset iz csv fajla
def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


def clean_dataset(dataset):
    dataset = dataset.copy()    # kopra dataset
    dataset = dataset.drop_duplicates() # brise duplikate
    dataset = dataset.dropna()  # brise redove sa nedostajucim vrednostima
    return dataset


def print_data_quality(dataset):

    # pokazuje prvih 5 redova da vidim izgled tabele
    print("Prvih nekoliko redova")
    print(dataset.head())

    # ispisuje br redova i kolona (50000, 11)
    print("\nDimenzije skupa")
    print(dataset.shape)

    # koliko nedostajucih vrednosti u svakoj koloni
    print("\nNedostajuce vrednosti")
    print(dataset.isna().sum())

    print("\nBroj duplikata")
    print(dataset.duplicated().sum())

    # koliko ima primera svake klase
    print("\nRaspodela ciljnih klasa")
    print(dataset[TARGET_COLUMN].value_counts())


# razdvaja dataset na ulazne podatke i ciljnu kolonu
def split_features_target(dataset):
    x = dataset[MODEL_FEATURES]
    y = dataset[TARGET_COLUMN]
    return x, y

# provera da li Sales_Volume direktno utice na Sales_Classification
def analyze_sales_volume_leakage(dataset):
    result = {}
    # grupise dataset po ciljnoj klasi (High, Low)
    for class_name, group in dataset.groupby(TARGET_COLUMN):
        result[class_name] = {
            "count": len(group),
            "min_sales_volume": group[LEAKAGE_COLUMN].min(),
            "max_sales_volume": group[LEAKAGE_COLUMN].max(),
            "mean_sales_volume": group[LEAKAGE_COLUMN].mean(),
        }

    # pravilo za predikciju na osnovu Sales_Volume, ako je >= 7000 onda High
    rule_prediction = dataset[LEAKAGE_COLUMN].apply(lambda value: "High" if value >= 7000 else "Low")
    # uporedjuje pravilo sa pravim klasam i racuna accuracy (bice 1)
    accuracy = (rule_prediction == dataset[TARGET_COLUMN]).mean()

    # dodaje pravilo i accuracy u tabelu
    leakage_table = pd.DataFrame(result).T
    leakage_table.loc["threshold_rule", "rule"] = "High if Sales_Volume >= 7000 else Low"
    leakage_table.loc["threshold_rule", "accuracy"] = accuracy
    return leakage_table

# racun  koliko se procentualno svaka kategorija atributa pojavljuje u High i Low klasama
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

# anomalije u datasetu, da li su vrednosti van opsega
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
