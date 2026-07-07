import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_high": precision_score(y_test, y_pred, pos_label="High", zero_division=0),
        "recall_high": recall_score(y_test, y_pred, pos_label="High", zero_division=0),
        "f1_high": f1_score(y_test, y_pred, pos_label="High", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)
        high_index = list(model.classes_).index("High")
        metrics["roc_auc"] = roc_auc_score((y_test == "High").astype(int), y_prob[:, high_index])
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        metrics["roc_auc"] = roc_auc_score((y_test == "High").astype(int), scores)
    else:
        metrics["roc_auc"] = None

    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["High", "Low"])
    return metrics, report, cm


def save_classification_report(path, model_name, report):
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Classification report - {model_name}\n\n")
        file.write(report)


def metrics_to_dataframe(metrics_list):
    return pd.DataFrame(metrics_list)
