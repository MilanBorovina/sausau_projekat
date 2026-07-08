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
        "precision_low": precision_score(y_test, y_pred, pos_label="Low", zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_high": recall_score(y_test, y_pred, pos_label="High", zero_division=0),
        "recall_low": recall_score(y_test, y_pred, pos_label="Low", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_high": f1_score(y_test, y_pred, pos_label="High", zero_division=0),
        "f1_low": f1_score(y_test, y_pred, pos_label="Low", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "support_high": int((y_test == "High").sum()),
        "support_low": int((y_test == "Low").sum()),
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


def save_classification_report(path, model_name, report, metrics=None, confusion=None):
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Classification report - {model_name}\n\n")

        if metrics is not None:
            file.write("Pregled po klasama\n")
            file.write("-----------------\n")
            file.write(
                "High: "
                f"precision={metrics['precision_high']:.4f}, "
                f"recall={metrics['recall_high']:.4f}, "
                f"f1={metrics['f1_high']:.4f}, "
                f"support={metrics['support_high']}\n"
            )
            file.write(
                "Low:  "
                f"precision={metrics['precision_low']:.4f}, "
                f"recall={metrics['recall_low']:.4f}, "
                f"f1={metrics['f1_low']:.4f}, "
                f"support={metrics['support_low']}\n\n"
            )

            file.write("Zajednicke metrike\n")
            file.write("-----------------\n")
            file.write(f"accuracy={metrics['accuracy']:.4f}\n")
            file.write(
                "macro avg:    "
                f"precision={metrics['precision_macro']:.4f}, "
                f"recall={metrics['recall_macro']:.4f}, "
                f"f1={metrics['f1_macro']:.4f}\n"
            )
            file.write(
                "weighted avg: "
                f"precision={metrics['precision_weighted']:.4f}, "
                f"recall={metrics['recall_weighted']:.4f}, "
                f"f1={metrics['f1_weighted']:.4f}\n"
            )
            if metrics["roc_auc"] is not None:
                file.write(f"roc_auc={metrics['roc_auc']:.4f}\n")

            if confusion is not None:
                file.write("\nMatrica konfuzije [High, Low]\n")
                file.write("-----------------------------\n")
                file.write(str(confusion))
                file.write("\n")

        else:
            file.write(report)


def metrics_to_dataframe(metrics_list):
    return pd.DataFrame(metrics_list)
