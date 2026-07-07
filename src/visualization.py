import matplotlib.pyplot as plt
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    CORRELATION_NUMERIC_FEATURES,
    FIGURES_DIR,
    LEAKAGE_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def save_all_figures(dataset, feature_importance=None, confusion=None, model_name=None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_class_distribution(dataset)
    save_numeric_histograms(dataset)
    save_boxplots(dataset)
    save_correlation_heatmap(dataset)
    save_category_target_plots(dataset)
    save_sales_volume_leakage_plot(dataset)
    if feature_importance is not None:
        save_feature_importance(feature_importance)
    if confusion is not None and model_name is not None:
        save_confusion_matrix(confusion, model_name)


def save_class_distribution(dataset):
    counts = dataset[TARGET_COLUMN].value_counts()
    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar", color=["#2f6f73", "#d78c45"])
    plt.title("Raspodela ciljne promenljive")
    plt.xlabel("Sales Classification")
    plt.ylabel("Broj primera")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png")
    plt.close()


def save_numeric_histograms(dataset):
    for column in NUMERIC_FEATURES:
        plt.figure(figsize=(8, 5))
        dataset[column].hist(bins=20, color="#2f6f73")
        plt.title(f"Raspodela atributa {column}")
        plt.xlabel(column)
        plt.ylabel("Broj primera")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{column.lower()}_histogram.png")
        plt.close()


def save_boxplots(dataset):
    for column in ["Price_USD", "Mileage_KM", "Engine_Size_L"]:
        plt.figure(figsize=(8, 4))
        dataset.boxplot(column=column)
        plt.title(f"Boxplot za {column}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{column.lower()}_boxplot.png")
        plt.close()


def save_correlation_heatmap(dataset):
    corr = dataset[CORRELATION_NUMERIC_FEATURES].corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Korelacija")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    plt.title("Korelaciona matrica numerickih atributa")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close()


def save_category_target_plots(dataset):
    for column in CATEGORICAL_FEATURES:
        table = pd.crosstab(dataset[column], dataset[TARGET_COLUMN], normalize="index") * 100
        table = table[["High", "Low"]] if set(["High", "Low"]).issubset(table.columns) else table
        ax = table.plot(kind="bar", stacked=True, figsize=(9, 5), color=["#2f6f73", "#d78c45"])
        ax.set_title(f"{column} vs Sales_Classification")
        ax.set_xlabel(column)
        ax.set_ylabel("Procenat")
        ax.legend(title=TARGET_COLUMN)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{column.lower()}_vs_sales_classification.png")
        plt.close()


def save_sales_volume_leakage_plot(dataset):
    plt.figure(figsize=(8, 5))
    dataset.boxplot(column=LEAKAGE_COLUMN, by=TARGET_COLUMN)
    plt.axhline(7000, color="red", linestyle="--", label="Prag 7000")
    plt.title("Sales_Volume po klasama")
    plt.suptitle("")
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel(LEAKAGE_COLUMN)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sales_volume_leakage.png")
    plt.close()


def save_feature_importance(feature_importance):
    top = feature_importance.sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"], color="#2f6f73")
    plt.gca().invert_yaxis()
    plt.title("Najznacajniji atributi")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png")
    plt.close()


def save_confusion_matrix(confusion, model_name):
    plt.figure(figsize=(5, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.title(f"Matrica konfuzije - {model_name}")
    plt.xlabel("Predvidjena klasa")
    plt.ylabel("Stvarna klasa")
    labels = ["High", "Low"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, confusion[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    plt.close()
