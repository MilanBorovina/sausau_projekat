import pandas as pd

from sklearn.model_selection import train_test_split

#uzima putanje i konstante iz config.py
from config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
)
#importovanje funkcija za pripremu podataka iz data_preparation.py
from data_preparation import (
    analyze_sales_volume_leakage,
    anomaly_table,
    category_class_percentages,
    clean_dataset,
    load_dataset,
    print_data_quality,
    split_features_target,
)
#importovanje funkcija za ocenu modela
from evaluation import evaluate_model, metrics_to_dataframe, save_classification_report
#importovanje modela
from modeling import (
    build_models,
    cross_validate_models,
    get_feature_importance,
    get_important_original_features,
    save_model,
)
#importovanje funkcija za grafove
from visualization import save_all_figures


def main():

#kreiranje foldera za rezultate, metrike i grafove
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    print_data_quality(dataset)
    dataset = clean_dataset(dataset)

#provera da li Sales_Volume direktno utice na Sales_Classification
    leakage = analyze_sales_volume_leakage(dataset)
    leakage.to_csv(METRICS_DIR / "sales_volume_leakage_analysis.csv")
    print("\nSales_Volume leakage analiza")
    print(leakage)

#procenti po kategorijama (high low)
    category_percentages = category_class_percentages(dataset)
    category_percentages.to_csv(METRICS_DIR / "category_class_percentages.csv", index=False)

    anomalies = anomaly_table(dataset)
    anomalies.to_csv(METRICS_DIR / "anomalies.csv", index=False)

#korelaciona metrika, koliko su numericki atributi povezani sa Sales_Volume
    dataset[["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "Sales_Volume"]].corr().to_csv(
        METRICS_DIR / "correlation_matrix.csv"
    )
#generise slike u results/figures
    save_all_figures(dataset)

#podla dataseta na ulazne kolone i ciljnu kolonu
#x atributi pomocu kojih model predvidja, y ono sto model predvidja
    x, y = split_features_target(dataset)
    
    # Prva podela: train (70%) i temp (30%)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,      #podela ista pri svakom pokretanju
        stratify=y,                     #odnos high i low slican u trening i temp skupu
    )
    
    # Druga podela: temp se deli na validation (50%) i test (50%)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,      #podela ista pri svakom pokretanju
        stratify=y_temp,                #odnos high i low slican u validaciji i test skupu
    )

    print("\nDimenzije trening skupa")
    print(x_train.shape)
    print("\nDimenzije validacijskog skupa")
    print(x_val.shape)
    print("\nDimenzije test skupa")
    print(x_test.shape)

#pravljenje modela, poziva se funkcija buid_models iz modeling.py
    models = build_models()
#cross validation
    cv_results = cross_validate_models(models, x_train, y_train)
    cv_results.to_csv(METRICS_DIR / "cross_validation_results.csv", index=False)

    validation_metrics_rows = []
    validation_reports = {}
    validation_confusions = {}

#treniranje svih modela
    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)
    
    #trenira model na trening skupu
        model.fit(x_train, y_train)

        # Evaluacija na validacijskom skupu
        print("\nEVALUACIJA NA VALIDACIJSKOM SKUPU:")
        val_metrics, val_report, val_confusion = evaluate_model(model, x_val, y_val, model_name)
        validation_metrics_rows.append(val_metrics)
        validation_reports[model_name] = val_report
        validation_confusions[model_name] = val_confusion
        print(val_report)
        print("Matrica konfuzije (validacijski skup)")
        print(val_confusion)

    #validacione matrike se spajaju sa cross-validation metrikama i cuvaju u csv fajl
    validation_metrics_data = metrics_to_dataframe(validation_metrics_rows)
    validation_metrics_data = validation_metrics_data.merge(cv_results, on="model", how="left")
    validation_metrics_data.to_csv(METRICS_DIR / "validation_model_comparison.csv", index=False)
    validation_metrics_data.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    #izbor najboljeg modela na osnovu F1 macro metrike na validacionom skupu (racuna F1 zaobe klase pa uzima prosek)
    best_row = validation_metrics_data.sort_values(by="f1_macro", ascending=False).iloc[0]
    final_model_name = best_row["model"]
    tuned_model = build_models()[final_model_name]

    # DODATO: izbor najbitnijih atributa na osnovu modela treniranog samo na trening skupu
    selection_feature_importance = get_feature_importance(models[final_model_name])
    important_features = get_important_original_features(selection_feature_importance)

    # DODATO: poredi rezultate modela kada koriste sve atribute i kada koriste samo najbitnije atribute
    if important_features is not None and 0 < len(important_features) < x_train.shape[1]:
        pd.DataFrame({"important_feature": important_features}).to_csv(
            METRICS_DIR / "selected_important_features.csv",
            index=False,
        )

        important_categorical_features = [
            feature for feature in CATEGORICAL_FEATURES if feature in important_features
        ]
        important_numeric_features = [
            feature for feature in NUMERIC_FEATURES if feature in important_features
        ]
        important_models = build_models(
            categorical_features=important_categorical_features,
            numeric_features=important_numeric_features,
        )
        important_metrics_rows = []

        for model_name, model in important_models.items():
            model.fit(x_train[important_features], y_train)
            important_metrics, _, _ = evaluate_model(
                model,
                x_val[important_features],
                y_val,
                model_name,
            )
            important_metrics_rows.append(important_metrics)

        all_features_comparison = validation_metrics_data.copy()
        all_features_comparison["feature_set"] = "svi_atributi"

        important_features_comparison = metrics_to_dataframe(important_metrics_rows)
        important_features_comparison["feature_set"] = "najbitniji_atributi"

        all_vs_important_features = pd.concat(
            [all_features_comparison, important_features_comparison],
            ignore_index=True,
            sort=False,
        )
        comparison_columns = [
            "feature_set",
            "model",
            "accuracy",
            "precision_high",
            "recall_high",
            "f1_high",
            "precision_low",
            "recall_low",
            "f1_low",
            "f1_macro",
            "f1_weighted",
            "roc_auc",
        ]
        all_vs_important_features[comparison_columns].round(4).to_csv(
            METRICS_DIR / "all_vs_important_features_comparison.csv",
            index=False,
        )

    print("\n" + "="*60)
    print("IZABRANI MODEL - REZULTAT NA VALIDACIJSKOM SKUPU")
    print("="*60)
    print(final_model_name)
    print(validation_reports[final_model_name])
    print(validation_confusions[final_model_name])
    
    # Finalni model se trenira na train + validation skupu nakon izbora modela.
    x_train_final = pd.concat([x_train, x_val])
    y_train_final = pd.concat([y_train, y_val])
    tuned_model.fit(x_train_final, y_train_final)
    
    # Evaluacija finalnog modela na test skupu
    print("\n" + "="*60)
    print("FINALNI MODEL - EVALUACIJA NA TEST SKUPU")
    print("="*60)
    tuned_metrics, tuned_report, tuned_confusion = evaluate_model(tuned_model, x_test, y_test, final_model_name)

    print("\nFinalni model na osnovu validacionog F1 macro metrike")
    print(final_model_name)
    print(tuned_report)
    print(tuned_confusion)

    #cuvanje finalnih metrika
    tuned_metrics_data = metrics_to_dataframe([tuned_metrics])
    tuned_metrics_data.to_csv(METRICS_DIR / "final_model_metrics.csv", index=False)
    tuned_metrics_data.to_csv(METRICS_DIR / "test_model_comparison.csv", index=False)
    save_classification_report(
        METRICS_DIR / "final_classification_report.txt",
        final_model_name,
        tuned_report,
        tuned_metrics,
        tuned_confusion,
    )

    #cuva tabelu najvaznijih atributa ako postoji za dati model
    feature_importance = get_feature_importance(tuned_model)
    if feature_importance is not None:
        feature_importance.to_csv(METRICS_DIR / "feature_importance.csv", index=False)

    #ponovno generisanje grafika sa dodatnim grafovima za finalni model
    save_all_figures(
        dataset,
        feature_importance=feature_importance,
        confusion=tuned_confusion,
        model_name=final_model_name,
    )

    save_model(tuned_model)
    print("\nFinalni model:", final_model_name)
    print("Model je sacuvan u:", MODEL_PATH)
    print("Ciljna promenljiva:", TARGET_COLUMN)


if __name__ == "__main__":
    main()
