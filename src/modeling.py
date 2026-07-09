import joblib
import pandas as pd

# alati za preprocessing, modele i evaluaciju
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Učitavanje konfiguracije projekta: kolone za obradu, putanja modela i random state
from config import CATEGORICAL_FEATURES, MODEL_FEATURES, MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE

# Kreira OneHotEncoder za pretvaranje kategorijskih/tekstualnih kolona u numerički oblik
def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# kategorijske kolone se kodiraju OneHotEncoder-om
 # Ako model zahteva skaliranje numeričkih vrednosti, koristi StandardScaler
# DODATO: funkcija moze da primi i podskup atributa kada poredimo sve i najbitnije atribute
def build_preprocessor(scale_numeric=False, categorical_features=None, numeric_features=None):
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), categorical_features),
            ("numeric", numeric_step, numeric_features),
        ]
    )


# DODATO: modeli mogu da se naprave za sve atribute ili samo za izabrane najbitnije atribute
def build_models(categorical_features=None, numeric_features=None):
    # pipeline, prvose izvrsi preprocesing a zatim trenira model
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True, categorical_features=categorical_features, numeric_features=numeric_features)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")), # solver nacin na koji algoritam racuna paramtre logisticke regresije
        ]),
        "KNN": Pipeline([ # K najblizih suseda
            ("preprocessor", build_preprocessor(scale_numeric=True, categorical_features=categorical_features, numeric_features=numeric_features)),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", build_preprocessor(categorical_features=categorical_features, numeric_features=numeric_features)),
            ("classifier", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)), #br stabala, brzina ucenja, dubina stabla
        ]),
        "Decision Tree": Pipeline([ # stablo odlucivanja
            ("preprocessor", build_preprocessor(categorical_features=categorical_features, numeric_features=numeric_features)),
            ("classifier", DecisionTreeClassifier(max_depth=20, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE)),  #maks dubina 20, koristi najmanje 20 primera
        ]),
        "Random Forest": Pipeline([ # vise stabala odlucivanja
            ("preprocessor", build_preprocessor(categorical_features=categorical_features, numeric_features=numeric_features)),
            ("classifier", RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)), #broj stabala 300, dubina 18, koristi najmanje 20 primera, stablo gleda samo sqrt atributa, balansira klase, obezbedjuje da random deo algoritam bude ponovljiv, koristi sva jezgra procesora
        ]),
    }

# unakrsna validacija
def cross_validate_models(models, x_train, y_train, cv_splits=3):
    # svaka podela cuva slican odnos high low
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for model_name, model in models.items():
        # trenira i proverava model kroz unakrsnu validaciju
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1_macro")
        rows.append({
            "model": model_name,    # ime modela
            "cv_f1_macro_mean": scores.mean(),  # srednja vrednost F1 macro metrike kroz sv podele
            "cv_f1_macro_std": scores.std(),    #koliko rezultat varira izmedju podela
        })
    return pd.DataFrame(rows)

# koji su atributi najvazniji za finalni model
def get_feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # pokusava da dobije imena atributa posle preprocesiranja(pokusava jer se kategoriske kolone menjaju preko OneHotEncoder-a) ako ne uspe pravi genericka imena
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{index}" for index in range(len(classifier.feature_importances_))]

    # neki modeli imaju direktnu informaciju o vaznosti atributa
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    # gledamo koji atribut najvise utice
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_[0])
    else:
        return None

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)


# DODATO: bira najbitnije originalne atribute bez zakucavanja tacnog broja atributa
def get_important_original_features(feature_importance):
    if feature_importance is None or feature_importance.empty:
        return None

    importance_limit = feature_importance["importance"].mean()
    important_rows = feature_importance[feature_importance["importance"] >= importance_limit]

    if important_rows.empty:
        important_rows = feature_importance.head(1)

    selected_features = set()
    for feature_name in important_rows["feature"]:
        if feature_name.startswith("numeric__"):
            selected_features.add(feature_name.replace("numeric__", "", 1))
        elif feature_name.startswith("categorical__"):
            encoded_name = feature_name.replace("categorical__", "", 1)
            for original_feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
                if encoded_name.startswith(f"{original_feature}_"):
                    selected_features.add(original_feature)
                    break

    return [feature for feature in MODEL_FEATURES if feature in selected_features]


# cuva finalni istrenirani model
def save_model(model):
    MODEL_PATH.parent.mkdir(exist_ok=True) # putanja iz config.py
    joblib.dump(model, MODEL_PATH) # cuva model u fajlu
