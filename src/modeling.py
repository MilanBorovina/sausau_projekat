import joblib
import pandas as pd

#alati za preprocessing, modele i evaluaciju
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Učitavanje konfiguracije projekta: kolone za obradu, putanja modela i random state
from config import CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE

# Kreira OneHotEncoder za pretvaranje kategorijskih/tekstualnih kolona u numerički oblik
def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# kategorijske kolone se kodiraju OneHotEncoder-om
 # Ako model zahteva skaliranje numeričkih vrednosti, koristi StandardScaler
def build_preprocessor(scale_numeric=False):
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), CATEGORICAL_FEATURES),
            ("numeric", numeric_step, NUMERIC_FEATURES),
        ]
    )


def build_models():
    # pipeline, prvose izvrsi preprocesing a zatim trenira model
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")), # solver nacin na koji algoritam racuna paramtre logisticke regresije
        ]),
        "KNN": Pipeline([ # K najblizih suseda
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=15, random_state=RANDOM_STATE)), #br stabala, brzina ucenja, dubina stabla
        ]),
        "Decision Tree": Pipeline([ # stablo odlucivanja
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(max_depth=20, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE)),  #maks dubina 20, koristi najmanje 20 primera
        ]),
        "Random Forest": Pipeline([ # vise stabala odlucivanja
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)), #broj stabala 300, dubina 18, koristi najmanje 20 primera, stablo gleda samo sqrt atributa, balansira klase, obezbedjuje da random deo algoritam bude ponovljiv, koristi sva jezgra procesora
        ]),
    }

# unakrsna validacija
def cross_validate_models(models, x_train, y_train, cv_splits=3):
    # svaka podela cuva slican odnos high low
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for model_name, model in models.items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1_macro")
        rows.append({
            "model": model_name,
            "cv_f1_macro_mean": scores.mean(),
            "cv_f1_macro_std": scores.std(), # racuna f1 macro za svaku podelu
        })
    return pd.DataFrame(rows)


def get_feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{index}" for index in range(len(classifier.feature_importances_))]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_[0])
    else:
        return None

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)


def save_model(model):
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
