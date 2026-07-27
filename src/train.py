"""
train.py

Trains models to predict Fast vs. Slow incident resolution
(Option C) using the processed dataset from preprocess.py.

Trains:
    1. Logistic Regression   (baseline, interpretable)
    2. Random Forest         (main model, gives feature importance)
    3. XGBoost                (if installed, usually the strongest performer)

Saves:
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/xgboost.pkl            (if xgboost is installed)
    models/model.pkl              (copy of whichever model scored best on F1)
    models/label_encoder.pkl      (maps Fast/Slow <-> 0/1)
    models/feature_columns.json   (exact column order used for training)
    data/processed/test_split.csv (held-out test set, for evaluate.py)

NOTE: This is a first working version, written without the Colab/reference
material yet. Once those are provided, this will be revised with attribution
comments showing where specific techniques came from.

Run from the project root:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROCESSED_PATH = Path("data/processed/cyber_processed.csv")
MODELS_DIR = Path("models")
TEST_SPLIT_PATH = Path("data/processed/test_split.csv")

TARGET_COLUMN = "resolution_class"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------
# DATA LOADING / SPLITTING
# ---------------------------------------------------------

def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {PROCESSED_PATH}. Run `python src/preprocess.py` first."
        )
    return pd.read_csv(PROCESSED_PATH)


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def encode_target(y: pd.Series):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)  # Fast -> 0, Slow -> 1 (alphabetical)
    print(f"Target classes: {list(encoder.classes_)} -> {list(range(len(encoder.classes_)))}")
    return y_encoded, encoder


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def score_model(model, X_test, y_test, name: str):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"  {name:<20} accuracy={acc:.3f}  f1={f1:.3f}")
    return {"accuracy": acc, "f1": f1}


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    df = load_processed_data()
    X, y = split_features_target(df)
    y_encoded, label_encoder = encode_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}
    trained_models = {}

    print("\nTraining models...")

    log_reg = train_logistic_regression(X_train, y_train)
    scores["logistic_regression"] = score_model(log_reg, X_test, y_test, "Logistic Regression")
    trained_models["logistic_regression"] = log_reg

    rf = train_random_forest(X_train, y_train)
    scores["random_forest"] = score_model(rf, X_test, y_test, "Random Forest")
    trained_models["random_forest"] = rf

    if XGBOOST_AVAILABLE:
        xgb = train_xgboost(X_train, y_train)
        scores["xgboost"] = score_model(xgb, X_test, y_test, "XGBoost")
        trained_models["xgboost"] = xgb
    else:
        print("  xgboost not installed - skipping (pip install xgboost to include it).")

    # Pick the best model by F1 score
    best_name = max(scores, key=lambda name: scores[name]["f1"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} (f1={scores[best_name]['f1']:.3f})")

    # Save every trained model individually, plus a copy of the best as model.pkl
    for name, model in trained_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    # Save feature column order so predict.py can build matching input rows
    feature_columns = list(X.columns)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    # Save the held-out test set so evaluate.py doesn't need to re-split
    TEST_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_test.assign(**{TARGET_COLUMN: y_test}).to_csv(TEST_SPLIT_PATH, index=False)

    print(f"\nSaved models to {MODELS_DIR}/")
    print(f"Saved test split to {TEST_SPLIT_PATH}")


if __name__ == "__main__":
    main()
