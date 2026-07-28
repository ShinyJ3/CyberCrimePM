"""
train.py

Trains models to predict Fast vs. Slow incident resolution (Option C)
using the processed dataset from preprocess.py.

Trains three models and tunes each one with cross-validated hyperparameter
search rather than just fitting a single fixed configuration:
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

===========================================================================
SOURCES USED IN THIS FILE (per-function citations are also inline below):
- GridSearchCV / RandomizedSearchCV for Random Forest:
  https://www.geeksforgeeks.org/machine-learning/random-forest-hyperparameter-tuning-in-python/
  https://www.geeksforgeeks.org/machine-learning/hyperparameters-of-random-forest-classifier/
- XGBoost hyperparameters (n_estimators, max_depth, learning_rate, subsample,
  colsample_bytree, scale_pos_weight):
  https://www.geeksforgeeks.org/machine-learning/xgbclassifier/
  https://www.geeksforgeeks.org/machine-learning/xgboost-parameters/
- StratifiedKFold cross-validation (keeps class proportions equal in every fold,
  important since Fast/Slow may not split perfectly evenly):
  https://www.geeksforgeeks.org/machine-learning/stratified-k-fold-cross-validation/
  https://scikit-learn.org/stable/modules/cross_validation.html
- class_weight="balanced" for handling class imbalance in Logistic Regression
  and Random Forest:
  https://www.geeksforgeeks.org/machine-learning/how-does-the-classweight-parameter-in-scikit-learn-work/
===========================================================================

A HONEST NOTE ON EXPECTATIONS: independent analyses of this exact Kaggle
dataset (e.g. a CMU statistics capstone project that ran regressions of
resolution time against year and attack type) found R-squared values close
to zero and non-significant p-values almost everywhere -- meaning the dataset
itself has very little real signal connecting these features to resolution
time. That means there is a ceiling on how accurate ANY model can get here,
no matter how well it's tuned. The tuning below is a legitimate improvement
over the first version (better validated, class-imbalance-aware, and no
longer relying on one lucky/unlucky train-test split), but don't expect
90%+ accuracy - if that ceiling is closer to 60-70%, this is a fact about
the dataset, not a bug in the code. This is a good, honest example of how
having a well-designed methodology, having the data suggest a research
conclusion beyond what you initially anticipated.

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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
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

# Number of folds for cross-validation during hyperparameter search.
# Each candidate hyperparameter combination is trained/validated CV_FOLDS
# times on different slices of the training data, and the scores are
# averaged - this gives a much more reliable estimate than a single split.
# Source: https://scikit-learn.org/stable/modules/cross_validation.html
CV_FOLDS = 5


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
    # X = every column except the target = what the model learns FROM
    # y = the resolution_class column = what the model tries to PREDICT
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def encode_target(y: pd.Series):
    # Models need numbers, not text, so "Fast"/"Slow" gets converted to 0/1.
    # LabelEncoder assigns 0 to whichever class comes first alphabetically,
    # which is "Fast" here.
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print(f"Target classes: {list(encoder.classes_)} -> {list(range(len(encoder.classes_)))}")
    return y_encoded, encoder


# ---------------------------------------------------------
# MODELS
#
# Each training function below now does a hyperparameter SEARCH instead of
# training one fixed configuration. Hyperparameters are settings we choose
# before training (like how many trees in a forest) as opposed to the
# coefficients/weights the model learns automatically from the data.
# Trying several combinations and keeping whichever validates best is
# standard practice for getting real performance gains out of these models.
# ---------------------------------------------------------

def train_logistic_regression(X_train, y_train, cv) -> LogisticRegression:
    """
    Logistic Regression baseline.

    class_weight="balanced" tells the model to pay more attention to
    whichever class is less common in the training data, instead of just
    optimizing for whichever class happens to be more frequent.
    Source: https://www.geeksforgeeks.org/machine-learning/how-does-the-classweight-parameter-in-scikit-learn-work/

    C controls regularization strength (how much the model is penalized for
    having large coefficients - lower C = simpler/more regularized model,
    which helps prevent overfitting).
    Source: https://www.geeksforgeeks.org/machine-learning/hyperparameters-of-random-forest-classifier/
    (same GridSearchCV pattern shown there, applied here to Logistic Regression's C)
    """
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

    base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    search = GridSearchCV(base_model, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"  Logistic Regression best params: {search.best_params_}")
    return search.best_estimator_


def train_random_forest(X_train, y_train, cv) -> RandomForestClassifier:
    """
    Random Forest: an ensemble of many decision trees, each trained on a
    random subset of the data/features, with the final prediction being
    a majority vote across all trees. This usually generalizes better than
    a single decision tree.

    Hyperparameters tuned here:
        n_estimators      - number of trees (more trees = more stable, but slower)
        max_depth         - how deep each tree can grow (limits overfitting)
        min_samples_split - minimum samples needed to split a node
        max_features      - how many features each tree considers per split
    Source: https://www.geeksforgeeks.org/machine-learning/random-forest-hyperparameter-tuning-in-python/
    Source: https://www.geeksforgeeks.org/machine-learning/hyperparameters-of-random-forest-classifier/
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    }

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )
    # RandomizedSearchCV samples a fixed number of random combinations instead
    # of trying every single one (GridSearchCV would), which is much faster
    # when the grid is this large while still finding a strong configuration.
    # Source: https://www.geeksforgeeks.org/machine-learning/comparing-randomized-search-and-grid-search-for-hyperparameter-estimation-in-scikit-learn/
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"  Random Forest best params: {search.best_params_}")
    return search.best_estimator_


def train_xgboost(X_train, y_train, cv):
    """
    XGBoost: gradient-boosted trees. Unlike Random Forest (which builds trees
    independently and averages them), XGBoost builds trees one at a time,
    where each new tree specifically tries to correct the mistakes of the
    trees before it.

    Hyperparameters tuned here:
        n_estimators     - number of boosting rounds (trees added sequentially)
        max_depth        - depth of each tree
        learning_rate    - how much each new tree corrects previous mistakes
                           (lower = more cautious, usually needs more trees)
        subsample        - fraction of rows used per tree (adds randomness,
                           reduces overfitting)
        colsample_bytree - fraction of columns/features used per tree
    scale_pos_weight helps XGBoost handle class imbalance, similar to
    class_weight="balanced" for the other two models.
    Source: https://www.geeksforgeeks.org/machine-learning/xgbclassifier/
    Source: https://www.geeksforgeeks.org/machine-learning/xgboost-parameters/
    """
    # scale_pos_weight is set to the ratio of negative to positive class
    # counts, as recommended in the XGBoost documentation referenced above.
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }

    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"  XGBoost best params: {search.best_params_}")
    return search.best_estimator_


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

    # stratify=y_encoded ensures the train and test sets both keep roughly
    # the same Fast/Slow ratio as the full dataset, rather than risking a
    # split that's accidentally lopsided.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")

    # StratifiedKFold is reused across every model's hyperparameter search so
    # each candidate is judged on the exact same folds - a fair comparison.
    # Source: https://www.geeksforgeeks.org/machine-learning/stratified-k-fold-cross-validation/
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}
    trained_models = {}

    print("\nTuning and training models (this takes longer than a single fit,"
          " because each model tries several hyperparameter combinations)...")

    log_reg = train_logistic_regression(X_train, y_train, cv)
    scores["logistic_regression"] = score_model(log_reg, X_test, y_test, "Logistic Regression")
    trained_models["logistic_regression"] = log_reg

    rf = train_random_forest(X_train, y_train, cv)
    scores["random_forest"] = score_model(rf, X_test, y_test, "Random Forest")
    trained_models["random_forest"] = rf

    if XGBOOST_AVAILABLE:
        xgb = train_xgboost(X_train, y_train, cv)
        scores["xgboost"] = score_model(xgb, X_test, y_test, "XGBoost")
        trained_models["xgboost"] = xgb
    else:
        print("  xgboost not installed - skipping (pip install xgboost to include it).")

    # Pick the best model by F1 score (a balance of precision and recall -
    # see evaluate.py for a full explanation of what that means and why
    # it's a better tiebreaker than raw accuracy for this project).
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
