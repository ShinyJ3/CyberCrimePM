# This file is used to train the model, and it is run from the command line. It loads the processed data, 
# splits it into training and test sets, trains multiple models with hyperparameter tuning, evaluates them, 
# and saves the best model along with the label encoder and feature columns for later use in prediction.

# All citations (and there are a lot of them) are embedded within the code below and all explanations for using XGBoost are also
# explained, as we didn't learn about this model during class, but I used it because it is a very popular and powerful model for 
# classification tasks, and it often performs well on tabular data like this.

# These are the libraries used for loading data, splitting it, training models, and saving artifacts.
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Since XGBoost is not part of the standard library, we need to check if it is installed and import it if available.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Paths to the processed data, models directory, and test split CSV.
PROCESSED_PATH = Path("data/processed/cyber_processed.csv")
MODELS_DIR = Path("models")
TEST_SPLIT_PATH = Path("data/processed/test_split.csv")

# Target column, test size, and random state for reproducibility.
TARGET_COLUMN = "resolution_class"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Here I am using a constant for the number of cross-validation folds, 
# which is a common practice in machine learning to evaluate model performance.
# I do not remember exactly if we went over CV Folds in class, but I needed a specific way of cross-validation to ensure
# an equal distribution of Fast and Slow incidents in each fold, so I used StratifiedKFold cross-validation.

# Source: https://scikit-learn.org/stable/modules/cross_validation.html
CV_FOLDS = 5

# This is where the main function of the training script starts, 
# and it loads the processed data, splits it into features and target,
def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {PROCESSED_PATH}. Run `python src/preprocess.py` first."
        )
    return pd.read_csv(PROCESSED_PATH)

# This is where the features are split from the target variable, 
# and the target variable is encoded into numerical values for model training.
def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

# Since we have a binary form of classification, we can use a label encoder to convert the target variable 
# into numerical values (0 and 1) for model training, allowing us to put the labels of Fast and Slow on 
# resolution times into a numerical format for the model to understand.
def encode_target(y: pd.Series):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print(f"Target classes: {list(encoder.classes_)} -> {list(range(len(encoder.classes_)))}")
    return y_encoded, encoder




# Now we are getting into training the models within the functions below and each function is responsible for 
# training a specific model (Logistic Regression, Random Forest, and XGBoost) with hyperparameter tuning using cross-validation.

# The hyperparameter tuning is done using GridSearchCV for Logistic Regression and RandomizedSearchCV for Random Forest and XGBoost,
# which allows us to search over a specified parameter grid and find the best combination of hyperparameters for optimal model performance.




# This function is for training the Logistic Regression, which will be one of the weaker models that we use for comparison, 
# but it is a good baseline model to start with. It uses the class_weight parameter to handle class imbalance 
# and the C parameter to control regularization strength.

# Sources for the class_weight and C parameters in Logistic Regression:
# https://www.geeksforgeeks.org/machine-learning/how-does-the-classweight-parameter-in-scikit-learn-work/
# https://www.geeksforgeeks.org/machine-learning/hyperparameters-of-random-forest-classifier/
def train_logistic_regression(X_train, y_train, cv) -> LogisticRegression:
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

    base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    search = GridSearchCV(base_model, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"  Logistic Regression best params: {search.best_params_}")
    return search.best_estimator_

# This function is for training the Random Forest model, which is an ensemble of decision trees.
# It uses RandomizedSearchCV for hyperparameter tuning to find the best combination of parameters efficiently.

# Sources for the hyperparameters of Random Forest:
# https://www.geeksforgeeks.org/machine-learning/random-forest-hyperparameter-tuning-in-python/
# https://www.geeksforgeeks.org/machine-learning/hyperparameters-of-random-forest-classifier/
def train_random_forest(X_train, y_train, cv) -> RandomForestClassifier:

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    }

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )

    # For another form of cross-validation, I used RandomizedSearchCV for hyperparameter tuning to find the best 
    # combination of parameters efficiently.

    # Source:  https://www.geeksforgeeks.org/machine-learning/comparing-randomized-search-and-grid-search-for-hyperparameter-estimation-in-scikit-learn/
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

# This function is for training the XGBoost model, which is a gradient-boosted tree model.
# It uses RandomizedSearchCV for hyperparameter tuning to find the best combination of parameters efficiently

# Hyperparameters tuned here:
# n_estimators     - number of boosting rounds (trees added sequentially)
# max_depth        - depth of each tree
# learning_rate    - how much each new tree corrects previous mistakes (lower = more cautious, usually needs more trees)
# subsample        - fraction of rows used per tree (adds randomness, reduces overfitting)
# colsample_bytree - fraction of columns/features used per tree

# Instead of class_weight="balanced" like the other two models, scale_pos_weight helps XGBoost handle class imbalance.

# I did research on this form of model using Claude as I consulted it for information on XGBoost and how to implement it, 
# and research for how to implement was also through Claude, which is cited on my presentation.

# Sources: 
# https://www.geeksforgeeks.org/machine-learning/xgbclassifier/
# https://www.geeksforgeeks.org/machine-learning/xgboost-parameters/
def train_xgboost(X_train, y_train, cv):
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

    # Again, the cross-validation is done using RandomizedSearchCV for hyperparameter
    #  tuning to find the best combination of parameters efficiently.
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

# The scoring function evaluates a trained model on the test set and prints its accuracy and F1 score.
def score_model(model, X_test, y_test, name: str):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"  {name:<20} accuracy={acc:.3f}  f1={f1:.3f}")
    return {"accuracy": acc, "f1": f1}

# This is the usual main function that is called when the script is run from the command line. It orchestrates the loading of data,
# splitting, training, evaluating, and saving of models and artifacts.
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

    # We did not learn about this, but I used StratifiedKFold cross-validation to ensure that each fold of the 
    # training data has the same proportion of Fast and Slow incidents as the overall dataset.
    # Source: https://www.geeksforgeeks.org/machine-learning/stratified-k-fold-cross-validation/
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}
    trained_models = {}

    print("\nTuning and training models (this takes longer than a single fit,"
          " because each model tries several hyperparameter combinations)...")

    # This section of the code runs the logistic regression training using the test and training data, 
    # and it saves the model to the models directory. It also scores the model using accuracy and F1 score.
    log_reg = train_logistic_regression(X_train, y_train, cv)
    scores["logistic_regression"] = score_model(log_reg, X_test, y_test, "Logistic Regression")
    trained_models["logistic_regression"] = log_reg

    # This section of the code runs the random forest training using the test and training data, 
    # and it saves the model to the models directory. It also scores the model using accuracy and F1 score.
    rf = train_random_forest(X_train, y_train, cv)
    scores["random_forest"] = score_model(rf, X_test, y_test, "Random Forest")
    trained_models["random_forest"] = rf

    # This section of the code runs the XGBoost training using the test and training data, 
    # and it saves the model to the models directory. It also scores the model using accuracy and F1 score.
    # Even though we did not learn about XGBoost, I used it because it is a very popular and powerful model for classification tasks, 
    # and it often performs well on tabular data like this.
    # Source: https://www.geeksforgeeks.org/machine-learning/xgbclassifier/
    if XGBOOST_AVAILABLE:
        xgb = train_xgboost(X_train, y_train, cv)
        scores["xgboost"] = score_model(xgb, X_test, y_test, "XGBoost")
        trained_models["xgboost"] = xgb
    else:
        print("  xgboost not installed - skipping (pip install xgboost to include it).")

    # After training all models, we determine the best model based on the F1 score.
    # We then save all trained models, the best model, the label encoder, and the feature columns to the models directory.
    # Finally, we save the test split to a CSV file for future evaluation.
    best_name = max(scores, key=lambda name: scores[name]["f1"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} (f1={scores[best_name]['f1']:.3f})")

    for name, model in trained_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    feature_columns = list(X.columns)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    TEST_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_test.assign(**{TARGET_COLUMN: y_test}).to_csv(TEST_SPLIT_PATH, index=False)

    print(f"\nSaved models to {MODELS_DIR}/")
    print(f"Saved test split to {TEST_SPLIT_PATH}")

# Again, as usual, this is the main function that is called when the script is run from the command line. 
if __name__ == "__main__":
    main()

