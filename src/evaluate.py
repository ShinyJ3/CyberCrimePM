# This file is used to evaluate the performance of the trained model on the test set. 
# It loads the test data, the trained model, and the necessary artifacts, and then computes various evaluation metrics such as 
# accuracy, precision, recall, F1 score, and confusion matrix. It also generates a plot of feature importances if applicable.

# All sources used for are embedded within comments in the code, 
# especially for XGBoost explanation and the tuning of the hyperparameters, 
# and the evaluation metrics. 

# The libraries are used for data manipulation, model loading, evaluation metrics, and visualization.
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# These are the same constants as in train.py, and these will be used for loading the test split and the saved model artifacts.
MODELS_DIR = Path("models")
TEST_SPLIT_PATH = Path("data/processed/test_split.csv")
FIGURES_DIR = Path("outputs/figures")
TARGET_COLUMN = "resolution_class"


# This method is used to load the test data from the CSV file produced by train.py. 
# It raises a FileNotFoundError if the test split file does not exist, prompting the user to run train.py first. 
# The method returns the features (X_test) and labels (y_test) for evaluation.
def load_test_data():
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {TEST_SPLIT_PATH}. Run `python src/train.py` first."
        )
    df = pd.read_csv(TEST_SPLIT_PATH)
    # These lines separate the features and labels from the test DataFrame. 
    # The features are all columns except the target column, while the labels are in the target column.
    X_test = df.drop(columns=[TARGET_COLUMN])
    y_test = df[TARGET_COLUMN]
    return X_test, y_test

# This method is used to load the models and artifacts from the previously trained code in order to evaluate them on the test set. 
# It loads the model, label encoder, and feature columns from the specified paths in the models directory. 
# The method returns the loaded model, label encoder, and feature columns for use in evaluation.
def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, label_encoder, feature_columns

# This function prints out the evaluation metrics for the model's predictions on the test set.
def print_metrics(y_test, preds, label_encoder):
    # In order to research further about the metrics, I used the following sources to understand the metrics and their definitions, 
    # which are included in the docstring below.
    
    # Accuracy is the overall fraction of correct predictions
    # Source: https://www.geeksforgeeks.org/machine-learning/sklearn-classification-metrics/
    
    # Precision is the fraction of predicted "Slow" incidents that were actually Slow
    # Source: https://www.geeksforgeeks.org/machine-learning/sklearn-classification-metrics/
    
    # Recall is the fraction of actual "Slow" incidents that were correctly predicted as Slow
    # Source: https://www.geeksforgeeks.org/machine-learning/sklearn-classification-metrics/
    
    # F1 Score is the harmonic mean of Precision and Recall.
    # Source: https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/

    # I also utilized the classification_report() method from scikit-learn to get a detailed breakdown of precision, recall, 
    # and F1 score for each class (Fast and Slow), and the source for this is 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html.
 
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy:  {acc:.3f}  (overall fraction of correct predictions)")
    print(f"Precision: {prec:.3f}  (of predicted 'Slow', fraction that really were)")
    print(f"Recall:    {rec:.3f}  (of actual 'Slow' incidents, fraction correctly caught)")
    print(f"F1 score:  {f1:.3f}  (balance of precision and recall)")
    print("\nFull classification report (per-class breakdown):")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

# This is where the confusion matrix that is used in my slide is established, which is a grid comparing predicted vs. actual classes.
# I also learnt about True Positive, True Negative, False Positive, and False Negative from the following sources:
# https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/
def plot_confusion_matrix(y_test, preds, label_encoder):
    cm = confusion_matrix(y_test, preds)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )

# Although this was not mentioned in the presentation, in order to get the models to know the importance of each feature, 
# I used the feature_importances_ attribute of tree-based models to get the importance of each feature in making predictions.
# I also used the following source to understand how to plot the feature importance
# Source: https://www.geeksforgeeks.org/machine-learning/hyperparameters-of-random-forest-classifier/.
def plot_feature_importance(model, feature_columns, top_n=15):
    if not hasattr(model, "feature_importances_"):
        print(
            f"{type(model).__name__} does not expose feature_importances_ "
            "(this is normal for Logistic Regression) - skipping plot."
        )
        return

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=False).head(top_n)

    # This section creates a bar plot of the top N feature importances, saves it to the figures directory, 
    # and prints the path to the saved plot.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_features.values, y=top_features.index, orient="h")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature importance plot to {out_path}")

# Again, like with the other files, this is the main function that is called when the script is run from the command line.
def main():
    X_test, y_test = load_test_data()
    model, label_encoder, feature_columns = load_artifacts()

    print(f"Evaluating: {type(model).__name__}\n")

    preds = model.predict(X_test)

    print_metrics(y_test, preds, label_encoder)
    plot_confusion_matrix(y_test, preds, label_encoder)
    plot_feature_importance(model, feature_columns)

# And again, this is the main function that is called when the script is run from the command line.
if __name__ == "__main__":
    main()

