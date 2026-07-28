"""
evaluate.py

Evaluates the best saved model (models/model.pkl) against the held-out
test set produced by train.py.

Produces:
    - Printed accuracy / precision / recall / F1 / classification report
      (each explained in plain English below, not just printed as raw numbers)
    - outputs/figures/confusion_matrix.png
    - outputs/figures/feature_importance.png (only for tree-based models)

===========================================================================
SOURCES USED IN THIS FILE:
- Confusion matrix concept (TP/TN/FP/FN):
  https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/
  https://www.geeksforgeeks.org/machine-learning/essential-metrics-for-model-assessment-tp-tn-fp-fn-in-machine-learning/
- Precision / Recall / Accuracy definitions:
  https://www.geeksforgeeks.org/machine-learning/sklearn-classification-metrics/
- F1 score definition and formula:
  https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/
- classification_report() output format (per-class precision/recall/f1/support,
  plus macro avg / weighted avg):
  https://www.geeksforgeeks.org/machine-learning/compute-classification-report-and-confusion-matrix-in-python/
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
===========================================================================

Run from the project root (after train.py):
    python src/evaluate.py
"""

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

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODELS_DIR = Path("models")
TEST_SPLIT_PATH = Path("data/processed/test_split.csv")
FIGURES_DIR = Path("outputs/figures")
TARGET_COLUMN = "resolution_class"


# ---------------------------------------------------------
# LOADING
# ---------------------------------------------------------

def load_test_data():
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {TEST_SPLIT_PATH}. Run `python src/train.py` first."
        )
    df = pd.read_csv(TEST_SPLIT_PATH)
    X_test = df.drop(columns=[TARGET_COLUMN])
    y_test = df[TARGET_COLUMN]
    return X_test, y_test


def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, label_encoder, feature_columns


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def print_metrics(y_test, preds, label_encoder):
    """
    Prints the four headline metrics, each with a plain-English explanation,
    plus the full per-class classification report.

    What each metric means (treating "Slow" as the positive class here,
    since that's usually the more actionable thing to catch early):

    ACCURACY - out of every prediction the model made, what fraction were
    correct overall (Fast predicted as Fast, AND Slow predicted as Slow).
    Simple to understand, but can be misleading if one class is much more
    common than the other (a model that always guesses the majority class
    can still score a high accuracy without actually learning anything).
    Source: https://www.geeksforgeeks.org/machine-learning/sklearn-classification-metrics/

    PRECISION - out of every incident the model predicted would be "Slow",
    what fraction actually were Slow. Low precision means the model cries
    wolf a lot (flags a lot of incidents as Slow that are really Fast).
    Source: https://www.geeksforgeeks.org/machine-learning/sklearn-classification-metrics/

    RECALL - out of every incident that was ACTUALLY Slow, what fraction did
    the model correctly catch. Low recall means the model misses a lot of
    genuinely slow incidents, incorrectly calling them Fast.
    Source: https://www.geeksforgeeks.org/machine-learning/sklearn-classification-metrics/

    F1 SCORE - the harmonic mean of precision and recall, giving a single
    number that only stays high when BOTH precision and recall are high
    (unlike a plain average, it drops sharply if either one is low). This
    is why we picked the "best model" by F1 in train.py instead of accuracy.
    Source: https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/

    CLASSIFICATION REPORT - breaks precision/recall/f1 down separately for
    EACH class (Fast and Slow), plus "support" (how many real examples of
    that class were in the test set), and macro/weighted averages across
    classes. Useful for spotting if the model is much better at predicting
    one class than the other.
    Source: https://www.geeksforgeeks.org/machine-learning/compute-classification-report-and-confusion-matrix-in-python/
    """
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


def plot_confusion_matrix(y_test, preds, label_encoder):
    """
    The confusion matrix is a grid comparing predicted vs. actual classes.
    For our binary Fast/Slow case, the four cells are:
        True Negative  (top-left):     actual Fast,  predicted Fast  (correct)
        False Positive (top-right):    actual Fast,  predicted Slow  (wrong)
        False Negative (bottom-left):  actual Slow,  predicted Fast  (wrong)
        True Positive  (bottom-right): actual Slow,  predicted Slow  (correct)
    This is more informative than a single accuracy number because it shows
    WHICH kind of mistake the model tends to make, not just how often it's
    wrong overall.
    Source: https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/
    """
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
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved confusion matrix to {out_path}")


def plot_feature_importance(model, feature_columns, top_n=15):
    """
    Tree-based models (Random Forest, XGBoost) can report how much each
    input feature contributed to reducing prediction error across all the
    trees. Higher importance = the model relied on that feature more often
    when deciding how to split the data. This is the chart that answers
    the project's core question: which defense mechanisms/attack types/etc.
    actually mattered most to the model's Fast/Slow predictions.
    Source: https://www.geeksforgeeks.org/machine-learning/hyperparameters-of-random-forest-classifier/
    (general concept of tree-based feature importance referenced there)
    """
    if not hasattr(model, "feature_importances_"):
        print(
            f"{type(model).__name__} does not expose feature_importances_ "
            "(this is normal for Logistic Regression) - skipping plot."
        )
        return

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=False).head(top_n)

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


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    X_test, y_test = load_test_data()
    model, label_encoder, feature_columns = load_artifacts()

    print(f"Evaluating: {type(model).__name__}\n")

    preds = model.predict(X_test)

    print_metrics(y_test, preds, label_encoder)
    plot_confusion_matrix(y_test, preds, label_encoder)
    plot_feature_importance(model, feature_columns)


if __name__ == "__main__":
    main()
