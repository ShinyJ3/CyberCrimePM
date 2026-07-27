"""
evaluate.py

Evaluates the best saved model (models/model.pkl) against the held-out
test set produced by train.py.

Produces:
    - Printed accuracy / precision / recall / F1 / classification report
    - outputs/figures/confusion_matrix.png
    - outputs/figures/feature_importance.png (only for tree-based models)

NOTE: First working version, written without the Colab/reference material
yet. Will be revised with attribution comments once those are provided.

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
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 score:  {f1:.3f}")
    print("\nFull classification report:")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))


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
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved confusion matrix to {out_path}")


def plot_feature_importance(model, feature_columns, top_n=15):
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
