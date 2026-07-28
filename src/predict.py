"""
predict.py

Takes a new incident's characteristics and returns a Fast/Slow
resolution prediction using the saved best model (models/model.pkl).

Usage as a script (edit the example at the bottom, or wire up argparse
later for a real CLI):
    python src/predict.py

Usage as a function from other code:
    from predict import predict
    label, confidence = predict(
        attack_type="Ransomware",
        target_industry="Healthcare",
        security_vulnerability_type="Unpatched Software",
        defense_mechanism_used="Firewall",
        attack_source="Hacker Group",
        country="USA",
    )

NOTE ON SOURCES: this file is mostly custom plumbing (matching a raw input
to the one-hot encoded columns the model expects) rather than a technique
pulled from an external tutorial, so there's nothing to cite here beyond
scikit-learn's own predict_proba() API, which is standard library usage:
https://scikit-learn.org/stable/glossary.html#term-predict_proba
"""

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path("models")


# ---------------------------------------------------------
# LOADING
# ---------------------------------------------------------

def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, label_encoder, feature_columns


# ---------------------------------------------------------
# BUILD A MATCHING INPUT ROW
#
# The trained model expects one-hot encoded columns like
# "attack_type_Ransomware", "target_industry_Healthcare", etc.
# This function builds a single row of all-zeros and flips on
# the columns that match the given inputs.
# ---------------------------------------------------------

def build_input_row(feature_columns, **kwargs) -> pd.DataFrame:
    row = {col: 0 for col in feature_columns}

    for field_name, value in kwargs.items():
        dummy_col = f"{field_name}_{value}"
        if dummy_col in row:
            row[dummy_col] = 1
        else:
            print(
                f"Warning: value '{value}' for '{field_name}' was not seen during "
                "training. This input will be treated as 'none of the known "
                "categories' for that field, which may reduce accuracy."
            )

    return pd.DataFrame([row])[feature_columns]  # enforce exact training column order


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

def predict(**kwargs):
    """
    Expected keyword arguments (must match preprocess.py's FEATURE_COLUMNS):
        attack_type
        target_industry
        security_vulnerability_type
        defense_mechanism_used
        attack_source
        country
    """
    model, label_encoder, feature_columns = load_artifacts()
    input_row = build_input_row(feature_columns, **kwargs)

    # predict_proba returns the model's estimated probability for EACH class
    # (e.g. [0.35, 0.65] = 35% Fast, 65% Slow). We take whichever class has
    # the higher probability as the prediction, and report that probability
    # as the "confidence" - it's the model's own estimate of how sure it is,
    # not a guarantee of correctness.
    probabilities = model.predict_proba(input_row)[0]
    predicted_index = probabilities.argmax()
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = probabilities[predicted_index]

    return predicted_label, confidence


if __name__ == "__main__":
    # Example prediction - swap these values to test different scenarios
    label, confidence = predict(
        attack_type="Ransomware",
        target_industry="Healthcare",
        security_vulnerability_type="Unpatched Software",
        defense_mechanism_used="Firewall",
        attack_source="Hacker Group",
        country="USA",
    )
    print(f"Predicted resolution: {label} ({confidence:.1%} confidence)")
