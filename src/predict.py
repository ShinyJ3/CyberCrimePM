"""
predict.py

Interactive command-line tool: asks the user to type in each incident
feature one at a time, then returns a Fast/Slow resolution prediction
with a confidence score, using the saved best model (models/model.pkl).

Run from the project root (after preprocess.py and train.py):
    python src/predict.py

This file can also be imported and used programmatically:
    from predict import predict
    label, confidence = predict(
        attack_type="Ransomware",
        target_industry="Healthcare",
        security_vulnerability_type="Unpatched Software",
        defense_mechanism_used="Firewall",
        attack_source="Hacker Group",
        country="USA",
    )

NOTE ON SOURCES: this file is mostly custom plumbing (matching a typed
input to the one-hot encoded columns the model expects, plus a plain
input()-based CLI loop) rather than a technique pulled from an external
tutorial, so there's nothing to cite here beyond scikit-learn's own
predict_proba() API, which is standard library usage:
https://scikit-learn.org/stable/glossary.html#term-predict_proba
"""

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path("models")

# Human-readable prompt + the prefix used in the one-hot encoded columns
# for that field (must match preprocess.py's FEATURE_COLUMNS order).
FIELDS = [
    ("attack_type", "Attack Type"),
    ("target_industry", "Target Industry"),
    ("security_vulnerability_type", "Security Vulnerability Type"),
    ("defense_mechanism_used", "Defense Mechanism Used"),
    ("attack_source", "Attack Source"),
    ("country", "Country"),
]


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
# FIGURE OUT VALID CATEGORIES PER FIELD
#
# feature_columns looks like ["attack_type_Ransomware",
# "attack_type_Phishing", "target_industry_Healthcare", ...]. This pulls
# out, for each field, the list of category values the model actually
# saw during training - so we can show the user their options instead of
# leaving them to guess exact spelling/capitalization.
# ---------------------------------------------------------

def get_known_categories(feature_columns, field_prefix):
    prefix = f"{field_prefix}_"
    return sorted(
        col[len(prefix):] for col in feature_columns if col.startswith(prefix)
    )


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


# ---------------------------------------------------------
# INTERACTIVE CLI
# ---------------------------------------------------------

def prompt_for_value(prompt_label, known_categories):
    """
    Shows the user the valid categories for a field (so they don't have to
    guess exact spelling), then asks them to type one. Keeps asking until
    they enter something non-empty; if they type something unrecognized,
    it warns them but still lets them proceed (build_input_row will treat
    it as "none of the known categories" for that field).
    """
    print(f"\n{prompt_label}")
    print("  Known values: " + ", ".join(known_categories))
    while True:
        value = input(f"  Enter {prompt_label}: ").strip()
        if value:
            return value
        print("  Please type a value (can't be blank).")


def run_interactive():
    print("=" * 60)
    print(" Cyberattack Resolution Predictor")
    print(" Type in the incident's details below.")
    print("=" * 60)

    model, label_encoder, feature_columns = load_artifacts()

    answers = {}
    for field_name, prompt_label in FIELDS:
        known_categories = get_known_categories(feature_columns, field_name)
        answers[field_name] = prompt_for_value(prompt_label, known_categories)

    input_row = build_input_row(feature_columns, **answers)

    probabilities = model.predict_proba(input_row)[0]
    predicted_index = probabilities.argmax()
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = probabilities[predicted_index]

    print("\n" + "-" * 60)
    print(" YOUR INPUT")
    for field_name, prompt_label in FIELDS:
        print(f"   {prompt_label}: {answers[field_name]}")
    print("-" * 60)
    print(f" PREDICTED RESOLUTION: {predicted_label.upper()}")
    print(f" CONFIDENCE:           {confidence:.1%}")
    print("-" * 60)


if __name__ == "__main__":
    run_interactive()