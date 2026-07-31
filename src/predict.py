# This file is used to make predictions using the trained model. 
# It provides an interactive command-line interface (CLI), which is essentially for users to input details about a cyberattack incident 
# and receive a prediction of whether the resolution time will be "Fast" or "Slow," along with the model's confidence in that prediction 
# (which is the probability of the predicted class).


# For this file, I only used the source of the scikit-learn library for the predict_proba() method, so no additional libraries are needed,
# and the link for this is https://scikit-learn.org/stable/glossary.html#term-predict_proba. This helped me how to do the predictions and along
# with the knowledge from the Colab files of Python and the scikit-learn library, I was able to implement the predict.py file.

# Libraries used for loading the model artifacts, handling data, and building the interactive CLI.
import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path("models")

# These are the fields that the user will be prompted to input values for, and they correspond to the features used in the model.
FIELDS = [
    ("attack_type", "Attack Type"),
    ("target_industry", "Target Industry"),
    ("security_vulnerability_type", "Security Vulnerability Type"),
    ("defense_mechanism_used", "Defense Mechanism Used"),
    ("attack_source", "Attack Source"),
    ("country", "Country"),
]


# Similar to other files, functions are used for this file instead of notebooks and cells as it makes it easier on me to just call and
# the artifacts function is used here to load the model, label encoder, and feature columns from the models directory. 
# This is necessary for making predictions with the trained model.
def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, label_encoder, feature_columns

# This function is used to get the known categories for a specific field based on the feature columns.
def get_known_categories(feature_columns, field_prefix):
    prefix = f"{field_prefix}_"
    return sorted(
        col[len(prefix):] for col in feature_columns if col.startswith(prefix)
    )

# This function is used to build a single-row DataFrame from the user-provided input values.
def build_input_row(feature_columns, **kwargs) -> pd.DataFrame:
    row = {col: 0 for col in feature_columns}

    # This for loop is neccessary because it iterates through the user-provided values for each field and 
    # sets the corresponding dummy variable in the row to 1.
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

    # This return statement creates a DataFrame with a single row, which is what the model expects for making predictions.
    return pd.DataFrame([row])[feature_columns]  

# This is the main function that takes user input for the features, builds the input row, 
# and uses the trained model to make a prediction.
def predict(**kwargs):
    model, label_encoder, feature_columns = load_artifacts()
    input_row = build_input_row(feature_columns, **kwargs)
    probabilities = model.predict_proba(input_row)[0]
    predicted_index = probabilities.argmax()
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = probabilities[predicted_index]
    # This return statement provides the predicted label and the confidence score, which can be used to inform the user about the prediction.
    return predicted_label, confidence

# This is the main part of the whole CLI program, which runs the interactive prompt for the user to input values for each field, 
# and then it calls the predict function to get the prediction and confidence score.
def prompt_for_value(prompt_label, known_categories):
    print(f"\n{prompt_label}")
    print("  Known values: " + ", ".join(known_categories))
    while True:
        value = input(f"  Enter {prompt_label}: ").strip()
        if value:
            return value
        print("  Please type a value (can't be blank).")

# This allows for an interface in the terminal to look somewhat appealing, and this is what a frontend of the project with a webstie
# would look like, but this is a CLI version of the project and it is not a web application.
def run_interactive():
    print("=" * 60)
    print(" Cyberattack Resolution Predictor")
    print(" Type in the incident's details below.")
    print("=" * 60)

    # This line loads the model, label encoder, and feature columns from the models directory, which are necessary for making predictions.
    model, label_encoder, feature_columns = load_artifacts()

    answers = {}

    # The for loop iterates through each field defined in FIELDS, 
    # prompting the user for input and storing the responses in the answers dictionary.
    for field_name, prompt_label in FIELDS:
        known_categories = get_known_categories(feature_columns, field_name)
        answers[field_name] = prompt_for_value(prompt_label, known_categories)

    input_row = build_input_row(feature_columns, **answers)

    # This section uses the model to predict the probabilities for each possible resolution based on the input row. 
    # It then determines the predicted label and the confidence score.
    probabilities = model.predict_proba(input_row)[0]
    predicted_index = probabilities.argmax()
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = probabilities[predicted_index]

    # This section prints out the user's input, the predicted resolution, and the confidence score in a formatted manner.
    print("\n" + "-" * 60)
    print(" YOUR INPUT")
    for field_name, prompt_label in FIELDS:
        print(f"   {prompt_label}: {answers[field_name]}")
    print("-" * 60)
    print(f" PREDICTED RESOLUTION: {predicted_label.upper()}")
    print(f" CONFIDENCE:           {confidence:.1%}")
    print("-" * 60)

# This allows the entire file to be run with one function call, allowing for encapsulation of the CLI program.
if __name__ == "__main__":
    run_interactive()
