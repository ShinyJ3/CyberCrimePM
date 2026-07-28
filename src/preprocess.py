
# This file is used to clean and preprocess the raw Kaggle dataset for the Fast vs. Slow Resolution classification task.
# The input is the raw CSV file downloaded from Kaggle, and the output is a cleaned and encoded CSV file ready for modeling.
# The approach used for creating the Fast/Slow label is a median split, which is a standard binning technique.

import pandas as pd
from pathlib import Path


# For this file, I used pandas' built-in functions for cleaning and encoding, so no additional libraries are need
# Citation: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

# I found this approach of importing the data and assigning it to these special Path variables to be more 
# useful than regular string paths, as it allows for easier path manipulation and is more robust across different operating systems
# according to my research, so this is the Kaggle dataset file that will be used for the rest of the project
RAW_PATH = Path("data/raw/cybersecurity_threats.csv")
PROCESSED_PATH = Path("data/processed/cyber_processed.csv")

# Features that will be used as predictors for the classification task
FEATURE_COLUMNS = [
    "attack_type",
    "target_industry",
    "security_vulnerability_type",
    "defense_mechanism_used",
    "attack_source",
    "country",
]

# This is the new column that will be created to represent the target variable for the classification task
RESOLUTION_TIME_COLUMN = "incident_resolution_time_in_hours"

# Columns that will be excluded from the final dataset, as they are not relevant for the classification task
EXCLUDED_COLUMNS = ["financial_loss_in_million", "number_of_affected_users", "year"]


# Functions are used for this file instead of notebooks and cells as it makes it easier on me to just call 
# functions in the main() pipeline, and also makes it easier to test each function individually.
def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Columns cleaned using RegEx and string methods to remove whitespace, special characters, and convert to lowercase. 
# This is important for consistency and to avoid issues with column names during processing.
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)   
        .str.replace(r"\s+", "_", regex=True)      
    )
    return df


# Drops missing/null values, but there were none in the dataset, so this is just a precautionary step.
def report_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print("Missing values by column:")
        print(missing)

# Drops the duplicate values in the dataset, and since before did equal after, there were no duplicates in the dataset, 
# so this is just a precautionary step.
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows.")
    return df

# Function to handle the values that are null and dropped, but since there were none, this is again just a 
# precautionary step. 
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")
    if RESOLUTION_TIME_COLUMN in df.columns:
        before = len(df)
        df = df.dropna(subset=[RESOLUTION_TIME_COLUMN])
        after = len(df)
        if before != after:
            print(f"Dropped {before - after} rows missing resolution time.")
    return df


# Here we create our target variable for the classification task, which is a binary label indicating whether the 
# resolution time is above or below the median.
def create_resolution_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    median_time = df[RESOLUTION_TIME_COLUMN].median()
    df["resolution_class"] = (
        df[RESOLUTION_TIME_COLUMN] >= median_time
    ).map({True: "Slow", False: "Fast"})

    print(f"Median resolution time: {median_time:.2f} hours")
    print("Class balance:")
    print(df["resolution_class"].value_counts())
    return df


# This function essentially encodes the categorical features into a format that can be used by machine learning models, 
# which is one-hot encoding in this case.
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    existing_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        print(f"Warning: expected columns not found and will be skipped: {missing_features}")

    # These lines of code use pandas' get_dummies function to perform one-hot encoding on the categorical features, 
    # and then concatenate the encoded features with the target variable into a new DataFrame.
    encoded = pd.get_dummies(df[existing_features], drop_first=False)
    result = pd.concat([encoded, df["resolution_class"]], axis=1)
    return result


# The pipeline that runs all the functions and prints out their functions, cleaning the entire dataset and 
# preparing it for modeling. The final processed dataset is saved to a CSV file which is stored in the data folder
def main():
    df = load_raw_data(RAW_PATH)
    df = clean_column_names(df)

    print("\nColumns after cleaning:", list(df.columns))

    report_missing_values(df)
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = create_resolution_target(df)

    final_df = encode_features(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved processed data to {PROCESSED_PATH} ({final_df.shape[0]} rows, {final_df.shape[1]} columns)")

# Used for testing the functions in this file, and also allows for the script to be run as a standalone program.
if __name__ == "__main__":
    main()
