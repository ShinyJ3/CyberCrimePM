"""
preprocess.py

Cleans and prepares the Global Cybersecurity Threats (2015-2024) dataset
for the "Fast vs. Slow Resolution" classification task (Option C).

Input : data/raw/cybersecurity_threats.csv   (raw Kaggle download)
Output: data/processed/cyber_processed.csv   (clean, encoded, ready for modeling)

Run from the project root:
    python src/preprocess.py
"""


import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIG — paths and column decisions live here so the
#    rest of the pipeline doesn't need to be touched later.
# ---------------------------------------------------------

RAW_PATH = Path("data/raw/cybersecurity_threats.csv")
PROCESSED_PATH = Path("data/processed/cyber_processed.csv")

# Columns used as predictors (features)
FEATURE_COLUMNS = [
    "attack_type",
    "target_industry",
    "security_vulnerability_type",
    "defense_mechanism_used",
    "attack_source",
    "country",
]

# Column we will bucket into the classification target
RESOLUTION_TIME_COLUMN = "incident_resolution_time_in_hours"

# Columns we intentionally EXCLUDE as predictors (per the plan:
# keeping financial loss / affected users out avoids them
# overshadowing the defense-mechanism signal we actually care about)
EXCLUDED_COLUMNS = ["financial_loss_in_million", "number_of_affected_users", "year"]


# ---------------------------------------------------------
# 2. LOAD
# ---------------------------------------------------------

def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find raw data at {path}. "
            "Download the CSV from Kaggle and place it there."
        )
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------
# 3. CLEAN COLUMN NAMES
#    Kaggle CSVs often have spaces/parentheses in headers,
#    e.g. "Incident Resolution Time (in Hours)".
#    Standardizing to snake_case avoids bugs down the line.
# ---------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)   # drop punctuation like ( ) $
        .str.replace(r"\s+", "_", regex=True)      # spaces -> underscores
    )
    return df


# ---------------------------------------------------------
# 4. MISSING VALUES & DUPLICATES
# ---------------------------------------------------------

def report_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print("Missing values by column:")
        print(missing)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Categorical predictors: fill with "Unknown" rather than dropping rows
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    # Rows missing the target itself can't be used for training
    if RESOLUTION_TIME_COLUMN in df.columns:
        before = len(df)
        df = df.dropna(subset=[RESOLUTION_TIME_COLUMN])
        after = len(df)
        if before != after:
            print(f"Dropped {before - after} rows missing resolution time.")
    return df


# ---------------------------------------------------------
# 5. CREATE THE TARGET VARIABLE
#    Median split: Fast = below median resolution time,
#    Slow = at/above median. Guarantees roughly balanced classes.
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 6. ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    existing_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        print(f"Warning: expected columns not found and will be skipped: {missing_features}")

    encoded = pd.get_dummies(df[existing_features], drop_first=False)
    result = pd.concat([encoded, df["resolution_class"]], axis=1)
    return result


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------

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


if __name__ == "__main__":
    main()
