################################################################################
######################### Import Requisite Libraries ###########################
import os
import sys
from pathlib import Path
import typer
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_tuner.pickleObjects import dumpObjects, loadObjects

from core.constants import (
    var_index,
    preproc_run_name,
    exp_artifact_name,
    drop_vars,
)

from core.functions import (
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    safe_to_numeric,
)

app = typer.Typer()

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")


@app.command()
def main(
    input_data_file: str = "./data/raw/nuforc_data.parquet",
    output_data_file: str = "./data/processed/nuforc_preprocessed.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Preprocessing pipeline for NUFORC UAP dataset.
    Outputs a single cleaned parquet. Splitting is handled downstream.

    Args:
        input_data_file (str): Path to input parquet.
        output_data_file (str): Path to save cleaned parquet.
        stage (str): 'training' or 'inference'.
        data_path (str): Directory for processed outputs and artifacts.
    """

    ############################################################################
    # Step 1. Load Raw Data
    ############################################################################

    print(f"Loading data from: {input_data_file}")
    df = pd.read_parquet(input_data_file)
    print(f"Raw data shape: {df.shape}")

    # Lowercase all column names immediately — must happen before any column
    # references below, and unconditionally so inference sees the same names.
    df.columns = df.columns.str.lower()

    # Set index if not already set
    if df.index.name != var_index:
        try:
            df.set_index(var_index, inplace=True)
            print(f"Index set to '{var_index}'.")
        except KeyError:
            print(
                f"Warning: '{var_index}' not found in columns - "
                "proceeding with default integer index."
            )
    else:
        print(f"Index '{var_index}' already set - skipping.")

    ############################################################################
    # Step 2. Build Features and Target Variables (training only)
    ############################################################################
    #
    # media: binary X feature — did the witness attach a photo/video?
    #   media=Y   --> 1
    #   media=NaN --> 0
    #
    # dramatic: binary target — did NUFORC editors flag the report as
    #   dramatic or unusual via link punctuation?
    #   link contains "!" or "." --> 1 (dramatic / unusual)
    #   no punctuation flag       --> 0 (ordinary)
    #   Derived from link before it is dropped in Step 6.
    #
    # explained: binary target — did NUFORC staff provide an explanation?
    #   explanation non-null (e.g. "Starlink", "Aircraft") --> 1 (debunked)
    #   explanation null                                    --> 0 (unexplained)
    #   Derived from explanation before it is dropped in Step 6.
    #
    # Both targets are retained in the preprocessed parquet. The active
    # outcome is controlled by target_outcome in core/constants.py.
    # At inference time these raw columns may not exist — skip silently.
    ############################################################################

    if stage == "training":
        # media --> binary feature
        if "media" in df.columns:
            df["media"] = (df["media"] == "Y").astype(int)
            print("*" * 80)
            print(f"\nMedia (feature) distribution:\n{df['media'].value_counts()}")
            print(f"\nMedia positive rate: {df['media'].mean() * 100:.1f}%")
            print("*" * 80)

        # dramatic --> binary target (must be derived before link is dropped)
        if "link" in df.columns:
            df["dramatic"] = (
                df["link"].str.contains(r"[.!]", regex=True, na=False).astype(int)
            )
            print("*" * 80)
            print(f"\nTarget (dramatic) distribution:\n{df['dramatic'].value_counts()}")
            print(f"\nDramatic positive rate: {df['dramatic'].mean() * 100:.1f}%")
            print("*" * 80)

        # explained --> binary target (must be derived before explanation is dropped)
        if "explanation" in df.columns:
            df["explained"] = df["explanation"].notna().astype(int)
            # Cast raw explanation to str so parquet-safe dtype step doesn't choke
            df["explanation"] = df["explanation"].fillna("").astype(str).str.strip()
            print("*" * 80)
            print(
                f"\nTarget (explained) distribution:\n{df['explained'].value_counts()}"
            )
            print(f"\nExplained positive rate: {df['explained'].mean() * 100:.1f}%")
            print("*" * 80)

    ############################################################################
    # Step 3. Feature Engineering — Temporal Features from Occurred
    ############################################################################

    if "occurred" in df.columns:
        occurred = pd.to_datetime(df["occurred"], errors="coerce")
        df["hour_of_day"] = occurred.dt.hour
        df["month"] = occurred.dt.month
        df["day_of_week"] = occurred.dt.dayofweek

    ############################################################################
    # Step 3b. Days to Report (Occurred -> Reported)
    ############################################################################
    # Difference in days between when the sighting occurred and when it was
    # submitted to NUFORC. Witnesses who report quickly may recall more detail.
    # Negative values (reported before occurred) are data entry errors — clip
    # to 0. Cap at 365 to limit influence of extreme outliers.
    ############################################################################

    if "occurred" in df.columns and "reported" in df.columns:
        reported = pd.to_datetime(df["reported"], errors="coerce")
        df["days_to_report"] = (reported - occurred).dt.days
        df["days_to_report"] = df["days_to_report"].clip(lower=0, upper=365)
        print(
            f"\ndays_to_report — mean: {df['days_to_report'].mean():.1f}, "
            f"median: {df['days_to_report'].median():.1f}, "
            f"nulls: {df['days_to_report'].isna().sum()}"
        )

    ############################################################################
    # Step 4. Clean Text
    ############################################################################
    # Fill nulls with empty string — CatBoost text_features requires no NaNs.
    ############################################################################

    df["summary"] = df["summary"].fillna("").astype(str).str.strip()

    ############################################################################
    # Step 5. Clean Categorical Features
    ############################################################################

    df["shape"] = df["shape"].fillna("Unknown").astype(str).str.strip()
    df["country"] = df["country"].fillna("Unspecified").astype(str).str.strip()
    df["state"] = df["state"].fillna("Unknown").astype(str).str.strip()

    # Normalize spelled-out state names to 2-char abbreviations
    state_abbrev = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "District of Columbia": "DC",
        # Canadian provinces
        "Ontario": "ON",
        "Quebec": "QC",
        "British Columbia": "BC",
        "Alberta": "AB",
        "Manitoba": "MB",
        "Saskatchewan": "SK",
        "Nova Scotia": "NS",
        "New Brunswick": "NB",
        "Newfoundland": "NL",
        "Prince Edward Island": "PE",
    }
    df["state"] = df["state"].replace(state_abbrev)

    # Flag invalid US/Canada states as Unknown; preserve international states
    valid_us_ca = set(state_abbrev.values()) | {"Unknown"}

    def normalize_state(row):
        state = row["state"]
        country = row["country"]
        if country in {"USA", "Canada"}:
            return state if state in valid_us_ca else "Unknown"
        return state

    df["state"] = df.apply(normalize_state, axis=1)

    ############################################################################
    # Step 6. Drop Columns
    ############################################################################
    # drop_vars in constants.py should include at minimum:
    #   - "occurred"    replaced by hour_of_day, month, day_of_week
    #   - "reported"    replaced by days_to_report
    #   - "link"        URL identifier — dramatic target extracted above
    #   - "explanation" raw staff label — explained target extracted above
    #   - "city"        high cardinality, not declared as cat_feature
    ############################################################################

    df.drop(columns=drop_vars, errors="ignore", inplace=True)
    print(f"\nDataFrame first 5 rows:\n{df.head()}")
    print(f"\nShape after dropping columns: {df.shape}\n")

    ############################################################################
    # Step 7. Safe Numeric Conversion
    ############################################################################

    df = df.apply(lambda x: safe_to_numeric(x))

    ############################################################################
    # Step 8. Zero Variance Columns (fit on training, apply on inference)
    ############################################################################

    if stage == "training":
        numeric_cols = df.select_dtypes(include=["number"]).columns
        zero_varlist_list = list(df[numeric_cols].var()[lambda v: v == 0].index)

        print("*" * 80)
        print(f"Zero-variance columns: {zero_varlist_list}")
        print("*" * 80)

        dumpObjects(zero_varlist_list, os.path.join(data_path, "zero_varlist_list.pkl"))
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
            obj=zero_varlist_list,
        )
        zero_varlist_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
        )

    if stage == "inference":
        zero_varlist_list = loadObjects(
            os.path.join(data_path, "zero_varlist_list.pkl")
        )

    df.drop(columns=zero_varlist_list, errors="ignore", inplace=True)

    ############################################################################
    # Step 9. Parquet-safe dtypes
    ############################################################################

    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype(str)

    ############################################################################
    # Step 10. Log String Columns to MLflow (training only, for reference)
    ############################################################################

    if stage == "training":
        string_cols_list = df.select_dtypes("object").columns.to_list()
        print("*" * 80)
        print(f"String columns ({len(string_cols_list)}): {string_cols_list}")
        print("*" * 80)

        dumpObjects(string_cols_list, os.path.join(data_path, "string_cols_list.pkl"))
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="string_cols_list",
            obj=string_cols_list,
        )
        string_cols_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="string_cols_list",
        )

    ############################################################################
    # Step 11. Save
    ############################################################################

    os.makedirs(data_path, exist_ok=True)
    df.to_parquet(output_data_file)
    print(f"\nSaved preprocessed data: {output_data_file}  ({len(df):,} rows)")


if __name__ == "__main__":
    app()
