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
    percent_miss,
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
    # Step 2. Build Target Variable (training only)
    ############################################################################
    # Media=Y   → 1 (witness submitted photo/video with report)
    # Media=NaN → 0 (no media submitted)
    # At inference time the target column won't exist — skip silently.
    ############################################################################

    if stage == "training":
        df["media"] = (df["Media"] == "Y").astype(int)
        print(f"\nTarget distribution:\n{df['media'].value_counts()}")
        print(f"Positive rate: {df['media'].mean():.3f}")

    ############################################################################
    # Step 3. Feature Engineering — Temporal Features from Occurred
    ############################################################################

    if "Occurred" in df.columns:
        occurred = pd.to_datetime(df["Occurred"], errors="coerce")
        df["hour_of_day"] = occurred.dt.hour
        df["month"] = occurred.dt.month
        df["day_of_week"] = occurred.dt.dayofweek

    ############################################################################
    # Step 4. Clean Text
    ############################################################################
    # Fill nulls with empty string — CatBoost text_features requires no NaNs.
    ############################################################################

    df["Summary"] = df["Summary"].fillna("").astype(str).str.strip()

    ############################################################################
    # Step 5. Clean Categorical Features
    ############################################################################

    df["Shape"] = df["Shape"].fillna("Unknown").astype(str).str.strip()
    df["Country"] = df["Country"].fillna("Unspecified").astype(str).str.strip()
    df["State"] = df["State"].fillna("Unknown").astype(str).str.strip()

    ############################################################################
    # Step 6. Drop Columns
    ############################################################################
    # drop_vars in constants.py should include at minimum:
    #   - "Occurred"    replaced by hour_of_day, month, day_of_week
    #   - "Reported"    post-event metadata
    #   - "Link"        URL identifier
    #   - "Media"       raw source of target — must drop to avoid leakage
    #   - "Explanation" post-hoc label, high missingness differential
    ############################################################################

    df.drop(columns=drop_vars, errors="ignore", inplace=True)
    print(f"\nDataFrame first 5 rows:\n{df.head()}")
    print(f"\nShape after dropping columns: {df.shape}")

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

        print(f"\nZero-variance columns: {zero_varlist_list}\n")

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
    # Step 9. Row-wise Missingness Percentage
    ############################################################################

    df[percent_miss] = df.isna().mean(axis=1)

    ############################################################################
    # Step 10. Parquet-safe dtypes
    ############################################################################

    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype(str)

    ############################################################################
    # Step 11. Log String Columns to MLflow (training only, for reference)
    ############################################################################

    if stage == "training":
        string_cols_list = df.select_dtypes("object").columns.to_list()
        print(f"\nString columns ({len(string_cols_list)}): {string_cols_list}")

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
    # Step 12. Save
    ############################################################################

    os.makedirs(data_path, exist_ok=True)
    df.to_parquet(output_data_file)
    print(f"\nSaved preprocessed data: {output_data_file}  ({len(df):,} rows)")


if __name__ == "__main__":
    app()
