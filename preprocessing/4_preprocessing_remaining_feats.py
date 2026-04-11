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
    input_data_file: str = "./data/processed/NUFORC_enriched.parquet",
    output_data_file: str = "./data/processed/df_sans_zero_missing.parquet",
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
        if "has_media" in df.columns:
            print("*" * 80)
            print(f"\nMedia (feature) distribution:\n{df['has_media'].value_counts()}")
            print(f"\nMedia positive rate: {df['has_media'].mean() * 100:.1f}%")
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

    ############################################################################
    # Step 3. Clean Text
    ############################################################################
    # Fill nulls with empty string. CatBoost text_features requires no NaNs.
    ############################################################################

    df["summary_clean"] = df["summary_clean"].fillna("").astype(str).str.strip()

    ############################################################################
    # Step 4. Clean Categorical Features
    ############################################################################

    df["shape"] = df["shape"].fillna("Unknown").astype(str).str.strip()
    df["country"] = df["country"].fillna("Unspecified").astype(str).str.strip()
    df["state"] = df["state"].fillna("Unknown").astype(str).str.strip()

    ############################################################################
    # Step 5. Drop Columns
    ############################################################################
    # drop_vars in constants.py should include at minimum:
    #   - "occurred"    replaced by hour_of_day, month, day_of_week
    #   - "reported"    replaced by days_to_report
    #   - "link"        URL identifier — dramatic target extracted above
    #   - "explanation" raw staff label — explained target extracted above
    #   - "media"       binary media presence feature extracted already
    #   - "city"        high cardinality, not declared as cat_feature
    #   - "shape_group" redundant with shape, and not declared as cat_feature
    #   - "summary"     cleaned raw text (summary_clean) retained for CatBoost
    #                   text_features, but not
    ############################################################################

    df.drop(columns=drop_vars, errors="ignore", inplace=True)
    print(f"\nDataFrame first 5 rows:\n{df.head()}")
    print(f"\nDropped columns: {drop_vars}")
    print(f"\nShape after dropping columns: {df.shape}\n")

    ############################################################################
    # Step 6. Safe Numeric Conversion
    ############################################################################

    df = df.apply(lambda x: safe_to_numeric(x))

    ############################################################################
    # Step 7. Zero Variance Columns (fit on training, apply on inference)
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
    # Step 8. Parquet-safe dtypes
    ############################################################################

    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype(str)

    ############################################################################
    # Step 9. Log String Columns to MLflow (training only, for reference)
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
    # Step 10. Save
    ############################################################################

    os.makedirs(data_path, exist_ok=True)
    df.to_parquet(output_data_file)
    print(f"\nSaved preprocessed data: {output_data_file}  ({len(df):,} rows)")


if __name__ == "__main__":
    app()
