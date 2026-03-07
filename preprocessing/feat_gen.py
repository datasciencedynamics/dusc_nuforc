################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import typer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_tuner.pickleObjects import dumpObjects, loadObjects

from core.functions import mlflow_dumpArtifact, mlflow_loadArtifact

from core.constants import (
    var_index,
    exp_artifact_name,
    preproc_run_name,
    target_outcome,
)

################################################################################
################ Step 2: Define Typer Application ##############################
################################################################################

app = typer.Typer()

################################################################################
################ Step 3: Define Main Function ##################################
################################################################################

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")


@app.command()
def main(
    input_data_file: str = "./data/processed/nuforc_preprocessed.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Separates X from y and saves feature and target parquets for
    downstream modeling. Splitting is handled downstream.

    Args:
        input_data_file: Path to preprocessed parquet (training or inference)
        stage: 'training' or 'inference'
        data_path: Path to data directory
    """

    ############################################################################
    ################ Step 4: Load Data #########################################
    ############################################################################

    print("\n" + "=" * 80)
    print("Loading preprocessed data...")
    print("=" * 80)

    df = pd.read_parquet(input_data_file)
    print(f"Loaded shape: {df.shape}")

    # Set index if not already set
    if df.index.name != var_index:
        try:
            df.set_index(var_index, inplace=True)
            print(f"Index set to '{var_index}'.")
        except KeyError:
            print(
                f"Warning: '{var_index}' not found — "
                "proceeding with default integer index."
            )
    else:
        print(f"Index '{var_index}' already set — skipping.")

    ############################################################################
    ################ Step 5: Training Stage ####################################
    ############################################################################

    if stage == "training":

        ########################################################################
        # Separate X and y
        ########################################################################

        X = df.drop(columns=[target_outcome]).copy()
        y = df[target_outcome].copy()

        # Retain numeric + object columns only
        cols_to_keep = (
            X.select_dtypes(include=np.number).columns.tolist()
            + X.select_dtypes(include="object").columns.tolist()
        )
        X = X[cols_to_keep]
        X_columns_list = X.columns.tolist()

        print(f"\nFeature columns ({len(X_columns_list)}): {X_columns_list}")
        print(f"\nX shape: {X.shape}")
        print(f"\nFirst 5 rows of X:\n{X.head()}")
        print(f"\nTarget distribution:\n{y.value_counts()}")
        print(f"Positive rate: {y.mean() * 100:.1f}%")

        ########################################################################
        # Save X_columns_list
        ########################################################################

        dumpObjects(X_columns_list, os.path.join(data_path, "X_columns_list.pkl"))
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="X_columns_list",
            obj=X_columns_list,
        )
        X_columns_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="X_columns_list",
        )

        ########################################################################
        # Save X and y
        ########################################################################

        print("\n" + "=" * 80)
        print("Saving feature and target files...")
        print("=" * 80)

        X.to_parquet(os.path.join(data_path, "X.parquet"))
        pd.DataFrame(y).to_parquet(
            os.path.join(data_path, f"y_{target_outcome}.parquet")
        )

        print(f"Saved X.parquet  ({X.shape})")
        print(f"Saved y_{target_outcome}.parquet  ({y.shape})")

    ############################################################################
    ################ Step 6: Inference Stage ###################################
    ############################################################################

    elif stage == "inference":

        print("\n" + "=" * 80)
        print("Inference mode: Loading X_columns_list...")
        print("=" * 80)

        X_columns_list = loadObjects(os.path.join(data_path, "X_columns_list.pkl"))
        print(f"Loaded {len(X_columns_list)} feature columns")

        X = df[X_columns_list].copy()
        X.to_parquet(os.path.join(data_path, "X.parquet"))

        print(f"Saved X.parquet  ({X.shape})")
        print("\nInference features generated!")


################################################################################

if __name__ == "__main__":
    app()
