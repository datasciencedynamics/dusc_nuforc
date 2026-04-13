#!/usr/bin/env python3
"""
save_predictions.py
===================
Generates and saves predicted probabilities, optimal thresholds, test
labels, test feature sets, and model feature names for the four best
models used in performance_assessment.ipynb.

Outputs (all saved to ./models/predictions/)
--------------------------------------------
  y_test.parquet                      — test set labels (dramatic outcome)
  X_test_cat_feats_and_text.parquet   — test features for CatBoost Feats+Text
  X_test_cat_text_only.parquet        — test features for CatBoost Text Only
  X_test_cat.parquet                  — test features for CatBoost Tabular
  X_test_lr.parquet                   — test features for Logistic Regression
  y_prob_cat_feats_and_text.parquet   — predicted probabilities
  y_prob_cat_text_only.parquet
  y_prob_cat.parquet
  y_prob_lr.parquet
  model_thresholds.json               — optimal thresholds per model
  model_feature_names.json            — feature names per model

This allows the notebook to be reproduced without access to mlruns by
loading these files directly instead of loading models from MLflow.

Usage
-----
    python modeling/save_predictions.py

Models loaded
-------------
  cat_feats_and_text  : best overall (text + tabular CatBoost)
  cat_text_only       : text-only CatBoost ablation
  cat (smote)         : best tabular CatBoost
  lr  (orig)          : best logistic regression
"""

import json
import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.constants import drop_vars
from core.config import PROCESSED_DATA_DIR
from core.functions import mlflow_load_model

app = typer.Typer()


@app.command()
def main(
    outcome: str = "dramatic",
    text_col: str = "summary_clean",
    cat_pipeline: str = "smote",
    lr_pipeline: str = "orig",
    output_dir: Path = Path("./models/predictions"),
):
    """
    Load the four best models from MLflow, generate predicted probabilities
    on the test set, and save everything to output_dir for notebook
    reproducibility without requiring access to mlruns.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ############################################################################
    # Step 1. Load models from MLflow
    ############################################################################

    logger.info("Loading models from MLflow...")

    model_cat_feats_and_text = mlflow_load_model(
        experiment_name=f"{outcome}_text_model",
        run_name="cat_feats_and_text_orig_training",
        model_name=f"cat_feats_and_text_{outcome}",
    )

    model_cat_text_only = mlflow_load_model(
        experiment_name=f"{outcome}_text_model",
        run_name="cat_text_only_orig_training",
        model_name=f"cat_text_only_{outcome}",
    )

    model_cat = mlflow_load_model(
        experiment_name=f"{outcome}_model",
        run_name=f"cat_{cat_pipeline}_training",
        model_name=f"cat_{outcome}",
    )

    model_lr = mlflow_load_model(
        experiment_name=f"{outcome}_model",
        run_name=f"lr_{lr_pipeline}_training",
        model_name=f"lr_{outcome}",
    )

    logger.success("All models loaded.")

    ############################################################################
    # Step 2. Load features and labels
    ############################################################################

    logger.info("Loading features and labels...")

    X_full = pd.read_parquet(PROCESSED_DATA_DIR / "X.parquet")
    y = pd.read_parquet(PROCESSED_DATA_DIR / f"y_{outcome}.parquet").squeeze()

    # Prepare X variants matching train.py preprocessing logic
    X_feats_and_text = X_full.drop(
        columns=[c for c in drop_vars if c in X_full.columns], errors="ignore"
    )
    X_feats_and_text[text_col] = X_feats_and_text[text_col].fillna("").astype(str)

    X_text_only = X_full[[text_col]].copy()
    X_text_only[text_col] = X_text_only[text_col].fillna("").astype(str)

    X_tabular = X_full.drop(columns=[text_col], errors="ignore")

    ############################################################################
    # Step 3. Get test splits
    ############################################################################

    logger.info("Extracting test splits...")

    X_test_feats_and_text, y_test = model_cat_feats_and_text.get_test_data(
        X_feats_and_text, y
    )
    X_test_text_only, _ = model_cat_text_only.get_test_data(X_text_only, y)
    X_test_cat, _ = model_cat.get_test_data(X_tabular, y)
    X_test_lr, _ = model_lr.get_test_data(X_tabular, y)

    logger.info(f"  Test set size: {len(y_test):,} rows")

    ############################################################################
    # Step 4. Generate predicted probabilities
    ############################################################################

    logger.info("Generating predicted probabilities...")

    y_prob_cat_feats_and_text = pd.Series(
        model_cat_feats_and_text.predict_proba(X_test_feats_and_text)[:, 1],
        index=X_test_feats_and_text.index,
        name="y_prob_cat_feats_and_text",
    )

    y_prob_cat_text_only = pd.Series(
        model_cat_text_only.predict_proba(X_test_text_only)[:, 1],
        index=X_test_text_only.index,
        name="y_prob_cat_text_only",
    )

    y_prob_cat = pd.Series(
        model_cat.predict_proba(X_test_cat)[:, 1],
        index=X_test_cat.index,
        name="y_prob_cat",
    )

    y_prob_lr = pd.Series(
        model_lr.predict_proba(X_test_lr)[:, 1],
        index=X_test_lr.index,
        name="y_prob_lr",
    )

    ############################################################################
    # Step 5. Extract optimal thresholds
    ############################################################################

    logger.info("Extracting optimal thresholds...")

    model_thresholds = {
        "CatBoost Feats + Text": next(
            iter(model_cat_feats_and_text.threshold.values())
        ),
        "CatBoost Text Only": next(iter(model_cat_text_only.threshold.values())),
        "CatBoost Tabular (SMOTE)": next(iter(model_cat.threshold.values())),
        "Logistic Regression": next(iter(model_lr.threshold.values())),
    }

    for name, thresh in model_thresholds.items():
        logger.info(f"  {name}: {thresh:.4f}")

    ############################################################################
    # Step 6. Extract feature names per model
    ############################################################################

    logger.info("Extracting feature names...")

    feature_names = {
        "cat_feats_and_text": model_cat_feats_and_text.get_feature_names(),
        "cat_text_only": model_cat_text_only.get_feature_names(),
        "cat": model_cat.get_feature_names(),
        "lr": model_lr.get_feature_names(),
    }

    for name, feats in feature_names.items():
        logger.info(f"  {name}: {len(feats)} features")

    ############################################################################
    # Step 7. Save all outputs
    ############################################################################

    logger.info(f"Saving outputs to {output_dir}...")

    # Labels
    y_test.to_frame().to_parquet(output_dir / "y_test.parquet")

    # Test feature sets — saved directly so the notebook needs no models
    X_test_feats_and_text.to_parquet(output_dir / "X_test_cat_feats_and_text.parquet")
    X_test_text_only.to_parquet(output_dir / "X_test_cat_text_only.parquet")
    X_test_cat.to_parquet(output_dir / "X_test_cat.parquet")
    X_test_lr.to_parquet(output_dir / "X_test_lr.parquet")

    # Predicted probabilities
    y_prob_cat_feats_and_text.to_frame().to_parquet(
        output_dir / "y_prob_cat_feats_and_text.parquet"
    )
    y_prob_cat_text_only.to_frame().to_parquet(
        output_dir / "y_prob_cat_text_only.parquet"
    )
    y_prob_cat.to_frame().to_parquet(output_dir / "y_prob_cat.parquet")
    y_prob_lr.to_frame().to_parquet(output_dir / "y_prob_lr.parquet")

    # Optimal thresholds
    with open(output_dir / "model_thresholds.json", "w") as f:
        json.dump(model_thresholds, f, indent=2)

    # Feature names per model — JSON for human readability
    with open(output_dir / "model_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.success(
        f"All outputs saved to {output_dir}:\n"
        f"  y_test.parquet\n"
        f"  X_test_cat_feats_and_text.parquet\n"
        f"  X_test_cat_text_only.parquet\n"
        f"  X_test_cat.parquet\n"
        f"  X_test_lr.parquet\n"
        f"  y_prob_cat_feats_and_text.parquet\n"
        f"  y_prob_cat_text_only.parquet\n"
        f"  y_prob_cat.parquet\n"
        f"  y_prob_lr.parquet\n"
        f"  model_thresholds.json\n"
        f"  model_feature_names.json"
    )


if __name__ == "__main__":
    app()
