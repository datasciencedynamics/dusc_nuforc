#!/usr/bin/env python3
"""
bootstrap_eval.py
=================
Runs bootstrapped metric evaluation across all trained models and outputs
results to a combined CSV. Uses percentile-method confidence intervals.

Models evaluated
----------------
  Tabular : lr, cat  (orig, smote, under, orig_rfe, smote_rfe, under_rfe)
  ML      : cat_feats_and_text, cat_text_only  (orig only)

Output
------
  ./models/eval/bootstrap_metrics_<text_col>[_noyear].csv

The --drop-year ablation
------------------------
The NUFORC dramatic-flag rate swings 5-fold across the study period (3.6% in
2022 vs 18.0% in 2024-25) with no corresponding change in the phenomena being
reported, so occurred_year proxies the editorial regime that labeled a report
rather than anything about the sighting itself. --drop-year 1 loads the TEXT
models trained without it and rebuilds their X to match.

Only the text models have _noyear variants. The tabular cat/lr models were
trained with occurred_year and still expect that column, so their X keeps it
even when --drop-year is set. That means the 12 tabular rows are identical
across the two output CSVs; only the two text-model rows differ.
"""

import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
)
from core.functions import mlflow_load_model
from model_tuner import evaluate_bootstrap_metrics

app = typer.Typer()

################################################################################
# Model registry -- all runs to evaluate
################################################################################

TABULAR_PIPELINES = ["orig", "smote", "under", "orig_rfe", "smote_rfe", "under_rfe"]

MODEL_RUNS = (
    # (model_type, pipeline_type)
    [(m, p) for m in ["lr", "cat"] for p in TABULAR_PIPELINES]
    + [("cat_feats_and_text", "orig"), ("cat_text_only", "orig")]
)

BOOTSTRAP_METRICS = [
    "roc_auc",
    "precision",
    "recall",
    "specificity",
    "average_precision",
    "f1_weighted",
    "neg_brier_score",
]


@app.command()
def main(
    outcome: str = "dramatic",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y_dramatic.parquet",
    text_col: str = "summary_clean",
    n_samples: int = -1,
    num_resamples: int = 5000,
    confidence_level: float = 0.95,
    output_dir: Path = Path("./models/eval"),
    output_csv: str = "bootstrap_metrics.csv",
    drop_year: int = 0,
):
    """
    Bootstrap metric evaluation across all trained models.
    Outputs a combined CSV with mean and 95% CI per metric per model.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if drop_year:
        logger.info("Ablation: text models evaluated WITHOUT occurred_year.")

    # Load features and labels once
    logger.info("Loading features and labels...")
    X_full = pd.read_parquet(features_path)
    y_full = pd.read_parquet(labels_path).squeeze()

    TEXT_COLS = {"summary_clean", "full_text_clean"}

    all_results = []

    for model_type, pipeline_type in MODEL_RUNS:

        run_label = f"{model_type}/{pipeline_type}"
        logger.info(f"Evaluating: {run_label}")

        is_text_model = model_type in {"cat_feats_and_text", "cat_text_only"}
        is_text_only = model_type == "cat_text_only"
        experiment_suffix = "text_model" if is_text_model else "model"
        experiment_name = f"{outcome}_{experiment_suffix}"
        estimator_name = model_definitions[model_type]["estimator_name"]

        # Text models are keyed by text_col AND by the year ablation, matching
        # train.py's run naming. Tabular models were never re-run without
        # occurred_year, so they carry neither tag.
        text_tag = f"_{text_col}" if is_text_model else ""
        year_tag = "_noyear" if (drop_year and is_text_model) else ""
        run_name = f"{estimator_name}_{pipeline_type}{text_tag}{year_tag}_training"

        # Load model from MLflow
        try:
            model = mlflow_load_model(
                experiment_name=experiment_name,
                run_name=run_name,
                model_name=f"{estimator_name}_{outcome}",
            )
        except Exception as e:
            logger.warning(f"  Skipping {run_label} — could not load model: {e}")
            continue

        # Prepare X for this model type (mirror train.py exactly). Whatever
        # columns train dropped, this must drop identically, or the feature
        # count will not match the fitted model.
        year_cols = ["occurred_year"] if (drop_year and is_text_model) else []

        if is_text_only:
            # occurred_year was never in the text-only X.
            X = X_full[[text_col]].copy()
            X[text_col] = X[text_col].fillna("").astype(str)
        elif is_text_model:
            other_text_cols = [
                c for c in TEXT_COLS if c != text_col and c in X_full.columns
            ]
            X = X_full.drop(columns=other_text_cols + year_cols, errors="ignore")
            X[text_col] = X[text_col].fillna("").astype(str)
        else:
            # Tabular: drop ALL text columns (tokenized AND raw) so nothing
            # string-typed reaches the numeric transformer / SMOTE.
            # occurred_year stays: these models were trained with it.
            text_all = {"summary_clean", "full_text_clean", "full_text", "summary"}
            X = X_full.drop(
                columns=[c for c in text_all if c in X_full.columns], errors="ignore"
            )

        y = y_full.copy()

        # Get test split
        try:
            X_test, y_test = model.get_test_data(X, y)
        except Exception as e:
            logger.warning(f"  Skipping {run_label} — could not get test split: {e}")
            continue

        # Get predicted probabilities
        try:
            y_pred_prob = pd.Series(
                model.predict_proba(X_test)[:, 1],
                index=X_test.index,
            )
        except Exception as e:
            logger.warning(f"  Skipping {run_label} — predict_proba failed: {e}")
            continue

        # evaluate_bootstrap_metrics defaults to threshold=0.5. The threshold is
        # not part of the fitted model, it is a post-hoc decision rule applied to
        # the predicted probabilities, and each model tuned its own during
        # training (feats+text 0.17, text_only 0.15, cat 0.22, lr 0.68). Leaving
        # the default in place scores precision/recall/specificity/F1 at a cutoff
        # nothing actually uses, which is why they disagreed with evaluate.py
        # while the threshold-free metrics (AUC, AP, Brier) matched exactly.
        model_threshold = next(iter(model.threshold.values()), 0.5)
        logger.info(f"  Threshold: {model_threshold}")

        # Run bootstrap evaluation
        try:
            boot_df = evaluate_bootstrap_metrics(
                y=y_test,
                y_pred_prob=y_pred_prob,
                n_samples=len(X_test) if n_samples == -1 else n_samples,
                num_resamples=num_resamples,
                metrics=BOOTSTRAP_METRICS,
                ci_method="percentile",
                confidence_level=confidence_level,
                model_type="classification",
                threshold=model_threshold,
            )
        except Exception as e:
            logger.warning(f"  Skipping {run_label} — bootstrap failed: {e}")
            continue

        # Tag results with model/pipeline info
        boot_df.insert(0, "pipeline_type", pipeline_type)
        boot_df.insert(0, "model_type", model_type)
        all_results.append(boot_df)
        logger.info(f"  Done: {run_label}  ({X_test.shape[1]} features)")

    if not all_results:
        logger.error("No results collected — check MLflow runs.")
        raise typer.Exit(1)

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.round(4)

    # Key the output filename by text_col and by the year ablation so the runs
    # don't overwrite each other's comparison tables.
    year_suffix = "_noyear" if drop_year else ""
    out_path = output_dir / output_csv.replace(".csv", f"_{text_col}{year_suffix}.csv")
    combined.to_csv(out_path, index=False)
    logger.success(f"Bootstrap metrics saved: {out_path}  ({len(combined):,} rows)")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    app()
