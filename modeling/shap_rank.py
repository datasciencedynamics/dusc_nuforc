#!/usr/bin/env python3
"""
shap_rank.py
============
Rank features by mean |SHAP| for the deployed CatBoost model, and report where
specific features sit in that ranking.

Reads the NATIVE CatBoost artifact written by train.py Step 11b, not the MLflow
pickle. MLflow's sklearn round-trip corrupts the CatBoost text dictionary after
catboost>=1.2.10, and a corrupted text feature silently changes the SHAP
attribution for every other feature too.

Usage
-----
    python modeling/shap_rank.py
    python modeling/shap_rank.py --text-col full_text_clean
    python modeling/shap_rank.py --text-col full_text_clean --drop-year 1
    python modeling/shap_rank.py --watch in_cluster,occurred_year,exp_certain

Why this exists
---------------
in_cluster reads as a spatial feature but its zeros are 96% "never assessed"
rather than "assessed and found isolated": DBSCAN in step_04 runs only on
US, geocoded rows, and step_04 then fills every ineligible row with 0. If
in_cluster carries SHAP weight, the model is partly keying on whether a report
was ELIGIBLE FOR MEASUREMENT, which is the same class of artifact as
occurred_year proxying the annotation regime.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import categorical_cols

app = typer.Typer()


@app.command()
def main(
    text_col: str = "summary_clean",
    drop_year: int = 0,
    model_dir: Path = Path("./models/deploy"),
    pred_dir: Path = Path("./models/predictions"),
    estimator_name: str = "cat_feats_and_text",
    watch: str = "in_cluster,occurred_year,cluster_id,exp_certain",
    top_n: int = 25,
):
    """Mean |SHAP| per feature, ranked, from the native .cbm artifact."""

    from catboost import CatBoostClassifier, Pool

    ablation_tag = "_noyear" if drop_year else ""

    ############################################################################
    # Step 1. Load the native model
    ############################################################################

    cbm_path = model_dir / f"{estimator_name}_{text_col}{ablation_tag}.cbm"
    if not cbm_path.exists():
        raise typer.BadParameter(
            f"{cbm_path} not found. train.py Step 11b writes it; re-run training "
            f"for --text-col {text_col}" + (" --drop-year 1" if drop_year else "") + "."
        )

    model = CatBoostClassifier()
    model.load_model(str(cbm_path))
    logger.success(f"Loaded {cbm_path}")

    ############################################################################
    # Step 2. Load the test features
    #
    # save_predictions.py writes these per text_col, so the subdir has to match
    # the model being explained or the feature order will not line up.
    ############################################################################

    subdir = f"{text_col}{ablation_tag}"
    x_path = pred_dir / subdir / f"X_test_{estimator_name}.parquet"
    if not x_path.exists():
        x_path = pred_dir / f"X_test_{estimator_name}.parquet"
    if not x_path.exists():
        raise typer.BadParameter(
            f"No X_test found at {pred_dir / subdir} or {pred_dir}. "
            f"Run save_predictions.py first."
        )

    X = pd.read_parquet(x_path)
    logger.info(f"Test features: {x_path}  {X.shape}")

    ############################################################################
    # Step 3. Rebuild the Pool exactly as train.py did
    #
    # Indices are positional, so they must be derived from THIS frame's column
    # order. Declaring them wrong does not error, it silently mis-attributes.
    ############################################################################

    cols = X.columns.tolist()
    X[text_col] = X[text_col].fillna("").astype(str)

    text_idx = [cols.index(text_col)]
    cat_idx = [cols.index(c) for c in categorical_cols if c in cols]

    for i in cat_idx:
        X.iloc[:, i] = X.iloc[:, i].fillna("Unknown").astype(str)

    pool = Pool(X, cat_features=cat_idx, text_features=text_idx)

    ############################################################################
    # Step 4. SHAP
    #
    # CatBoost computes exact tree SHAP natively, so the shap package is not
    # needed. Some CatBoost builds refuse ShapValues on models carrying text
    # features; PredictionValuesChange is the documented fallback and answers
    # the same ranking question, though the units differ.
    ############################################################################

    try:
        raw = model.get_feature_importance(data=pool, type="ShapValues")
        # Last column is the expected value, not a feature.
        contrib = np.abs(raw[:, :-1]).mean(axis=0)
        metric = "mean |SHAP|"
    except Exception as e:
        logger.warning(f"ShapValues unavailable ({e}); falling back.")
        contrib = model.get_feature_importance(data=pool, type="PredictionValuesChange")
        metric = "PredictionValuesChange"

    imp = (
        pd.DataFrame({"feature": cols, "importance": contrib})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp["rank"] = imp.index + 1
    imp["share_%"] = 100 * imp["importance"] / imp["importance"].sum()

    ############################################################################
    # Step 5. Report
    ############################################################################

    print()
    print("=" * 68)
    print(f"{metric} -- {estimator_name} / {text_col}{ablation_tag}")
    print(f"n = {len(X):,} test rows, {len(cols)} features")
    print("=" * 68)
    print(
        imp.head(top_n)[["rank", "feature", "importance", "share_%"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    watched = [w.strip() for w in watch.split(",") if w.strip()]
    if watched:
        print()
        print("-" * 68)
        print("Watched features")
        print("-" * 68)
        for w in watched:
            row = imp[imp["feature"] == w]
            if row.empty:
                print(f"  {w:24} not in the feature set")
                continue
            r = row.iloc[0]
            print(
                f"  {w:24} rank {int(r['rank']):>3} of {len(imp)}   "
                f"{metric} {r['importance']:.4f}   {r['share_%']:.2f}% of total"
            )

    out = pred_dir / subdir / f"shap_rank_{estimator_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out, index=False)
    print()
    logger.success(f"Full ranking -> {out}")


if __name__ == "__main__":
    app()
