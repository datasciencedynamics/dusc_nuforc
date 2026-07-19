from pathlib import Path
import csv
import json
import sys

csv.field_size_limit(sys.maxsize)

import typer
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import PROCESSED_DATA_DIR

from bertuner.BERTuner import BERTuneClassifier
from bertuner.constants import DEFAULT_SEARCH_SPACE_SINGLELABEL

app = typer.Typer()


################################################################################
# Model choices and search space
################################################################################

# Base-size encoders only. Narratives have a median of 42 whitespace tokens and
# a 99th percentile of 266, so large models add parameters without adding
# context they can use. ModernBERT and the 8192-token path are deliberately
# excluded: only 0.15% of reports exceed 512 tokens, so long-context attention
# would be spent almost entirely on padding.
MODEL_CHOICES = {
    "bert-base": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "deberta-v3-base": "microsoft/deberta-v3-base",
}

# Deviations from DEFAULT_SEARCH_SPACE_SINGLELABEL:
#   learning_rate floor raised to 1e-5 -- sub-1e-5 rates on a base encoder with
#     ~12k training rows mostly waste trials.
#   loss_type drops label_smoothing -- the positive rate is 11.3%, and smoothing
#     works against rare positives.
SEARCH_SPACE = {
    **DEFAULT_SEARCH_SPACE_SINGLELABEL,
    "model": list(MODEL_CHOICES.keys()),
    "learning_rate": {"low": 1e-5, "high": 5e-5, "log": True},
    "batch_size": [16, 32],
    "loss_type": ["weighted", "focal"],
}


################################################################################
# Main
################################################################################


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "df_final.parquet",
    models_dir: Path = Path("./models"),
    text_col: str = "full_text_clean",
    outcome: str = "dramatic",
    max_length: int = 512,
    n_trials: int = 15,
    scoring: str = "avg_precision",
    study_name: str = "",
    tracking_uri: str = "./mlruns",
    output_dir: Path = Path("./models/eval"),
    skip_optimize: int = 0,
):
    """
    Fine-tune a transformer classifier on NUFORC narratives via BERTuner.

    Runs Optuna hyperparameter search (logged to MLflow), then retrains on the
    best parameters, tunes the decision threshold on the validation set, and
    evaluates on the test set. Model, tokenizer, and bertuner_config.json are
    written under models_dir/final_model/model.

    Uses full_text_clean so the comparison against cat_text_only and
    cat_feats_and_text is controlled: all three models see identical input.
    Note that the cleaned column is stopword-stripped and detokenized, so the
    transformer cannot exploit word order or function words here. Any narrowing
    of the gap against the CatBoost text models should be read in that light.

    Pass --skip-optimize 1 to retrain from an existing study without repeating
    the search.
    """

    ############################################################################
    # Step 1. Resolve run naming
    ############################################################################

    # Mirrors train.py: runs are keyed by text column so different text sources
    # never overwrite each other in MLflow or on disk.
    study_name = study_name or f"bert_{outcome}_{text_col}"

    eval_dir = Path(output_dir) / outcome / "bert" / text_col
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Study:        {study_name}")
    logger.info(f"Features:     {features_path}")
    logger.info(f"Text col:     {text_col}")
    logger.info(f"Outcome:      {outcome}")
    logger.info(f"Max length:   {max_length}")
    logger.info(f"Tracking URI: {tracking_uri}")

    ############################################################################
    # Step 2. Load data
    # BERTuner's data_path argument calls pd.read_csv unconditionally, with no
    # format detection, so a parquet path fails on the PAR1 magic bytes. Read it
    # here and hand over a DataFrame instead. Parquet is preferred: CSV
    # round-trips of free text are fragile around embedded quotes and newlines.
    ############################################################################

    if features_path.suffix.lower() == ".csv":
        logger.warning(
            "Reading CSV. Prefer the parquet: CSV round-trips of narrative "
            "text can silently shift rows on embedded newlines or quotes."
        )
        df = pd.read_csv(features_path)
    else:
        df = pd.read_parquet(features_path)

    missing = [c for c in (text_col, outcome) if c not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"Column(s) {missing} not found in {features_path}. "
            f"Available: {sorted(df.columns.tolist())}"
        )

    # Keep only what the classifier needs. Passing the full frame invites
    # BERTuner to trip over unrelated dtypes, and the index (report_id) is not
    # a feature.
    df = df[[text_col, outcome]].reset_index(drop=True)
    df[text_col] = df[text_col].fillna("").astype(str)

    ############################################################################
    # Step 2b. Input audit
    # Empty narratives tokenize to [CLS] [SEP] and yield a near-constant
    # prediction. They are not dropped, but the count belongs in the record.
    ############################################################################

    n_total = len(df)
    n_empty = int(df[text_col].str.strip().eq("").sum())
    pos_rate = float(df[outcome].mean())
    tok_counts = df[text_col].str.split().str.len()

    print()
    print("=" * 60)
    print("Input audit")
    print("=" * 60)
    print(f"  rows:            {n_total}")
    print(f"  empty narrative: {n_empty} ({n_empty / n_total:.2%})")
    print(f"  positive rate:   {pos_rate:.2%}")
    print(
        f"  tokens p50/p95/p99/max: "
        f"{tok_counts.quantile(.50):.0f} / "
        f"{tok_counts.quantile(.95):.0f} / "
        f"{tok_counts.quantile(.99):.0f} / "
        f"{tok_counts.max():.0f}"
    )
    print()

    if n_empty:
        logger.warning(
            f"{n_empty} rows have an empty {text_col} and will train and "
            f"score on [CLS] [SEP] alone."
        )

    ############################################################################
    # Step 3. Initialize classifier
    ############################################################################

    classifier = BERTuneClassifier(
        dataframe=df,
        models_dir=str(models_dir),
        text_feature=text_col,
        target_cols=[outcome],
        max_length=max_length,
        mlflow_tracking_uri=tracking_uri,
    )

    ############################################################################
    # Step 4. Configure model choices and search space
    ############################################################################

    classifier.initialize_model_choices(MODEL_CHOICES)
    classifier.initialize_search_space(SEARCH_SPACE)

    ############################################################################
    # Step 5. Hyperparameter optimization
    ############################################################################

    best_value = None
    if not skip_optimize:
        logger.info(f"Running {n_trials} Optuna trials on {scoring} ...")
        best_value = classifier.optimize(
            n_trials=n_trials,
            optimize_metric=scoring,
            study_name=study_name,
        )
        logger.success(f"Best {scoring}: {best_value}")
    else:
        logger.info("skip_optimize set, reusing existing study parameters.")

    ############################################################################
    # Step 6. Train final model
    ############################################################################

    metrics, model, test_ds = classifier.train_final_model()

    print()
    print("=" * 60)
    print(f"{outcome} -- final model metrics")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

    ############################################################################
    # Step 7. Persist metrics alongside the other eval artifacts
    ############################################################################

    def _coerce(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    metrics_path = eval_dir / "bert_metrics.json"
    payload = {
        "study_name": study_name,
        "text_col": text_col,
        "outcome": outcome,
        "max_length": max_length,
        "n_trials": None if skip_optimize else n_trials,
        "scoring": scoring,
        "best_value": best_value,
        "model_choices": MODEL_CHOICES,
        "n_rows": n_total,
        "n_empty_narrative": n_empty,
        "positive_rate": pos_rate,
        "metrics": {k: _coerce(v) for k, v in metrics.items()},
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.success(f"Metrics written to {metrics_path}")
    logger.success("BERT training complete.")


if __name__ == "__main__":
    app()
