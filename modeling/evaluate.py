from pathlib import Path
import csv
import sys

csv.field_size_limit(sys.maxsize)
import typer
from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
)
from core.functions import (
    mlflow_load_model,
    log_mlflow_metrics,
    mlflow_log_parameters_model,
)

app = typer.Typer()


################################################################################
# Helpers
################################################################################


def print_confusion_matrix(tp, fp, fn, tn):
    print("-" * 80)
    print("          Predicted:")
    print("              Pos    Neg")
    print("-" * 80)
    print(f"Actual: Pos  {tp} (tp)   {fn} (fn)")
    print(f"        Neg  {fp} (fp)  {tn} (tn)")
    print("-" * 80)


def compute_metrics(y_true, y_prob, y_pred, estimator_name: str) -> pd.DataFrame:
    ap = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print_confusion_matrix(tp, fp, fn, tn)

    return pd.DataFrame(
        {
            "Metric": [
                "Precision/PPV",
                "Average Precision",
                "Sensitivity",
                "Specificity",
                "AUC ROC",
                "Brier Score",
            ],
            "Value": [
                prec,
                ap,
                rec,
                tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                auc,
                brier,
            ],
        }
    )


def make_roc_pr_plots(
    y_true, y_prob, estimator_name: str, split: str, output_dir: Path
) -> dict:
    plots = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0], name=estimator_name)
    axes[0].set_title(f"ROC Curve — {estimator_name} ({split})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)

    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, ax=axes[1], name=estimator_name
    )
    axes[1].set_title(f"Precision-Recall Curve — {estimator_name} ({split})")

    plt.tight_layout()
    plot_path = output_dir / f"{estimator_name}_{split}_roc_pr.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plots[f"{estimator_name}_{split}_roc_pr.png"] = fig
    plt.close(fig)

    return plots


################################################################################
# LIME for text models
################################################################################


def run_lime(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob_test: np.ndarray,
    text_col: str,
    estimator_name: str,
    scoring: str,
    n_lime_samples: int,
    output_dir: Path,
) -> dict:
    """Run LIME explainability on text model test set samples."""

    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.warning("lime not installed — skipping LIME. pip install lime")
        return {}

    lime_dir = output_dir / "lime_explanations"
    lime_dir.mkdir(parents=True, exist_ok=True)

    explainer = LimeTextExplainer(class_names=["Not Dramatic", "Dramatic"])

    def predict_fn(texts):
        text_df = pd.DataFrame({text_col: list(texts)})
        return model.predict_proba(text_df)

    threshold_val = model.threshold.get(scoring, 0.5)
    pos_idx = np.where(y_prob_test >= threshold_val)[0]
    neg_idx = np.where(y_prob_test < threshold_val)[0]

    n_each = max(1, n_lime_samples // 2)
    sample_pos = (
        np.random.choice(pos_idx, size=min(n_each, len(pos_idx)), replace=False)
        if len(pos_idx) > 0
        else np.array([], dtype=int)
    )
    sample_neg = (
        np.random.choice(neg_idx, size=min(n_each, len(neg_idx)), replace=False)
        if len(neg_idx) > 0
        else np.array([], dtype=int)
    )
    sample_indices = np.concatenate([sample_pos, sample_neg])

    lime_results = []
    for i, idx in enumerate(sample_indices):
        text = X_test[text_col].iloc[idx]
        true_label = y_test.iloc[idx]
        pred_proba = y_prob_test[idx]

        exp = explainer.explain_instance(
            text, predict_fn, num_features=20, num_samples=500
        )

        for word, weight in exp.as_list():
            lime_results.append(
                {
                    "sample_idx": int(idx),
                    "true_label": int(true_label),
                    "pred_proba": round(float(pred_proba), 4),
                    "word": word,
                    "weight": round(float(weight), 6),
                }
            )

        html_path = lime_dir / f"lime_sample_{i}_idx{idx}.html"
        exp.save_to_file(str(html_path))

    lime_df = pd.DataFrame(lime_results)
    lime_csv_path = output_dir / "lime_word_importance.csv"
    lime_df.to_csv(lime_csv_path, index=False)
    logger.info(f"LIME word importance saved to {lime_csv_path}")

    plots = {}
    if not lime_df.empty:
        word_importance = (
            lime_df.groupby("word")["weight"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_weight", "std": "std_weight"})
            .sort_values("mean_weight", key=abs, ascending=False)
        )

        agg_path = output_dir / "lime_aggregate_word_importance.csv"
        word_importance.to_csv(agg_path)

        print(f"\nTop 10 LIME Words ({len(sample_indices)} samples):")
        print(word_importance.head(10).to_string())

        top_n = min(20, len(word_importance))
        top_words = word_importance.head(top_n).iloc[::-1]
        colors = ["#d62728" if w > 0 else "#2ca02c" for w in top_words["mean_weight"]]

        fig1, ax1 = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax1.barh(top_words.index, top_words["mean_weight"], color=colors)
        ax1.axvline(x=0, color="black", linewidth=0.8)
        ax1.set_xlabel("Mean LIME Weight")
        ax1.set_title(
            f"Top {top_n} Words by Impact — {estimator_name}\n"
            "(Red = Dramatic / Green = Not Dramatic)"
        )
        plt.tight_layout()
        fig_path = output_dir / "lime_top_words.png"
        fig1.savefig(fig_path, dpi=150, bbox_inches="tight")
        plots["lime_top_words.png"] = fig1
        plt.close(fig1)

        risk_words = word_importance[word_importance["mean_weight"] > 0].head(top_n)
        protective_words = word_importance[word_importance["mean_weight"] < 0].head(
            top_n
        )

        if not risk_words.empty and not protective_words.empty:
            for label, words, color, fname in [
                (
                    "Dramatic Words (Increase Prediction)",
                    risk_words,
                    "#d62728",
                    "lime_dramatic_words.png",
                ),
                (
                    "Non-Dramatic Words (Decrease Prediction)",
                    protective_words,
                    "#2ca02c",
                    "lime_nondramatic_words.png",
                ),
            ]:
                wplot = words.iloc[::-1]
                fig, ax = plt.subplots(figsize=(10, max(6, len(wplot) * 0.35)))
                ax.barh(wplot.index, wplot["mean_weight"].abs(), color=color)
                ax.set_xlabel("Mean |LIME Weight|")
                ax.set_title(label)
                plt.tight_layout()
                fpath = output_dir / fname
                fig.savefig(fpath, dpi=150, bbox_inches="tight")
                plots[fname] = fig
                plt.close(fig)

    return {
        "lime_df": lime_df,
        "lime_dir": lime_dir,
        "agg_path": agg_path,
        "plots": plots,
    }


################################################################################
# LLM evaluation (loads saved predictions parquet)
################################################################################


def evaluate_llm(
    labels_path: Path,
    preds_path: Path,
    prompt_type: str,
    model_name: str,
    output_dir: Path,
    experiment_name: str,
) -> dict:
    """Evaluate pre-computed LLM predictions and log to MLflow."""

    y_true = pd.read_parquet(labels_path).squeeze().values
    preds = pd.read_parquet(preds_path)
    y_prob = preds["y_prob"].values
    y_pred = preds["y_pred"].values

    estimator_name = f"llm_{prompt_type}_{model_name.replace('-', '_')}"

    print("\n" + "*" * 80)
    print(f"Report Model Metrics: {estimator_name} — text only")
    metrics_df = compute_metrics(y_true, y_prob, y_pred, estimator_name)
    print(metrics_df.to_string(index=False))
    print("*" * 80)

    plots = make_roc_pr_plots(y_true, y_prob, estimator_name, "test", output_dir)

    metrics_dict = dict(zip(metrics_df["Metric"], metrics_df["Value"]))

    # MLflow logging
    with mlflow.start_run(run_name=f"{estimator_name}_eval", nested=False) as run:
        mlflow.set_experiment(experiment_name)
        mlflow.log_params(
            {
                "model": model_name,
                "prompt_type": prompt_type,
                "estimator_name": estimator_name,
            }
        )
        for k, v in metrics_dict.items():
            mlflow.log_metric(k.replace("/", "_").replace(" ", "_"), v)
        for fname, fig in plots.items():
            fig_path = output_dir / fname
            mlflow.log_artifact(str(fig_path), artifact_path="plots")

    logger.info(f"LLM eval logged to MLflow run {run.info.run_id}")
    return metrics_dict


################################################################################
# Main
################################################################################


@app.command()
def main(
    model_type: str = "cat",
    pipeline_type: str = "orig",
    outcome: str = "dramatic",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y_dramatic.parquet",
    scoring: str = "average_precision",
    text_col: str = "summary",
    n_lime_samples: int = 10,
    output_dir: Path = Path("./models/eval"),
    # LLM-specific args (only used when model_type == "llm")
    llm_preds_path: Path = Path(
        "./models/train/llm/llm_dramatic_preds_zero_shot.parquet"
    ),
    llm_model_name: str = "llama-3.1-8b-instant",
    llm_prompt_type: str = "zero_shot",
):
    """
    Unified evaluation script for all model types:
      - lr, cat           (tabular)
      - cat_text          (text + tabular, LIME)
      - cat_text_only     (text only, LIME)
      - llm               (pre-computed predictions)
    """

    is_text_model = model_type in {"cat_text", "cat_text_only"}
    is_llm = model_type == "llm"
    is_text_only = model_type == "cat_text_only"

    experiment_suffix = "text_model" if is_text_model else "model"
    experiment_name = f"{outcome}_{experiment_suffix}"

    eval_dir = Path(output_dir) / outcome / model_type / pipeline_type
    eval_dir.mkdir(parents=True, exist_ok=True)

    ############################################################################
    # LLM branch — load pre-computed predictions, no model object needed
    ############################################################################

    if is_llm:
        evaluate_llm(
            labels_path=labels_path,
            preds_path=llm_preds_path,
            prompt_type=llm_prompt_type,
            model_name=llm_model_name,
            output_dir=eval_dir,
            experiment_name=f"{outcome}_llm_model",
        )
        logger.success("LLM evaluation complete.")
        return

    ############################################################################
    # Step 1. Load model from MLflow
    ############################################################################

    estimator_name = model_definitions[model_type]["estimator_name"]

    model = mlflow_load_model(
        experiment_name=experiment_name,
        run_name=f"{estimator_name}_{pipeline_type}_training",
        model_name=f"{estimator_name}_{outcome}",
    )

    ############################################################################
    # Step 2. Load features and labels
    ############################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path).squeeze()

    if is_text_only:
        X = X[[text_col]].copy()
        X[text_col] = X[text_col].fillna("").astype(str)
    elif is_text_model:
        X[text_col] = X[text_col].fillna("").astype(str)
    else:
        X = X.drop(columns=["summary"], errors="ignore")

    ############################################################################
    # Step 3. Get train/valid/test splits
    ############################################################################

    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test = model.get_test_data(X, y)

    print(f"\nTrain/Valid/Test sizes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Valid: {X_valid.shape}")
    print(f"  Test:  {X_test.shape}")

    ############################################################################
    # Step 4. Compute metrics and plots for each split
    ############################################################################

    all_metrics = {}
    all_plots = {}

    for split_name, (X_sp, y_sp) in [
        ("train", (X_train, y_train)),
        ("valid", (X_valid, y_valid)),
        ("test", (X_test, y_test)),
    ]:
        y_prob = model.predict_proba(X_sp)[:, 1]
        threshold = model.threshold.get(scoring, 0.5)
        y_pred = (y_prob >= threshold).astype(int)

        print(f"\n{'*' * 80}")
        print(f"Report Model Metrics: {estimator_name} — {split_name}")
        print(f"Threshold: {threshold:.4f}")
        metrics_df = compute_metrics(y_sp.values, y_prob, y_pred, estimator_name)
        print(metrics_df.to_string(index=False))
        print("*" * 80)

        all_metrics[split_name] = dict(zip(metrics_df["Metric"], metrics_df["Value"]))
        all_plots.update(
            make_roc_pr_plots(y_sp.values, y_prob, estimator_name, split_name, eval_dir)
        )

    ############################################################################
    # Step 5. LIME (text models only)
    ############################################################################

    lime_artifacts = {}
    if is_text_model:
        y_prob_test = model.predict_proba(X_test)[:, 1]
        lime_artifacts = run_lime(
            model=model,
            X_test=X_test,
            y_test=y_test,
            y_prob_test=y_prob_test,
            text_col=text_col,
            estimator_name=estimator_name,
            scoring=scoring,
            n_lime_samples=n_lime_samples,
            output_dir=eval_dir,
        )
        all_plots.update(lime_artifacts.get("plots", {}))

    ############################################################################
    # Step 6. Log to MLflow
    # log_mlflow_metrics expects a flat DataFrame; we have a nested dict
    # (train/valid/test), so we log directly via mlflow instead.
    ############################################################################

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.mlflow.runName = '{estimator_name}_{pipeline_type}_training'"
        ),
    )
    run_id = runs.iloc[0]["run_id"]

    with mlflow.start_run(run_id=run_id):
        # Log metrics per split with prefix
        for split_name, split_metrics in all_metrics.items():
            for metric_name, value in split_metrics.items():
                safe_name = metric_name.replace("/", "_").replace(" ", "_").lower()
                mlflow.log_metric(f"{split_name}_{safe_name}", value)

        # Log ROC/PR plot images
        for fname, fig in all_plots.items():
            fig_path = eval_dir / fname
            if not fig_path.exists():
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(str(fig_path), artifact_path="plots")

    # Log LIME CSVs if present
    if lime_artifacts:
        with mlflow.start_run(run_id=run_id):
            lime_csv = eval_dir / "lime_word_importance.csv"
            agg_csv = eval_dir / "lime_aggregate_word_importance.csv"
            if lime_csv.exists():
                mlflow.log_artifact(str(lime_csv), artifact_path="lime")
            if agg_csv.exists():
                mlflow.log_artifact(str(agg_csv), artifact_path="lime")
            for html_file in lime_artifacts.get("lime_dir", Path()).glob("*.html"):
                mlflow.log_artifact(str(html_file), artifact_path="lime/explanations")

        logger.info("LIME artifacts logged to MLflow")

    plt.close("all")
    logger.success("Evaluation complete.")


if __name__ == "__main__":
    app()
