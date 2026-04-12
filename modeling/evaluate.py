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

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
)
from core.constants import mlflow_models_data
from core.functions import (
    mlflow_load_model,
    log_mlflow_metrics,
    mlflow_log_parameters_model,
    mlflow_dumpArtifact,
    return_model_metrics,
    return_model_plots,
    create_shap_plots,
)

app = typer.Typer()


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
    is_text_only: bool = False,
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
        if is_text_only:
            # text-only model: single column DataFrame is sufficient
            text_df = pd.DataFrame({text_col: list(texts)})
        else:
            # cat_text model: needs all columns — hold tabular features constant
            # at the first test row's values and vary only the summary
            ref_row = X_test.iloc[[0]].copy()
            rows = [ref_row.copy().assign(**{text_col: t}) for t in texts]
            text_df = pd.concat(rows, ignore_index=True)
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
    agg_path = None

    if not lime_df.empty:
        word_importance = (
            lime_df.groupby("word")["weight"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "mean_weight", "count": "count"})
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
    text_col: str = "summary_clean",
    n_lime_samples: int = 10,
    output_dir: Path = Path("./models/eval"),
):
    """
    Unified evaluation script for all model types:
      - lr, cat           (tabular)
      - cat_feats_and_text (text + tabular, LIME)
      - cat_text_only     (text only, LIME)
    """

    is_text_model = model_type in {"cat_feats_and_text", "cat_text_only"}
    is_text_only = model_type == "cat_text_only"

    experiment_suffix = "text_model" if is_text_model else "model"
    experiment_name = f"{outcome}_{experiment_suffix}"

    eval_dir = Path(output_dir) / outcome / model_type / pipeline_type
    eval_dir.mkdir(parents=True, exist_ok=True)

    estimator_name = model_definitions[model_type]["estimator_name"]
    run_name = f"{estimator_name}_{pipeline_type}_training"

    ############################################################################
    # Step 1. Load model from MLflow
    ############################################################################

    model = mlflow_load_model(
        experiment_name=experiment_name,
        run_name=run_name,
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
        X = X.drop(columns=[text_col], errors="ignore")

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

    inputs = {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }

    metrics = return_model_metrics(
        inputs=inputs,
        model=model,
        estimator_name=estimator_name,
    )

    all_plots = return_model_plots(
        inputs=inputs,
        model=model,
        estimator_name=estimator_name,
        scoring=scoring,
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
            is_text_only=is_text_only,
        )
        all_plots.update(lime_artifacts.get("plots", {}))

    ############################################################################
    # Step 5b. SHAP (tree-based models only)
    ############################################################################

    shap_ran = False
    shap_figs = {}
    shap_importance = None
    shap_importance_expanded = None

    if model_type in {"cat", "cat_feats_and_text", "lr"}:
        try:
            logger.info(f"Running SHAP for {model_type} — saving to {eval_dir}")
            shap_importance_expanded, shap_importance, shap_figs = create_shap_plots(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                output_dir=eval_dir,
                max_display=20,
                sample_size=200,
                side_by_side=True,
            )
            logger.info(f"SHAP figures generated: {list(shap_figs.keys())}")
            all_plots.update(shap_figs)
            shap_ran = True
            logger.success("SHAP analysis complete.")
        except Exception as e:
            logger.exception(f"SHAP failed: {e}")

    ############################################################################
    # Step 6. Save plots to eval_dir
    ############################################################################

    for fname, fig in all_plots.items():
        fig_path = eval_dir / fname
        if not fig_path.exists():
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    ############################################################################
    # Step 7. Log metrics to MLflow
    ############################################################################

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    run_id = runs.iloc[0]["run_id"]

    with mlflow.start_run(run_id=run_id):
        for split_name in ["train", "valid", "test"]:
            split_metrics = {
                k.replace(f"{split_name} ", ""): v
                for k, v in metrics[estimator_name].items()
                if k.startswith(split_name)
            }
            for metric_name, value in split_metrics.items():
                safe_name = metric_name.replace("/", "_").replace(" ", "_").lower()
                mlflow.log_metric(f"{split_name}_{safe_name}", value)

    ############################################################################
    # Step 8. Log plots to MLflow (PNG + SVG)
    ############################################################################

    # PNG plots — exclude SHAP figs here since they're logged separately in Step 9
    non_shap_plots = {k: v for k, v in all_plots.items() if not k.startswith("shap")}
    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        images=non_shap_plots,
    )

    for name, fig in non_shap_plots.items():
        mlflow_dumpArtifact(
            experiment_name=experiment_name,
            run_name=run_name,
            obj_name=Path(name).stem,
            obj=fig,
            artifacts_data_path=mlflow_models_data,
            artifact_format="svg",
        )

    ############################################################################
    # Step 9. Log SHAP artifacts to MLflow
    ############################################################################

    if shap_ran:
        # PNG plots
        log_mlflow_metrics(
            experiment_name=experiment_name,
            run_name=run_name,
            images={f"{name}.png": fig for name, fig in shap_figs.items()},
        )

        # SVG plots
        for name, fig in shap_figs.items():
            mlflow_dumpArtifact(
                experiment_name=experiment_name,
                run_name=run_name,
                obj_name=name,
                obj=fig,
                artifacts_data_path=mlflow_models_data,
                artifact_format="svg",
            )

        # Collapsed feature importance CSV
        mlflow_dumpArtifact(
            experiment_name=experiment_name,
            run_name=run_name,
            obj_name="shap_feature_importance",
            obj=shap_importance,
            artifacts_data_path=mlflow_models_data,
            artifact_format="csv",
        )

        # Expanded feature importance CSV
        if not isinstance(shap_importance_expanded, pd.DataFrame):
            shap_importance_expanded = pd.DataFrame(shap_importance_expanded)
        mlflow_dumpArtifact(
            experiment_name=experiment_name,
            run_name=run_name,
            obj_name="shap_feature_importance_expanded",
            obj=shap_importance_expanded,
            artifacts_data_path=mlflow_models_data,
            artifact_format="csv",
        )

        # Expanded beeswarm pkl for Dash app
        mlflow_dumpArtifact(
            experiment_name=experiment_name,
            run_name=run_name,
            obj_name="shap_beeswarm_expanded",
            obj=eval_dir / "shap_beeswarm_expanded.pkl",
            artifacts_data_path=mlflow_models_data,
            artifact_format="pkl",
        )

        logger.info("SHAP artifacts logged to MLflow.")

    ############################################################################
    # Step 10. Log LIME artifacts to MLflow
    ############################################################################

    if lime_artifacts:
        with mlflow.start_run(run_id=run_id):
            lime_csv = eval_dir / "lime_word_importance.csv"
            agg_csv = lime_artifacts.get("agg_path")
            if lime_csv.exists():
                mlflow.log_artifact(str(lime_csv), artifact_path="lime")
            if agg_csv and Path(agg_csv).exists():
                mlflow.log_artifact(str(agg_csv), artifact_path="lime")
            lime_dir = lime_artifacts.get("lime_dir")
            if lime_dir and Path(lime_dir).exists():
                for html_file in Path(lime_dir).glob("*.html"):
                    mlflow.log_artifact(
                        str(html_file), artifact_path="lime/explanations"
                    )
        logger.info("LIME artifacts logged to MLflow.")

    plt.close("all")
    logger.success("Evaluation complete.")


if __name__ == "__main__":
    app()
