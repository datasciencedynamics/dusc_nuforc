from pathlib import Path
import csv
import sys

csv.field_size_limit(sys.maxsize)
import typer
from loguru import logger
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV
from model_tuner import Model
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

################################################################################
# Step 1. Import Configurations and Constants
################################################################################

from core.constants import var_index, drop_vars

from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
    rstate,
    pipelines,
    numerical_cols,
    categorical_cols,
)
from core.functions import (
    clean_feature_selection_params,
    mlflow_log_parameters_model,
    adjust_preprocessing_pipeline,
    mlflow_load_model,
)

app = typer.Typer()


################################################################################
# Step 2. Define CLI Arguments with Default Values
################################################################################


@app.command()
def main(
    model_type: str = "lr",
    pipeline_type: str = "orig",
    outcome: str = "dramatic",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Optional[Path] = None,  # derived from outcome if not passed
    text_col: str = "summary_clean",
    scoring: str = "average_precision",
    pretrained: int = 0,
):
    # Derive labels_path from outcome if not explicitly provided
    if labels_path is None:
        labels_path = PROCESSED_DATA_DIR / f"y_{outcome}.parquet"

    is_text_model = model_type in {"cat_feats_and_text", "cat_text_only"}
    is_text_only = model_type == "cat_text_only"

    ################################################################################
    # Step 3. Load Feature and Label Datasets
    ################################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path).squeeze()

    if is_text_only:
        # Text-only ablation: keep nothing but summary
        X = X[[text_col]].copy()
        X[text_col] = X[text_col].fillna("").astype(str)
    elif is_text_model:
        # Text + tabular: drop drop_vars (e.g. city) but retain numeric/cat cols
        cols_to_drop = [c for c in drop_vars if c in X.columns]
        X = X.drop(columns=cols_to_drop, errors="ignore")
        X[text_col] = X[text_col].fillna("").astype(str)
    else:
        # Tabular only: drop raw text column
        X = X.drop(columns=["summary_clean"], errors="ignore")

    ################################################################################
    # Step 4. Retrieve Model and Pipeline Configurations
    ################################################################################

    clc = model_definitions[model_type]["clc"]
    estimator_name = model_definitions[model_type]["estimator_name"]
    tuned_parameters = model_definitions[model_type]["tuned_parameters"]
    randomized_grid = model_definitions[model_type]["randomized_grid"]
    n_iter = model_definitions[model_type]["n_iter"]
    early_stop = model_definitions[model_type]["early"]

    if is_text_model:
        # Passthrough pipeline — CatBoost handles text natively via fit_params
        pipeline_steps = [
            (
                "Preprocessor",
                FunctionTransformer(func=lambda x: x, feature_names_out="one-to-one"),
            ),
        ]
        sampler = None
        feature_selection = False
    else:
        pipeline_steps = pipelines[pipeline_type]["pipeline"]
        sampler = pipelines[pipeline_type]["sampler"]
        feature_selection = pipelines[pipeline_type]["feature_selection"]
        print("Sampler:", sampler)

    ################################################################################
    # Step 5. Clean up pipeline
    # Step 5a. Remove feature selection keys if RFE isn't in the pipeline
    ################################################################################

    if not is_text_model:
        clean_feature_selection_params(pipeline_steps, tuned_parameters)

        # Step 5b. Skip imputer/scaler for tree-based models
        pipeline_steps = adjust_preprocessing_pipeline(
            model_type,
            pipeline_steps,
            numerical_cols,
            categorical_cols,
            sampler=sampler,
        )

    ################################################################################
    # Step 6. No demographic stratification for NUFORC
    # stratify_y=True handles outcome-based stratification in model_tuner
    ################################################################################

    stratify_df = None

    ################################################################################
    # Step 6a. Print outcome
    ################################################################################

    print()
    print(f"Outcome:")
    print("-" * 60)
    print()
    print("=" * 60)
    print(f"{outcome}")
    print("=" * 60)

    ################################################################################
    # Step 6b. Build fit_params for CatBoost text
    # Indices are resolved dynamically after drop_vars have been removed from X.
    # text_features: [summary_clean index]
    # cat_features:  [state, country, shape indices] — string cols that are NOT
    #                text features; must be declared or CatBoost tries float parsing
    ################################################################################

    fit_params = {}
    if is_text_only:
        fit_params[f"{estimator_name}__text_features"] = [0]
    elif is_text_model:
        col_list = X.columns.tolist()
        text_col_idx = col_list.index(text_col)
        cat_col_indices = [col_list.index(c) for c in categorical_cols if c in col_list]
        fit_params[f"{estimator_name}__text_features"] = [text_col_idx]
        fit_params[f"{estimator_name}__cat_features"] = cat_col_indices
    elif model_type == "cat":
        if pipeline_type in {"orig", "under"}:
            # CT outputs string array; resampler preserves dtype --> cat_features needed
            n_num = len([c for c in numerical_cols if c in X.columns])
            cat_col_indices = list(range(n_num, n_num + len(categorical_cols)))
            fit_params[f"{estimator_name}__cat_features"] = cat_col_indices
        # smote: SMOTE interpolation converts strings --> float --> cat_features invalid
        # *_rfe: OHE encodes categoricals --> float --> cat_features invalid
    ################################################################################
    # Step 7. Define and Initialize the Model Pipeline
    ################################################################################

    logger.info(f"Training {estimator_name} for {outcome} ...")

    experiment_suffix = "text_model" if is_text_model else "model"

    if pretrained:
        print("Loading Pretrained Model...")
        model = mlflow_load_model(
            experiment_name=f"{outcome}_{experiment_suffix}",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}",
        )

    else:
        model = Model(
            pipeline_steps=pipeline_steps,
            name=estimator_name,
            model_type="classification",
            estimator_name=estimator_name,
            calibrate=not is_text_model,
            estimator=clc,
            kfold=False,
            grid=tuned_parameters,
            n_jobs=5,
            randomized_grid=randomized_grid,
            n_iter=n_iter,
            scoring=[scoring],
            random_state=rstate,
            stratify_cols=stratify_df,  # None — no demographic stratification
            stratify_y=True,  # stratify splits by outcome
            boost_early=early_stop,
            imbalance_sampler=sampler,
            feature_selection=feature_selection,
        )

        ################################################################################
        # Step 8. Hyperparameter Tuning
        ################################################################################

        model.grid_search_param_tuning(
            X,
            y,
            f1_beta_tune=True,
            betas=[1],
            fit_params=fit_params,
        )

    ################################################################################
    # Step 9. Extract Training, Validation, and Test Splits
    ################################################################################

    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, _ = model.get_test_data(X, y)

    print(f"Train/Valid/Test sizes:")
    print(X_train.shape, X_valid.shape, X_test.shape)
    print(
        f"Total Train_Val_Test size: "
        f"{X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]}"
    )

    ################################################################################
    # Step 10. Train the Model
    ################################################################################

    if not pretrained:
        if model_type in {"xgb", "cat"} or is_text_model:
            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                score=scoring,
                fit_params=fit_params,
            )
        else:
            model.fit(
                X_train,
                y_train,
                score=scoring,
            )

    ################################################################################
    # Step 11. Calibrate the Model
    ################################################################################

    if is_text_model:
        # Manual calibration: model_tuner's calibrateModel can't route fit_params
        # through CalibratedClassifierCV cleanly. The model is already fitted in
        # Step 10, so cv="prefit" just adjusts probabilities on the validation set.
        model.estimator = CalibratedClassifierCV(
            model.estimator, cv="prefit", method="sigmoid"
        ).fit(X_valid, y_valid)
    elif model_type in {"xgb", "cat"}:
        if model.calibrate:
            model.calibrateModel(X, y, fit_params=fit_params)
    else:
        if model.calibrate:
            model.calibrateModel(X, y, score=scoring)

    ################################################################################
    # Step 12. Evaluate and Log to MLflow
    ################################################################################

    model.return_metrics(
        X=X_valid,
        y=y_valid,
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )

    if pretrained:
        mlflow_log_parameters_model(
            experiment_name=f"{outcome}_{experiment_suffix}",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}",
            model=model,
        )
    else:
        mlflow_log_parameters_model(
            model_type=model_type,
            n_iter=n_iter,
            kfold=False,
            outcome=outcome,
            experiment_name=f"{outcome}_{experiment_suffix}",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}",
            model=model,
            hyperparam_dict=model.best_params_per_score[scoring],
        )

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
