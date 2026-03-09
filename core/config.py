import numpy as np

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from core.constants import (
    exp_artifact_name,
    preproc_run_name,
)
from core.functions import mlflow_loadArtifact

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR_INFER = DATA_DIR / "processed/inference"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
RESULTS_DIR = PROJ_ROOT / MODELS_DIR / "results"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

features_path = PROCESSED_DATA_DIR / "X.parquet"

################################################################################
############################ Global Constants ##################################
################################################################################

rstate = 222  # random state for reproducibility
threshold_target_metric = "precision"  # target metric for threshold optimization
target_precision = 0.5  # target precision for threshold optimization

sampler_definitions = {
    "None": None,
    "SMOTE": SMOTE(random_state=rstate),
    "RandomUnderSampler": RandomUnderSampler(random_state=rstate),
}

rfe_estimator = LogisticRegression(
    max_iter=100,
    n_jobs=-2,
)


# Remove 10% of features per iteration
rfe = RFE(
    estimator=rfe_estimator,
    step=0.1,
)


################################################################################
# Categorical and text feature columns
################################################################################

categorical_cols = ["shape", "country", "state"]
text_features = ["summary"]

# Load feature column names from MLflow
try:
    X_columns_list = mlflow_loadArtifact(
        experiment_name=exp_artifact_name,
        run_name=preproc_run_name,
        obj_name="X_columns_list",
        verbose=False,
    )
    if X_columns_list is None:
        raise ValueError("X_columns_list is None - failed to load from artifacts")
except Exception as e:
    raise Exception(f"Failed to load X_columns_list: {str(e)}")

# Numerical cols: exclude categorical and text columns
numerical_cols = [
    col
    for col in X_columns_list
    if col not in categorical_cols and col not in text_features
]


################################################################################
############################### Transformers ###################################
################################################################################

numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("imputer", SimpleImputer(strategy="mean")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
)

################################################################################
################################ Pipelines #####################################
################################################################################

pipeline_scale_imp_rfe = [
    ("Preprocessor", preprocessor),
    ("RFE", rfe),
]

pipeline_scale_imp = [
    ("Preprocessor", preprocessor),
]

pipelines = {
    "orig": {
        "pipeline": pipeline_scale_imp,
        "sampler": None,
        "feature_selection": False,
    },
    "smote": {
        "pipeline": pipeline_scale_imp,
        "sampler": SMOTE(random_state=rstate),
        "feature_selection": False,
    },
    "under": {
        "pipeline": pipeline_scale_imp,
        "sampler": RandomUnderSampler(random_state=rstate),
        "feature_selection": False,
    },
    "orig_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": None,
        "feature_selection": True,
    },
    "smote_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": SMOTE(random_state=rstate),
        "feature_selection": True,
    },
    "under_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": RandomUnderSampler(random_state=rstate),
        "feature_selection": True,
    },
}


################################################################################
############################ Logistic Regression ###############################
################################################################################

lr_name = "lr"
lr = LogisticRegression(random_state=rstate, max_iter=1000)

tuned_parameters_lr = [
    {
        "lr__C": [0.01, 0.1, 1.0, 10.0],
        "lr__penalty": ["l1", "l2"],
        "lr__solver": ["saga"],
        "lr__class_weight": [None, "balanced"],
    }
]

lr_definition = {
    "clc": lr,
    "estimator_name": lr_name,
    "tuned_parameters": tuned_parameters_lr,
    "randomized_grid": True,
    "n_iter": 20,
    "early": False,
}


################################################################################
############################### XGBoost Classifier #############################
################################################################################

xgb_name = "xgb"
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=rstate,
    tree_method="hist",
    device="cuda",
    n_jobs=16,
    enable_categorical=True,
)

xgb_parameters = [
    {
        "xgb__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
        "xgb__n_estimators": [500, 1000, 2000, 5000],
        "xgb__max_depth": [3, 5, 7, 9],
        "xgb__subsample": [0.6, 0.8, 1.0],
        "xgb__colsample_bytree": [0.6, 0.8, 1.0],
        "xgb__colsample_bylevel": [0.6, 0.8, 1.0],
        "xgb__min_child_weight": [1, 3, 5, 10],
        "xgb__gamma": [0, 0.1, 1],
        "xgb__alpha": [0, 0.1, 1, 10],
        "xgb__lambda": [0, 0.1, 10, 100],
        "xgb__early_stopping_rounds": [100, 200],
        "xgb__verbose": [0],
        "feature_selection_RFE__n_features_to_select": [10, 0.1, 0.5, 0.7, 1.0],
    }
]

xgb_definition = {
    "clc": xgb,
    "estimator_name": xgb_name,
    "tuned_parameters": xgb_parameters,
    "randomized_grid": True,
    "n_iter": 50,
    "early": True,
}


################################################################################
########################### CatBoost Tabular Classifier ########################
################################################################################

cat_name = "cat"
cat = CatBoostClassifier(
    task_type="CPU",
    random_state=rstate,
    loss_function="Logloss",
    eval_metric="AUC",
    bootstrap_type="Bernoulli",
)

cat_parameters = [
    {
        "cat__depth": [4, 6, 8, 10],
        "cat__learning_rate": [0.02, 0.03, 0.05],
        "cat__l2_leaf_reg": [3, 10, 30, 100],
        "cat__n_estimators": [3000, 6000, 10000],
        "cat__early_stopping_rounds": [75, 150],
        "cat__subsample": [0.7, 0.8, 1.0],
        "cat__min_data_in_leaf": [5, 10],
        "cat__leaf_estimation_method": ["Newton"],
        "cat__verbose": [0],
        "feature_selection_RFE__n_features_to_select": [10, 0.1, 0.5, 0.7, 1.0],
    }
]

cat_definition = {
    "clc": cat,
    "estimator_name": cat_name,
    "tuned_parameters": cat_parameters,
    "randomized_grid": True,
    "n_iter": 5,
    "early": True,
}


################################################################################
#################### CatBoost Text + Tabular Classifier #######################
################################################################################

cat_text_name = "cat_text"
cat_text = CatBoostClassifier(
    task_type="CPU",
    random_state=rstate,
    loss_function="Logloss",
    eval_metric="AUC",
    bootstrap_type="Bernoulli",
    # text_features and cat_features are passed at fit time via fit_params
)

_text_processing_options = [
    {
        "tokenizers": [{"tokenizer_id": "Space", "token_types": ["Word"]}],
        "dictionaries": [{"dictionary_id": "Word", "max_dictionary_size": "50000"}],
        "feature_processing": {
            "default": [{"dictionaries_names": ["Word"], "feature_calcers": ["BoW"]}]
        },
    },
    {
        "tokenizers": [{"tokenizer_id": "Space", "token_types": ["Word"]}],
        "dictionaries": [{"dictionary_id": "Word", "max_dictionary_size": "10000"}],
        "feature_processing": {
            "default": [{"dictionaries_names": ["Word"], "feature_calcers": ["BM25"]}]
        },
    },
    {
        "tokenizers": [{"tokenizer_id": "Space", "token_types": ["Word"]}],
        "dictionaries": [{"dictionary_id": "Word", "max_dictionary_size": "50000"}],
        "feature_processing": {
            "default": [
                {"dictionaries_names": ["Word"], "feature_calcers": ["BoW", "BM25"]}
            ]
        },
    },
    {
        "tokenizers": [{"tokenizer_id": "Space", "token_types": ["Word"]}],
        "dictionaries": [{"dictionary_id": "Word", "max_dictionary_size": "50000"}],
        "feature_processing": {
            "default": [
                {
                    "dictionaries_names": ["Word"],
                    "feature_calcers": ["BoW", "NaiveBayes", "BM25"],
                }
            ]
        },
    },
]

cat_text_parameters = [
    {
        "cat_text__text_processing": _text_processing_options,
        "cat_text__depth": [4, 6, 8],
        "cat_text__learning_rate": [0.02, 0.03, 0.05],
        "cat_text__l2_leaf_reg": [3, 10, 30],
        "cat_text__n_estimators": [3000, 6000, 10000],
        "cat_text__early_stopping_rounds": [75, 150],
        "cat_text__subsample": [0.7, 0.8, 1.0],
        "cat_text__min_data_in_leaf": [5, 10],
        "cat_text__leaf_estimation_method": ["Newton"],
        "cat_text__verbose": [0],
    }
]

cat_text_definition = {
    "clc": cat_text,
    "estimator_name": cat_text_name,
    "tuned_parameters": cat_text_parameters,
    "randomized_grid": True,
    "n_iter": 5,
    "early": True,
}


################################################################################
####################### CatBoost Text-Only Classifier #########################
################################################################################

cat_text_only_name = "cat_text_only"
cat_text_only = CatBoostClassifier(
    task_type="CPU",
    random_state=rstate,
    loss_function="Logloss",
    eval_metric="AUC",
    bootstrap_type="Bernoulli",
    # text_features passed at fit time via fit_params; no cat_features
)

cat_text_only_parameters = [
    {
        "cat_text_only__text_processing": _text_processing_options,
        "cat_text_only__depth": [4, 6, 8],
        "cat_text_only__learning_rate": [0.02, 0.03, 0.05],
        "cat_text_only__l2_leaf_reg": [3, 10, 30],
        "cat_text_only__n_estimators": [3000, 6000, 10000],
        "cat_text_only__early_stopping_rounds": [75, 150],
        "cat_text_only__subsample": [0.7, 0.8, 1.0],
        "cat_text_only__min_data_in_leaf": [5, 10],
        "cat_text_only__leaf_estimation_method": ["Newton"],
        "cat_text_only__verbose": [0],
    }
]

cat_text_only_definition = {
    "clc": cat_text_only,
    "estimator_name": cat_text_only_name,
    "tuned_parameters": cat_text_only_parameters,
    "randomized_grid": True,
    "n_iter": 5,
    "early": True,
}


################################################################################
########################### Model Definitions Map ##############################
################################################################################

model_definitions = {
    lr_name: lr_definition,
    xgb_name: xgb_definition,
    cat_name: cat_definition,
    cat_text_name: cat_text_definition,
    cat_text_only_name: cat_text_only_definition,
}
