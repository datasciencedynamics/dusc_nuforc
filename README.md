# NUFORC Media

A machine learning pipeline for classifying reports from the National UFO Reporting Center (NUFORC) by narrative *dramaticness*, a measure of how vivid, detailed, and extraordinary a witness account is. The project combines structured features, free-text NLP, gradient-boosted models, an LLM baseline, and SHAP-based explainability behind a deployed dashboard.

## What this project does

The NUFORC database contains decades of public UFO sighting reports submitted by witnesses across the United States and abroad. Most reports describe brief, mundane observations (lights in the sky, ambiguous shapes), while a small minority are highly dramatic narratives describing structured craft, occupants, sustained encounters, or other extraordinary content. This project builds models that score each report on a dramaticness scale and explains *why* a given report received the score it did.

The pipeline:

1. Ingests scraped NUFORC report data.
2. Engineers a combined set of structured and NLP-derived features from each report's free-text summary.
3. Trains and tunes several model families: logistic regression, CatBoost on tabular features, CatBoost on text features, CatBoost combining both, and a zero-/few-shot LLM classification baseline.
4. Evaluates models with stratified cross-validation, average-precision scoring, and bootstrap confidence intervals.
5. Generates SHAP explanations for individual predictions.
6. Serves predictions and explanations through a live dashboard.

The work extends RAND's 2023 report *Not the X-Files*, which analyzed geographic and temporal patterns in NUFORC reports, by adding a content-aware dimension grounded in the language of the reports themselves.

## Live application

A live version of the dashboard is deployed at:

**[apps.datasciencedynamics.com/uap_classifier](https://apps.datasciencedynamics.com/uap_classifier)**

The app is built on a Flask/Dash WSGI dispatcher (entry point: `app.py`) and lets users browse scored reports, inspect per-report SHAP explanations, and explore aggregate patterns in dramaticness across regions, shapes, and report years.

## Models

| Model key            | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| `lr`                 | Logistic regression on tabular features (baseline)                   |
| `cat`                | CatBoost on tabular features                                         |
| `cat_text_only`      | CatBoost on free-text features only                                  |
| `cat_feats_and_text` | CatBoost combining tabular and text features                         |
| `train_llm`          | Zero-shot and few-shot LLM classification baseline                   |

Each tabular model can be run under six pipeline variants that combine class-imbalance handling (`orig`, `smote`, `under`) with optional recursive feature elimination (`_rfe`). All runs are tracked with MLflow.

## Project structure

```
nuforc_media/
├── app.py                      # Flask/Dash entry point for the dashboard
├── core/                       # Shared config, constants, and utility functions
│   ├── config.py
│   ├── constants.py
│   └── functions.py
├── preprocessing/              # Data ingestion and feature engineering
│   ├── 1_data_gen.py
│   ├── 2_nlp_feature_engineer_nuforc.py
│   ├── 3_nuforc_analytics.py
│   ├── 4_preprocessing_remaining_feats.py
│   └── 5_feat_gen.py
├── modeling/                   # Training, evaluation, explanation, inference
│   ├── train.py                # LR + CatBoost training across pipeline variants
│   ├── train_llm.py            # Zero-/few-shot LLM baseline
│   ├── evaluate.py
│   ├── bootstrap_evaluation.py
│   ├── save_predictions.py
│   ├── explainer.py            # SHAP explainer fitting
│   └── explanations_training.py
├── notebooks/
│   ├── raw_data_exploration.ipynb
│   ├── data_exploration.ipynb
│   └── performance_assessment.ipynb
├── models/                     # Trained models, predictions, evaluation artifacts
│   ├── eval/
│   ├── predictions/
│   ├── results/
│   └── train/
├── data/                       # Raw, interim, processed datasets (gitignored)
├── mlruns/                     # MLflow tracking store
├── Makefile                    # Pipeline orchestration
├── requirements.txt
└── setup.py
```

## Setup

Requires Python 3.12.

```bash
# Create and activate a virtual environment
python -m venv nuforc_venv
source nuforc_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Running the pipeline

The full pipeline is orchestrated through the `Makefile`. A typical end-to-end workflow:

```bash
# 1. Preprocessing: ingest, NLP feature engineering, analytics, feature generation
make preproc_pipeline

# 2. Train all models (LR, CatBoost variants, text-only, combined)
make train_all_models

# 3. Evaluate models and bootstrap confidence intervals
make eval_all_models
make bootstrap_eval

# 4. Fit SHAP explainer and generate per-report explanations
make model_explaining_training

# Inspect MLflow runs
make mlflow_ui
```

For inference on a new batch of reports:

```bash
make preproc_pipeline_inference
```

Run `make help` for a full list of available targets.

## Data

Source reports come from the [National UFO Reporting Center](https://nuforc.org). Note that the NUFORC site renders its tables via a JavaScript wpDataTables plugin, so direct `pandas.read_html()` does not work. Ingestion iterates the static per-month subindex pages at `nuforc.org/ndx/?id=event` with rate limiting.

Raw and processed data files are gitignored.

## Authors

<table>
  <tr>
    <td width="160" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/main/photos/leonshpaner.jpg" width="140" alt="Leon Shpaner">
    </td>
    <td valign="top">
      <b>Leon Shpaner, M.S.</b><br><br>
      Leon is a Data Scientist at UCLA Health with over 15 years of experience across healthcare, financial services, and education. He serves as an adjunct professor at the University of San Diego, where he teaches statistics and machine learning in the M.S. in Applied Artificial Intelligence program. He has contributed to clinical prediction research, co-developed a production-grade EDA toolkit contracted for publication with Taylor &amp; Francis, and presented at JupyterCon 2025.
    </td>
  </tr>
  <tr>
    <td width="160" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/main/photos/Oscar_LinkedIn_Pic.jpeg" width="140" alt="Oscar Gil">
    </td>
    <td valign="top">
      <b>Oscar Gil, M.S.</b><br><br>
      Oscar is a Data Scientist at the University of California, Riverside, with over ten years of experience in the education data management industry. He excels in data warehousing, analytics, machine learning, SQL, Python, R, and report authoring, and holds an M.S. in Applied Data Science from the University of San Diego. He has co-developed analytical tools and pipelines deployed in research and institutional settings, and presented alongside Leon at JupyterCon 2025.
    </td>
  </tr>
</table>

Data Science Dynamics: [datasciencedynamics.com](https://datasciencedynamics.com)

## References

- Gromis, A. et al. (2023). *Not the X-Files: An Analysis of UFO Reporting in the United States.* RAND Corporation.
- National UFO Reporting Center: [nuforc.org](https://nuforc.org)

## License

TBD.