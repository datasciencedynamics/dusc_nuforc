<p align="left">
  <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/refs/heads/main/data_science_dynamics_logo.svg" alt="Data Science Dynamics" width="250"/>
</p>

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td valign="middle">
      <h1>Dramatic and Unusual UAP Sightings Classifier</h1>
      <p>A machine learning pipeline for classifying reports from the National UFO Reporting Center (NUFORC) by narrative <em>dramaticness</em>, a measure of how vivid, detailed, and extraordinary a witness account is. The project combines structured features, free-text NLP, gradient-boosted models, an LLM baseline, and SHAP-based explainability behind a deployed dashboard.</p>
    </td>
    <td valign="middle" width="300" align="center">
      <img src="https://raw.githubusercontent.com/datasciencedynamics/dusc_nuforc/refs/heads/main/assets/dusc_logo.svg" alt="DUSC" width="300"/>
    </td>
  </tr>
</table>

## Data attribution

> **Data used in this study was provided by the National UFO Reporting Center (NUFORC) and is used with NUFORC's special permission.**

This repository does not redistribute NUFORC report data. No report text, no
database export, and no raw report files are committed here; the `data/`
directory is gitignored in full.

**Please do not scrape nuforc.org.** NUFORC is a small organization and
automated traffic is a real burden on it. Direct data requests to NUFORC at
[nuforc.org](https://nuforc.org). The permission covering this project is
specific to Data Science Dynamics and does not extend to users of this
repository.

## What this project does

The NUFORC database contains decades of public UFO sighting reports submitted by witnesses across the United States and abroad. Most reports describe brief, mundane observations (lights in the sky, ambiguous shapes), while a small minority are highly dramatic narratives describing structured craft, occupants, sustained encounters, or other extraordinary content. This project builds models that score each report on a dramaticness scale and explains *why* a given report received the score it did.

The pipeline:

1. Ingests NUFORC report data obtained under the permission described above.
2. Engineers a combined set of structured and NLP-derived features from each report's summary and full witness narrative.
3. Trains and tunes several model families: logistic regression, CatBoost on tabular features, CatBoost on text features, CatBoost combining both, and a zero-/few-shot LLM classification baseline.
4. Evaluates models with stratified splits, average-precision scoring, and bootstrap confidence intervals.
5. Generates SHAP and LIME explanations for individual predictions.
6. Serves predictions and explanations through a live dashboard.

The work extends RAND's 2023 report *Not the X-Files*, which analyzed geographic and temporal patterns in NUFORC reports, by adding a content-aware dimension grounded in the language of the reports themselves.

## Live applications

Two applications are deployed:

**[apps.datasciencedynamics.com/uap_classifier](https://apps.datasciencedynamics.com/uap_classifier)**
Scores a sighting description that the visitor types in, estimating the
probability that NUFORC would mark the report tier 1 or tier 2, and shows a LIME
explanation of which words moved the score. It publishes no NUFORC reports.

**[apps.datasciencedynamics.com/dusc_dash](https://apps.datasciencedynamics.com/dusc_dash)**
Model performance dashboard: ROC and precision-recall curves, threshold sweep,
calibration, subgroup performance, the year ablation, pairwise DeLong tests, and
a map of model-versus-label agreement on the held-out test sample. The map shows
approximate city-level coordinates and links back to the corresponding NUFORC
reports; it does not reproduce report narratives.

Both run in a single process behind a Flask/Dash WSGI dispatcher.

## Models

| Model key            | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| `lr`                 | Logistic regression on tabular features (baseline)                   |
| `cat`                | CatBoost on tabular features                                         |
| `cat_text_only`      | CatBoost on free-text features only                                  |
| `cat_feats_and_text` | CatBoost combining tabular and text features                         |
| `train_llm`          | Zero-shot and few-shot LLM classification baseline                   |

Each tabular model can be run under six pipeline variants that combine class-imbalance handling (`orig`, `smote`, `under`) with optional recursive feature elimination (`_rfe`). All runs are tracked with MLflow.

## Configuration

Pipeline behavior is controlled by a small set of Makefile variables. Each can be
overridden on the command line, and each propagates into MLflow run names, log
filenames, and evaluation output directories so that variants never collide.

| Variable    | Default           | Purpose                                                                 |
|-------------|-------------------|-------------------------------------------------------------------------|
| `OUTCOME`   | `dramatic`        | Target label; also selects `data/processed/y_<outcome>.parquet`         |
| `TEXT_COL`  | `full_text_clean` | Which text feature the text models consume                              |
| `DROP_YEAR` | `0`               | `1` excludes `occurred_year` from the feature matrix (see Ablations)    |
| `PIPELINES` | six variants      | Imbalance and RFE combinations applied to the tabular models            |
| `SCORING`   | `average_precision` | Tuning and threshold-selection metric                                 |
| `GEO_OUT`   | `./geo_maps`      | Destination root for downloaded map layers (see Geospatial assets)      |

### Text column selection

Text models are keyed by `TEXT_COL`, so `summary_clean` and `full_text_clean`
runs are trained, logged, and evaluated independently rather than overwriting
each other:

```bash
make train_cat_feats_and_text TEXT_COL=summary_clean
make train_cat_feats_and_text TEXT_COL=full_text_clean
```

Tabular models (`lr`, `cat`) drop all text columns and are therefore identical
regardless of `TEXT_COL`; they carry no text tag in their MLflow run names.

### Ablations

`occurred_year` is the single strongest SHAP feature, but the dramatic base rate
swings from 3.6% in 2022 to 18.0% in 2024 to 2025. Per NUFORC, that reflects
changes in their review policy rather than anything about the sightings
themselves: systematic tier 1 review began for reports submitted on or after
17 March 2023, tier 2 was added as a second classification layer in October
2024, and earlier reports were reviewed only when a specific case resurfaced.
Setting `DROP_YEAR=1` excludes the column so the cost of that artifact can be
quantified directly. Ablated runs receive a `_noyear` suffix throughout.

## Project structure

```
dusc_nuforc/
├── core/                       # Shared config, constants, and utility functions
│   ├── config.py
│   ├── constants.py
│   └── functions.py
├── preprocessing/              # Ingestion and feature engineering
│   ├── step_00_NUFORC_Extractor.py
│   ├── step_01_data_gen.py
│   ├── step_03_nlp_feature_engineer_nuforc.py
│   ├── step_04_nuforc_analytics.py
│   ├── step_05_preprocessing_remaining_feats.py
│   ├── step_06_feat_gen.py
│   ├── step_07_build_eda_frame.py      # Joins raw and model frames for EDA
│   └── step_08_download_geo_maps.py    # Fetches basemap shapefiles
├── debug_scripts/              # One-off maintenance, not part of the pipeline
│   ├── backfill_summary_text.py
│   ├── audit_truncation.py
│   ├── prune_truncated_checkpoint.py
│   ├── prune_all_questionable.py
│   ├── summary_equals_full_text.py
│   └── verify_fix.py
├── modeling/                   # Training, evaluation, explanation, inference
│   ├── train.py                # LR + CatBoost training across pipeline variants
│   ├── train_llm.py            # Zero-/few-shot LLM baseline
│   ├── evaluate.py             # Metrics, plots, SHAP, LIME
│   ├── bootstrap_evaluation.py
│   ├── save_predictions.py
│   ├── explainer.py            # SHAP explainer fitting
│   └── explanations_training.py
├── notebooks/
│   ├── raw_data_exploration.ipynb
│   ├── data_exploration.ipynb  # Choropleths and point maps, needs geo_maps/
│   └── performance_assessment.ipynb
├── models/                     # Trained models, predictions, evaluation artifacts
│   ├── deploy/                 # Native CatBoost .cbm + calibration JSON
│   ├── eval/
│   ├── predictions/
│   └── results/
├── data/                       # Raw, interim, processed datasets (gitignored)
├── geo_maps/                   # Basemap shapefiles (gitignored, see below)
│   ├── us/                     # Census TIGER state boundaries
│   └── world/                  # Natural Earth country boundaries
├── mlruns/                     # MLflow tracking store
├── Makefile                    # Pipeline orchestration
├── requirements.txt
└── setup.py
```

The deployed Flask/Dash applications live in a separate repository; this one
covers the data pipeline, modeling, and evaluation.

## Setup

Requires Python 3.12.

```bash
# Create and activate a virtual environment
python -m venv nuforc_venv
source nuforc_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Fetch the basemap shapefiles the EDA notebooks need
make download_geo_maps
```

## Geospatial assets

The mapping cells in `notebooks/data_exploration.ipynb` draw sightings against
US state and world country boundaries. Those basemaps are public-domain
shapefiles from Census TIGER and Natural Earth, and they are **gitignored**
rather than committed. Shapefiles are binary, so git stores each revision more
or less in full and cannot delta-compress them; committing them inflates the
repository permanently for everyone who clones it, and removing them later
means rewriting history.

Fetch them instead:

```bash
make download_geo_maps
```

The target wraps `preprocessing/step_08_download_geo_maps.py`, which pulls each
layer, extracts the TIGER archive, and reads every resulting shapefile back
through geopandas so a partial or corrupt download surfaces immediately rather
than at plot time. It is idempotent: files already on disk are skipped unless
`--force 1` is passed.

| Layer                                     | Source                        | Lands in           |
|-------------------------------------------|-------------------------------|--------------------|
| `ne_110m_admin_0_countries`               | Natural Earth (GitHub mirror) | `geo_maps/world/`  |
| `ne_110m_admin_0_boundary_lines_land`     | Natural Earth (GitHub mirror) | `geo_maps/world/`  |
| `tl_2023_us_state`                        | Census TIGER 2023             | `geo_maps/us/`     |

Natural Earth comes off the
[`nvkelso/natural-earth-vector`](https://github.com/nvkelso/natural-earth-vector)
mirror rather than naturalearthdata.com, which serves a malformed download URL
and goes down periodically. Each layer is fetched as its component sidecar
files: `.shp`, `.shx`, and `.dbf` are required to read the geometry, `.prj`
carries the CRS, and `.cpg` is an optional encoding hint that does not exist for
every theme.

Overrides for scale, themes, and TIGER vintage:

```bash
make download_geo_maps GEO_SCALE=50m
make download_geo_maps TIGER_VINTAGE=2024 TIGER_LAYERS="STATE COUNTY"
make download_geo_maps GEO_OUT=/tmp/scratch_maps
```

### Use in the EDA notebook

`notebooks/data_exploration.ipynb` reads the layers directly with geopandas.
Paths are relative to the repository root, so launch Jupyter from the root
rather than from inside `notebooks/`:

```python
import geopandas as gpd

states = gpd.read_file("geo_maps/us/tl_2023_us_state.shp")
world = gpd.read_file("geo_maps/world/ne_110m_admin_0_countries.shp")
```

Both arrive in EPSG:4326, matching the report coordinates, so no reprojection is
needed before joining. The notebook uses them for the per-state sighting-rate
choropleth, the point map of geocoded reports, and the world-boundary backdrop
on the international-reports panel. If those cells raise a missing-file error on
a fresh clone, `make download_geo_maps` has not been run yet.

Note that geopandas removed the bundled `naturalearth_lowres` dataset in
version 1.0. Older code calling `gpd.datasets.get_path("naturalearth_lowres")`
should read from `geo_maps/world/` instead.

## Running the pipeline

Everything is orchestrated through the `Makefile`. Run `make help` for the full
target list.

### Ingestion and preprocessing

The ingestion targets are retained for reproducibility of the authors' own
permitted collection. They are not intended for reuse: see the data attribution
notice above.

| Target                    | What it does                                                        |
|---------------------------|---------------------------------------------------------------------|
| `nuforc_details`          | Resumable detail retrieval with checkpointing and rate limiting     |
| `backfill_nuforc_text`    | Fills `Full_Text` from `Summary` for summary-only reports           |
| `preproc_pipeline`        | Data generation, NLP features, analytics, preprocessing, feature gen |
| `build_eda_frame`         | Joins the raw and model frames on `report_id` for notebook use      |
| `download_geo_maps`       | Fetches and verifies the basemap shapefiles                         |

### Training

| Target                             | What it does                                      |
|------------------------------------|---------------------------------------------------|
| `train_lr`                         | Logistic regression across all pipeline variants  |
| `train_cat`                        | Tabular CatBoost across all pipeline variants     |
| `train_cat_text_only`              | Text-only CatBoost                                |
| `train_cat_feats_and_text`         | Tabular + text CatBoost                           |
| `train_all_tabular`                | `train_lr` and `train_cat`                        |
| `train_cat_text_only_and_tab_text` | Both text models                                  |
| `train_all_models`                 | Everything                                        |

### Evaluation

| Target                            | What it does                                                    |
|-----------------------------------|-----------------------------------------------------------------|
| `eval_lr`, `eval_cat`             | Metrics and plots for the tabular models                        |
| `eval_cat_text_only`              | Text-only model, with LIME                                      |
| `eval_cat_feats_and_text`         | Tabular + text model, with SHAP and LIME                        |
| `eval_cat_text_only_and_tab_text` | Both text models                                                |
| `eval_all_models`                 | Everything                                                      |
| `save_predictions`                | Writes scored predictions for the deployed app                  |
| `bootstrap_eval`                  | 5,000-resample confidence intervals at each model's tuned threshold |

### Ablation sweeps

`sweep` runs any target twice, once per `DROP_YEAR` value. Each sub-invocation
re-expands the tag, so baseline and no-year artifacts stay separate.

```bash
make sweep T=train_cat_feats_and_text     # both variants of one model
make sweep T=eval_all_models              # both variants of every evaluation
```

### Combined pipelines

| Target                                     | What it does                                                        |
|--------------------------------------------|---------------------------------------------------------------------|
| `modeling_text_only_tab_text_eval_pipeline` | Trains and evaluates both text models at the current `DROP_YEAR`    |
| `modeling_text_ablation_pipeline`           | Trains and evaluates both text models at `DROP_YEAR=0` and `1` (four models) |
| `modeling_train_eval_pipeline`              | Trains and evaluates every model family                             |

A typical end-to-end workflow:

```bash
# 0. Fetch basemaps (once per clone)
make download_geo_maps

# 1. Preprocessing
make preproc_pipeline

# 2. Build the joined frame the EDA notebook reads
make build_eda_frame

# 3. Train and evaluate the text models, baseline and ablated
make modeling_text_ablation_pipeline

# 4. Bootstrap confidence intervals and deployment predictions
make bootstrap_eval
make save_predictions

# 5. Fit SHAP explainer and generate per-report explanations
make model_explaining_training

# Inspect MLflow runs
make mlflow_ui
```

For inference on a new batch of reports:

```bash
make preproc_pipeline_inference
```

### A note on exit codes

Every recipe pipes through `tee`, which masks the exit status of the underlying
Python process. To make a failed training run halt a sweep rather than letting
the evaluation phase proceed against a model that was never written, set the
following near the top of the `Makefile`:

```makefile
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
```

## Contributing

Nothing derived belongs in version control. Before opening a pull request,
confirm that no data files, model artifacts, shapefiles, virtual environments,
or regenerated figures are staged:

```bash
git status --short
git ls-files --others --exclude-standard
```

Adding a path to `.gitignore` does not untrack a file that is already committed.
If something has slipped in, untrack it before pushing:

```bash
git rm -r --cached path/to/directory
```

The rule of thumb: if a script can regenerate it or a URL can fetch it, commit
the script rather than the output. Cleaning a large binary out of the history
after the fact requires `git filter-repo` and a force-push, which invalidates
every existing clone.

## Data

Source reports come from the
[National UFO Reporting Center](https://nuforc.org), used with NUFORC's special
permission as stated above. Raw and processed data files are gitignored and are
not distributed with this repository.

Reports carry a two-tier editorial classification applied by NUFORC staff. Per
NUFORC, every report submitted on or after 17 March 2023 has been reviewed for
tier 1 inclusion; tier 2 was added as a second classification layer in October
2024; reports submitted before those dates were reviewed only where a specific
case came up to be looked at again. NUFORC also began rejecting roughly a third
of submissions outright in 2023, where before that nearly all submissions were
published. Both facts bear directly on how the labels in this project should be
interpreted, and both are documented in the analysis.

Basemap layers are separate from the report data and carry their own terms:
Natural Earth is public domain, and Census TIGER/Line files are US Government
works in the public domain. Neither is redistributed here.

## Authors

<table>
  <tr>
    <td width="160" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/main/photos/leonshpaner.jpg" width="140" alt="Leon Shpaner">
    </td>
    <td valign="top">
      <b><a href="https://leonshpaner.com/"> Leon Shpaner, M.S.</a></b><br><br>
      Leon is a Data Scientist at UCLA Health with over 15 years of experience across healthcare, financial services, and education. He serves as an adjunct professor at the University of San Diego, where he teaches statistics and machine learning in the M.S. in Applied Artificial Intelligence program. He has contributed to clinical prediction research, co-developed a production-grade EDA toolkit contracted for publication with Taylor &amp; Francis, and presented at JupyterCon 2025.
    </td>
  </tr>
  <tr>
    <td width="160" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/main/photos/Oscar_LinkedIn_Pic.jpeg" width="140" alt="Oscar Gil">
    </td>
    <td valign="top">
      <b><a href="https://oscargildata.com/"> Oscar Gil, M.S.</a></b><br><br>
      Oscar is a Data Scientist at the University of California, Riverside, with over ten years of experience in the education data management industry. He excels in data warehousing, analytics, machine learning, SQL, Python, R, and report authoring, and holds an M.S. in Applied Data Science from the University of San Diego. He has co-developed analytical tools and pipelines deployed in research and institutional settings, and presented alongside Leon at JupyterCon 2025.
    </td>
  </tr>
  <tr>
    <td width="160" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/main/photos/sean_torres.jpeg" width="140" alt="Sean Michael Torres">
    </td>
    <td valign="top">
      <b><a href="https://github.com/seantorres"> Sean Michael Torres, M.S.</a></b><br><br>
      Sean is a data analyst with experience across public service, operations, and business analytics. Focused on workflow automation, data quality, business intelligence, and predictive analytics. Builds Python reporting solutions, ETL workflows, Tableau dashboards, and data validation processes that support operational decision-making and improve efficiency.
    </td>
  </tr>

  <tr>
    <td width="160" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/main/Astro/public/photos/Nick_Shpaner.jpg" width="140" alt="Nicholas J. Shpaner">
    </td>
    <td valign="top">
      <b><a href="https://github.com/nshpaner"> Nicholas J. Shpaner</a></b><br><br>
      Nick is an aspiring Data Scientist, currently pursuing his Bachelor's degree at the University of California Merced. He currently works as a special projects assistant with UC Merced's Division of Undergraduate Education, and has experience in Python, R, and Excel. He has contributed to UAP research and analysis, as well as modeling search interest in Julian Apple Pie Company data.
    </td>
  </tr>

</table>

Data Science Dynamics: [datasciencedynamics.com](https://datasciencedynamics.com)

## References

- Carlotto, M. (2021). *A preliminary analysis of historical UFO report data.* SSRN. https://doi.org/10.2139/ssrn.3857231
- Medina, R. M., Brewer, S. C., & Kirkpatrick, S. M. (2023). An environmental analysis of public UAP sightings and sky view potential. *Scientific Reports*, 13, 22213. https://doi.org/10.1038/s41598-023-49527-x
- National UFO Reporting Center: [nuforc.org](https://nuforc.org)
- Natural Earth. *Free vector and raster map data at 1:10m, 1:50m, and 1:110m scales.* https://www.naturalearthdata.com
- Posard, M. N., Gromis, A., & Lee, M. (2023). *Not the X-Files: Mapping Public Reports of Unidentified Aerial Phenomena Across America.* RAND Corporation (RR-A2475-1). https://www.rand.org/pubs/research_reports/RRA2475-1.html
- US Census Bureau. *TIGER/Line Shapefiles.* https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Leon Shpaner and Oscar Gil.

The MIT licence covers the code in this repository only. It does not cover
NUFORC report data, which is not distributed here and is not licensed for reuse
by this project.