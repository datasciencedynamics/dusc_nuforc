# Makefile
# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
PROJECT_NAME = dusc_nuforc
PYTHON_VERSION = 3.12.7
PYTHON_INTERPRETER = python
VENV_DIR = nuforc_venv
CONDA_ENV_NAME = nuforc_conda
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIRECTORY := $(abspath $(MAKEFILE_DIR))


############################## Training Globals ################################

OUTCOME := dramatic
TEXT_COL = full_text_clean
# DROP_YEAR must be assigned before YEAR_TAG is referenced, and YEAR_TAG must be
# recursive (=, not :=) so overrides on the command line or from a sub-make
# propagate into the tag.
DROP_YEAR = 0
YEAR_TAG = $(if $(filter-out 0,$(DROP_YEAR)),_noyear,)
PIPELINES := orig smote under orig_rfe smote_rfe under_rfe
PIPELINES = orig smote under orig_rfe smote_rfe under_rfe
SCORING = average_precision
PROMPT_TYPES := zero_shot few_shot
PRETRAINED ?= 0  # 0 if you want to train the models, 1 if calibrate pretrained

############################# Production Globals ###############################

# Model outcome variable used in production 
EXPLAN_OUTCOME = dramatic # explainer outcome variable
PROD_OUTCOME = dramatic # production outcome variable

# ------------------------------------------------------------------------------
# COMMANDS
# ------------------------------------------------------------------------------
.PHONY: init_config
init_config:
	@CURRENT_DIR=$$(sed -n 's/^PROJECT_DIRECTORY = //p' Makefile); \
	\
	read -p "Enter project name: " project_name; \
	read -p "Enter Python version (e.g., 3.10.12): " python_version; \
	read -p "Enter Python interpreter (default: python): " python_interpreter; \
	read -p "Enter virtual environment directory name: " venv_dir; \
	read -p "Enter conda environment name: " conda_env; \
	python_interpreter=$${python_interpreter:-python}; \
	\
	if [ -d "$$CURRENT_DIR" ] && [ "$$CURRENT_DIR" != "$$project_name" ]; then \
		mv "$$CURRENT_DIR" "$$project_name"; \
	fi; \
	\
	# Cross-platform sed command (works on both macOS and Linux) \
	if [ "$$(uname)" = "Darwin" ]; then \
		sed -i '' \
			-e "s/^PROJECT_NAME = .*/PROJECT_NAME = $${project_name}/" \
			-e "s/^PYTHON_VERSION = .*/PYTHON_VERSION = $${python_version}/" \
			-e "s/^PYTHON_INTERPRETER = .*/PYTHON_INTERPRETER = $${python_interpreter}/" \
			-e "s/^VENV_DIR = .*/VENV_DIR = $${venv_dir}/" \
			-e "s/^CONDA_ENV_NAME = .*/CONDA_ENV_NAME = $${conda_env}/" \
			-e "s|^PROJECT_DIRECTORY = .*|PROJECT_DIRECTORY = $${project_name}|" \
			Makefile; \
	else \
		sed -i \
			-e "s/^PROJECT_NAME = .*/PROJECT_NAME = $${project_name}/" \
			-e "s/^PYTHON_VERSION = .*/PYTHON_VERSION = $${python_version}/" \
			-e "s/^PYTHON_INTERPRETER = .*/PYTHON_INTERPRETER = $${python_interpreter}/" \
			-e "s/^VENV_DIR = .*/VENV_DIR = $${venv_dir}/" \
			-e "s/^CONDA_ENV_NAME = .*/CONDA_ENV_NAME = $${conda_env}/" \
			-e "s|^PROJECT_DIRECTORY = .*|PROJECT_DIRECTORY = $${project_name}|" \
			Makefile; \
	fi; \
	\
	# Replace project name in Python files and other text files only \
	if [ "$$(uname)" = "Darwin" ]; then \
		find "./$$project_name" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i '' "s/$$CURRENT_DIR/$$project_name/g" {} \;; \
	else \
		find "./$$project_name" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i "s/$$CURRENT_DIR/$$project_name/g" {} \;; \
	fi; \
	\
	echo "Configuration updated successfully. Folder '$$CURRENT_DIR' -> '$$project_name'."

.PHONY: check_vars
check_vars:
	@echo "Dummy configuration detected."
	@echo ""
	@echo "Please update the following variables in your Makefile before proceeding:"
	@echo " - PROJECT_NAME"
	@echo " - PYTHON_VERSION"
	@echo " - VENV_DIR"
	@echo " - CONDA_ENV_NAME"
	@echo " - OUTCOME"
	@echo " - PIPELINES"
	@echo " - SCORING"
	@echo " - EXPLAN_OUTCOME"
	@echo " - PROD_OUTCOME"
	@echo ""
	@echo "Once you've replaced the dummy values, you can run your full pipeline commands safely."

## Set up python interpreter environment
create_conda_env:
	@echo "Run 'conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)' to create conda environment"

## Activate the conda environment
activate_conda_env:
	@echo "Run 'conda activate $(CONDA_ENV_NAME)' to activate the conda environment"
	
# Target to create a virtual environment
create_venv:
	# Create the virtual environment using the specified Python version
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Virtual environment created with $(PYTHON_INTERPRETER)$(PYTHON_VERSION)"

# Target to activate the virtual environment (Unix-based systems)
activate_venv:
	@echo "Run 'conda deactivate' to deactivate the $(CONDA_ENV_NAME) conda environment"
	@echo "Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment"

# Target to clean the virtual environment
clean_venv:
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# Instantiate MLFlow                                                            #
#################################################################################

.PHONY: mlflow_ui
mlflow_ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5501

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
# clean directories
clean_dir:
	@echo "Cleaning directory..."
	rm -rf data/


# Folder Creation 
.PHONY: create_folders
create_folders:
	# Create data subdirectories
	mkdir -p data/external data/interim data/processed data/raw data/processed/inference
	mkdir -p modeling
	mkdir -p core
	mkdir -p preprocessing
	touch data/interim/.gitkeep
	touch data/processed/.gitkeep
	touch data/processed/inference/.gitkeep
	touch modeling/__init__.py
	touch preprocessing/__init__.py
	touch core/__init__.py

	# Create models subdirectories for each outcome
	@for outcome in $(OUTCOME); do \
		mkdir -p models/results/$$outcome; \
		mkdir -p models/eval/$$outcome; \
	done

################################################################################
# NUFORC shared vars
################################################################################
NUFORC_RAW        = ./data/raw/NUFORC_DATA_04_10_2026.xlsx
NUFORC_ENRICHED   = ./data/raw/NUFORC_DATA_04_10_2026_enriched.xlsx
NUFORC_CHECKPOINT = ./data/raw/nuforc_enrich_checkpoint_TEST.jsonl
SCRAPE_MIN_DELAY ?= 5
SCRAPE_MAX_DELAY ?= 8

################################################################################
# PREPROCESSING -- step_00 detail scraper (part of the real pipeline)
################################################################################

# Full scrape: walks all rows, fetches only those not already in the checkpoint.
.PHONY: scrape_nuforc_details
scrape_nuforc_details:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/step_00_NUFORC_Extractor.py \
		--input-data-file "$(NUFORC_RAW)" \
		--output-data-file "$(NUFORC_ENRICHED)" \
		--checkpoint "$(NUFORC_CHECKPOINT)" \
		--link-col 1 \
		--min-delay $(SCRAPE_MIN_DELAY) \
		--max-delay $(SCRAPE_MAX_DELAY) \
		2>&1 | tee ./data/raw/scrape_nuforc_details.txt

# Alias for clarity when re-fetching a pruned subset (same script, same args).
.PHONY: rescrape_nuforc_details
rescrape_nuforc_details: scrape_nuforc_details

# Re-scrape at a faster delay. Run AFTER debug_prune_* , not chained, so
# resuming never re-prunes already-fixed rows. Override delays as needed.
.PHONY: rescrape_nuforc_fast
rescrape_nuforc_fast:
	$(MAKE) scrape_nuforc_details SCRAPE_MIN_DELAY=2 SCRAPE_MAX_DELAY=3

# Smoke test: first 10 rows into a separate checkpoint + separate output.
.PHONY: scrape_nuforc_details_test
scrape_nuforc_details_test:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/step_00_NUFORC_Extractor.py \
		--input-data-file "$(NUFORC_RAW)" \
		--output-data-file "./data/raw/NUFORC_DATA_04_10_2026_enriched_TEST.xlsx" \
		--checkpoint "./data/raw/nuforc_enrich_checkpoint_SMOKE.jsonl" \
		--test-limit 10 \
		2>&1 | tee ./data/raw/scrape_nuforc_details_test.txt

################################################################################
# DEBUG / ONE-OFF MAINTENANCE -- debug_scripts/ (NOT part of the pipeline)
################################################################################

NUFORC_BACKFILLED = ./data/raw/NUFORC_DATA_04_10_2026.xlsx

# Backfill Full_Text from Summary for summary-only reports (no scraping, local
# column edit). Adds a text_is_summary_only flag and reports class balance of
# affected rows. Run AFTER the scrape is clean, BEFORE data_gen.
.PHONY: backfill_nuforc_text
backfill_nuforc_text:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/debug_scripts/backfill_summary_text.py \
		--input-file "$(NUFORC_RAW)" \
		--output-file "$(NUFORC_BACKFILLED)" \
		--outcome-col $(OUTCOME) \
		2>&1 | tee ./data/raw/step_02_backfill_summary_text.txt

# Audit: bucket every row so you can confirm the truncation is gone.
.PHONY: debug_audit_truncation
debug_audit_truncation:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/debug_scripts/audit_truncation.py \
		2>&1 | tee ./data/raw/debug_audit_truncation.txt

# Prune truncated rows (empty or Full_Text == Summary) from the checkpoint.
.PHONY: debug_prune_truncated
debug_prune_truncated:
	cp "$(NUFORC_CHECKPOINT)" "$(NUFORC_CHECKPOINT).bak"
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/debug_scripts/prune_truncated_checkpoint.py \
		2>&1 | tee ./data/raw/debug_prune_truncated.txt

# Combined prune: SCRAPE FAILED + empty + Full_Text==Summary in one pass.
.PHONY: debug_prune_all
debug_prune_all:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/debug_scripts/prune_all_questionable.py \
		--enriched-file "$(NUFORC_RAW)" \
		--checkpoint "$(NUFORC_CHECKPOINT)" \
		2>&1 | tee ./data/raw/debug_prune_all.txt

# Verify the parser fix on real pages before a mass re-scrape.
.PHONY: debug_verify_fix
debug_verify_fix:
	PYTHONPATH=$(PROJECT_DIRECTORY) $(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/debug_scripts/verify_fix.py \
		2>&1 | tee ./data/raw/debug_verify_fix.txt

# Export the empty/==Summary rows to a standalone workbook for manual review.
.PHONY: debug_export_summary_eq
debug_export_summary_eq:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/debug_scripts/summary_equals_full_text.py \
		--enriched-file "$(NUFORC_RAW)" \
		--output-file "./data/raw/NUFORC_review_summary_eq_full_text.xlsx" \
		2>&1 | tee ./data/raw/debug_export_summary_eq.txt

# One-off structural diagnostic on a single short page.
.PHONY: debug_inspect_short
debug_inspect_short:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/debug_scripts/less_than_120_char.py \
		2>&1 | tee ./data/raw/debug_inspect_short.txt

################################################################################
################################### Training ###################################
####################### Preprocessing (+) Dataprep Pipeline ####################
################################################################################

.PHONY: data_gen
data_gen:
	$(PYTHON_INTERPRETER) preprocessing/step_01_data_gen.py \
		--input-data-file "./data/raw/NUFORC_DATA_04_10_2026.xlsx" \
		--output-data-file "./data/raw/nuforc_data.parquet" \
		2>&1 | tee ./data/raw/step_01_data_gen.txt

.PHONY: nlp_feature_engineer_nuforc
nlp_feature_engineer_nuforc:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/step_03_nlp_feature_engineer_nuforc.py \
		--input-parquet "./data/raw/nuforc_data.parquet" \
		--output-parquet "./data/processed/nuforc_engineered.parquet" \
		--output-metadata "./data/processed/nuforc_feature_metadata.json" \
		2>&1 | tee ./data/processed/step_03_nlp_feature_engineer_nuforc.txt

.PHONY: nuforc_analytics
nuforc_analytics:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/step_04_nuforc_analytics.py \
		--input-parquet "./data/processed/nuforc_engineered.parquet" \
		--output-parquet "./data/processed/NUFORC_enriched.parquet" \
		2>&1 | tee ./data/processed/step_04_nuforc_analytics.txt

.PHONY: data_prep_preprocessing_training
data_prep_preprocessing_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/step_05_preprocessing_remaining_feats.py \
		--input-data-file ./data/processed/NUFORC_enriched.parquet \
		--output-data-file ./data/processed/df_sans_zero_missing.parquet \
		--stage training \
		--data-path ./data/processed \
		2>&1 | tee ./data/processed/step_05_preprocessing_remaining_feats.txt

.PHONY: feat_gen_training
feat_gen_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/step_06_feat_gen.py \
		--input-data-file ./data/processed/df_sans_zero_missing.parquet \
		--output-data-file ./data/processed/df_final.parquet \
		--stage training \
		--data-path ./data/processed \
		2>&1 | tee ./data/processed/step_06_feat_gen_training.txt

preproc_pipeline: data_gen \
                  nlp_feature_engineer_nuforc \
				  nuforc_analytics \
				  data_prep_preprocessing_training  \
				  feat_gen_training

################################################################################
################################# Training #####################################
################################################################################

define train_text_model
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/train.py \
		--model-type $(1) \
		--pipeline-type $(2) \
		--text-col $(TEXT_COL) \
		--outcome $(OUTCOME) \
		--scoring average_precision \
		--drop-year $(DROP_YEAR) \
		--pretrained 0 \
	2>&1 | tee models/results/$(OUTCOME)/$(1)_$(2)_$(TEXT_COL)$(YEAR_TAG)_train.txt
endef

# Tabular models -- loop over all pipeline types
train_lr:
	$(foreach p,$(PIPELINES),$(call train_text_model,lr,$(p)) &&) true

train_cat:
	$(foreach p,$(PIPELINES),$(call train_text_model,cat,$(p)) &&) true

# Text models -- pipeline_type is ignored internally but passed for MLflow run naming
train_cat_feats_and_text:
	$(call train_text_model,cat_feats_and_text,orig)

train_cat_text_only:
	$(call train_text_model,cat_text_only,orig)

# Rollups
train_all_tabular: train_lr train_cat

train_cat_text_only_and_tab_text: train_cat_text_only train_cat_feats_and_text

train_all_models: train_all_tabular train_cat_feats_and_text train_cat_text_only

.PHONY: train_lr train_cat train_cat_feats_and_text train_cat_text_only \
        train_all_tabular train_cat_text_only_and_tab_text train_all_models

################################################################################
############################### Model Evaluation ###############################
################################################################################

# Mirrors evaluate.py: text models are keyed by text_col, tabular models drop
# all text and carry no tag. Keeping this in one place stops the Makefile's
# directory layout from drifting away from what evaluate.py actually writes.
TEXT_TAG_FOR = $(if $(filter cat_feats_and_text cat_text_only,$(1)),_$(TEXT_COL),)

define eval_model
	mkdir -p models/eval/$(3)/$(1)
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluate.py \
		--model-type $(1) \
		--pipeline-type $(2) \
		--outcome $(3) \
		--text-col $(TEXT_COL) \
		--drop-year $(DROP_YEAR) \
		--output-dir ./models/eval \
	2>&1 | tee models/eval/$(3)/$(1)/$(2)$(call TEXT_TAG_FOR,$(1))$(YEAR_TAG)_eval.txt
endef

eval_lr:      ; $(foreach p,$(PIPELINES),$(call eval_model,lr,$(p),$(OUTCOME)) &&) true
eval_cat:     ; $(foreach p,$(PIPELINES),$(call eval_model,cat,$(p),$(OUTCOME)) &&) true
eval_cat_feats_and_text:  ; $(call eval_model,cat_feats_and_text,orig,$(OUTCOME))
eval_cat_text_only:       ; $(call eval_model,cat_text_only,orig,$(OUTCOME))

# Rollups
eval_cat_text_only_and_tab_text: eval_cat_text_only eval_cat_feats_and_text

eval_all_models: eval_lr eval_cat eval_cat_feats_and_text eval_cat_text_only

.PHONY: eval_lr eval_cat eval_cat_feats_and_text eval_cat_text_only \
        eval_cat_text_only_and_tab_text eval_all_models

################################################################################
################################ Ablation Sweep ################################
################################################################################

# Run any target twice, once per DROP_YEAR value. Each sub-make re-expands
# YEAR_TAG, so log filenames, eval dirs, and MLflow run names stay distinct
# between the baseline and the no-year variant.
#
#   make sweep T=train_cat_feats_and_text
#   make sweep T=eval_all_models
#   make sweep T=train_cat_text_only_and_tab_text

sweep:
	@test -n "$(T)" || { echo "usage: make sweep T=<target>"; exit 1; }
	$(MAKE) $(T) DROP_YEAR=0
	$(MAKE) $(T) DROP_YEAR=1

.PHONY: sweep

################################################################################
# Full text-model ablation pipeline: trains and evaluates all four runs
# (cat_text_only and cat_feats_and_text, each with DROP_YEAR=0 and 1).
# Uses recipe lines rather than prerequisites so train always completes
# before eval, even under make -j.
################################################################################

modeling_text_ablation_pipeline:
	$(MAKE) sweep T=train_cat_text_only_and_tab_text
	$(MAKE) sweep T=eval_cat_text_only_and_tab_text

.PHONY: modeling_text_ablation_pipeline

.PHONY: save_predictions
save_predictions:
	mkdir -p ./models/eval/predictions ./models/predictions/$(TEXT_COL)$(YEAR_TAG)
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/save_predictions.py \
		--outcome $(OUTCOME) \
		--text-col $(TEXT_COL) \
		--drop-year $(DROP_YEAR) \
		--cat-pipeline smote \
		--lr-pipeline orig \
		--output-dir ./models/predictions/$(TEXT_COL)$(YEAR_TAG) \
		2>&1 | tee ./models/eval/predictions/save_predictions_$(TEXT_COL)$(YEAR_TAG).txt

.PHONY: bootstrap_eval
bootstrap_eval:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/bootstrap_evaluation.py \
		--outcome $(OUTCOME) \
		--text-col $(TEXT_COL) \
		--drop-year $(DROP_YEAR) \
		--n-samples -1 \
		--num-resamples 5000 \
		--output-dir ./models/eval \
		--output-csv bootstrap_metrics.csv \
		2>&1 | tee ./models/eval/bootstrap_eval_$(TEXT_COL)$(YEAR_TAG).txt
		# --n-samples -1 means use full test set size (len(X_test))


################################################################################
################################ BERT Fine-Tuning ##############################
################################################################################

BERT_TEXT_COL      ?= $(TEXT_COL)
BERT_MAX_LENGTH    ?= 512
# TPESampler uses n_startup_trials=10 for random exploration before its model
# engages, so anything at or below 10 is pure random search. 14 buys four
# TPE-guided trials.
BERT_N_TRIALS      ?= 15
BERT_INPUT         ?= ./data/processed/df_final.parquet
BERT_SKIP_OPTIMIZE ?= 0

# BERTuner writes a checkpoint per trial to models/optuna_trial_N and only
# prunes them in _cleanup_trials after the study completes. An interrupted run
# leaves those behind, and since trial numbering restarts at 0, the next run
# writes into stale directories. Always clear them before starting a study.
.PHONY: clean_bert_trials
clean_bert_trials:
	@echo "Removing stale Optuna trial checkpoints ..."
	@rm -rf models/optuna_trial_*
	@echo "Done."

.PHONY: train_bert
train_bert: clean_bert_trials
	mkdir -p models/eval/$(OUTCOME)/bert models/results/$(OUTCOME)
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/train_bert.py \
		--features-path "$(BERT_INPUT)" \
		--models-dir ./models \
		--text-col $(BERT_TEXT_COL) \
		--outcome $(OUTCOME) \
		--max-length $(BERT_MAX_LENGTH) \
		--n-trials $(BERT_N_TRIALS) \
		--scoring avg_precision \
		--tracking-uri ./mlruns \
		--output-dir ./models/eval \
		--skip-optimize $(BERT_SKIP_OPTIMIZE) \
	2>&1 | tee models/results/$(OUTCOME)/bert_$(BERT_TEXT_COL)_train.txt

# Retrain the final model from an existing study without repeating the search.
# Does NOT clean trial checkpoints, since the best trial's weights are needed.
.PHONY: train_bert_final_only
train_bert_final_only:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/train_bert.py \
		--features-path "$(BERT_INPUT)" \
		--models-dir ./models \
		--text-col $(BERT_TEXT_COL) \
		--outcome $(OUTCOME) \
		--max-length $(BERT_MAX_LENGTH) \
		--scoring avg_precision \
		--tracking-uri ./mlruns \
		--output-dir ./models/eval \
		--skip-optimize 1 \
	2>&1 | tee models/results/$(OUTCOME)/bert_$(BERT_TEXT_COL)_final.txt

################################ Modeling Pipeline #############################
### Shortcut to run full modeling pipeline: training, evaluation
################################################################################

modeling_text_only_tab_text_eval_pipeline: train_cat_text_only_and_tab_text eval_cat_text_only_and_tab_text

# Both variants: 4 models total (text_only and feats_and_text, each x2)
modeling_text_only_tab_text_eval_pipeline_ablation:
	$(MAKE) train_all_text_ablation
	$(MAKE) eval_all_text_ablation

modeling_train_eval_pipeline: train_all_models eval_all_models

################################################################################
#################### Best Model Explainer and Explanations #####################
################################################################################

.PHONY: model_explainer
model_explainer:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explainer.py \
			--outcome $$outcome \
			--metric-name "valid_r2" \
			--mode max \
			2>&1 | tee ./data/processed/model_explainer_$$outcome.txt; \
	done

.PHONY: model_explanations_training
model_explanations_training:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explanations_training.py \
			--features-path ./data/processed/X_test.parquet \
			--outcome $$outcome \
			--metric-name "valid_r2" \
			--mode max \
			--top-n 5 \
			--shap-val-flag 1 \
			--explanations-path ./data/processed/shap_predictions_$$outcome.csv \
			2>&1 | tee ./data/processed/model_explanations_training_$$outcome.txt; \
	done

model_explaining_training: model_explainer model_explanations_training


################################################################################
################################# Production ###################################
############################### Model Predict ##################################
################################################################################

.PHONY: data_prep_preprocessing_inference
data_prep_preprocessing_inference:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
		--input-data-file ./data/raw/acled_ukraine_data_2026_01_02.parquet \
		--output-data-file ./data/processed/inference/df_inference_process.parquet \
		--stage inference \
		--data-path ./data/processed \
		2>&1 | tee ./data/processed/inference/data_prep_preprocessing_inference.txt

.PHONY: feat_gen_inference
feat_gen_inference:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
		--input-data-file ./data/processed/inference/df_inference_process.parquet \
		--stage inference \
		--data-path ./data/processed/inference \
		2>&1 | tee ./data/processed/inference/feat_gen_inference.txt

.PHONY: predict
predict:
	@for outcome in $(PROD_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/predict.py \
			--input-data-file data/processed/inference/X.parquet \
			--predictions-path ./data/processed/inference/predictions_$$outcome.csv \
			--outcome $$outcome \
			--metric-name "valid_r2" \
			--mode max \
			2>&1 | tee ./data/processed/inference/predict_$$outcome.txt; \
	done

.PHONY: model_explainer_inference
model_explainer_inference:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explainer.py \
			--outcome $$outcome \
			--metric-name "valid_r2" \
			--mode max; \
	done

.PHONY: model_explanations_inference
model_explanations_inference:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explanations_inference.py \
			--features-path ./data/processed/inference/X.parquet \
			--outcome $$outcome \
			--metric-name "test_r2" \
			--mode max \
			--top-n 5 \
			--shap-val-flag 1 \
			--explanations-path ./data/processed/inference/shap_predictions_$$outcome.csv \
			2>&1 | tee ./data/processed/inference/model_explanations_$$outcome.txt; \
	done

preproc_pipeline_inference: data_prep_preprocessing_inference \
    feat_gen_inference \
    predict \
	model_explainer_inference \
    model_explanations_inference

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)