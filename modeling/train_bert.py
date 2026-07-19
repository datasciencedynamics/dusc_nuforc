from bertuner.BERTuner import BERTuneClassifier
import pandas as pd
import os

# data_path = ("./data/processed/df_final.parquet",)

# df = pd.read_parquet(os.path.join(*data_path))

# print(df)


# 1. Initialize
classifier = BERTuneClassifier(
    data_path="../data/processed/df_final.parquet",  # or dataframe=my_df
    models_dir="../models/",
    text_feature="full_text_clean",  # column containing the text
    target_cols=["dramatic"],  # one column = single-label
    max_length=8192,
)

# 2. Configure (optional: uses defaults if called without arguments)
classifier.initialize_model_choices()
classifier.initialize_search_space()

# 3. Optimize — runs Optuna trials and logs to MLflow
best_value = classifier.optimize(
    n_trials=20,
    optimize_metric="avg_precision",
    study_name="bert_experiment_v1",
)

# 4. Train final model — retrains on best params, optimises the decision
#    threshold on the validation set, evaluates on the test set, and saves
#    model + tokenizer + bertuner_config.json under models_dir/final_model/model
metrics, model, test_ds = classifier.train_final_model()
print(metrics)
