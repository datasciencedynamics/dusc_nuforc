import json
from core.functions import mlflow_load_model

model = mlflow_load_model(
    experiment_name="dramatic_text_model",
    run_name="cat_feats_and_text_orig_full_text_clean_training",
    model_name="cat_feats_and_text_dramatic",
)

est = model.estimator
cc = est.calibrated_classifiers_[0]
cb = cc.estimator.steps[-1][1]

print("got catboost:", type(cb))
cb.save_model("/tmp/uap_fresh.cbm")
print("saved")

from catboost import CatBoostClassifier

m = CatBoostClassifier()
m.load_model("/tmp/uap_fresh.cbm")
print("reloaded OK")
