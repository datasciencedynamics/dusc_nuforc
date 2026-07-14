#!/usr/bin/env python3
"""
build_dashboard_bootstrap.py
============================
Reshape the bootstrap CIs that bootstrap_evaluation.py already produced into the
form the dashboard reads.

Does NOT re-bootstrap. Re-running the resampler would draw different replicates
and put slightly different intervals in the dashboard than in the paper, for the
same quantity. One bootstrap, one source of truth.

Prerequisite — run BOTH variants first:

    python modeling/bootstrap_evaluation.py --text-col full_text_clean
    python modeling/bootstrap_evaluation.py --text-col full_text_clean --drop-year 1

Only cat_feats_and_text has a real year ablation. The tabular cat/lr models were
fit WITH occurred_year and never re-run; cat_text_only never saw it. So four of
the five dashboard columns come from the with-year file, and only the fifth is
pulled from the ablated one.

Usage
-----
    python dashboard_prep/build_dashboard_bootstrap.py
"""

from pathlib import Path

import pandas as pd

EVAL_DIR = Path("models/eval")
SRC = {
    "": EVAL_DIR / "bootstrap_metrics_full_text_clean.csv",
    "_noyear": EVAL_DIR / "bootstrap_metrics_full_text_clean_noyear.csv",
}
OUT = Path("../flask_apps/dusc_nuforc_dash/data/bootstrap_metrics.csv")

# (dashboard column, source file, model_type, pipeline_type)
# Pipeline choices mirror build_dashboard.py: lr/orig and cat/smote.
WANT = [
    ("model_lr", "", "lr", "orig"),
    ("model_cat", "", "cat", "smote"),
    ("model_cat_text_only", "", "cat_text_only", "orig"),
    ("model_cat_feats_and_text", "", "cat_feats_and_text", "orig"),
    ("model_cat_feats_and_text_noyear", "_noyear", "cat_feats_and_text", "orig"),
]

# neg_brier_score is written positive by the bootstrapper despite the name, so
# no sign flip is needed. Verified against models/eval output.
METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Average Precision",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1_weighted": "F1 (Weighted)",
    "neg_brier_score": "Brier Score",
}

# Display order in the dashboard table.
METRIC_ORDER = [
    "ROC-AUC",
    "Average Precision",
    "Precision",
    "Recall",
    "Specificity",
    "F1 (Weighted)",
    "Brier Score",
]


def main():
    for suffix, path in SRC.items():
        if not path.exists():
            raise SystemExit(
                f"missing {path}\n"
                "Run bootstrap_evaluation.py for both variants first — see the "
                "docstring."
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = {suffix: pd.read_csv(path) for suffix, path in SRC.items()}

    rows = []
    for col, suffix, mtype, ptype in WANT:
        src = frames[suffix]
        sub = src[(src["model_type"] == mtype) & (src["pipeline_type"] == ptype)]
        if sub.empty:
            raise SystemExit(
                f"no rows for {mtype}/{ptype} in {SRC[suffix]} — check the "
                "pipeline_type against what bootstrap_evaluation.py actually ran."
            )

        for _, r in sub.iterrows():
            metric = METRIC_LABELS.get(r["Metric"])
            if metric is None:
                continue
            rows.append(
                {
                    "model": col,
                    "metric": metric,
                    "point": r["Mean"],
                    "ci_lo": r["95% CI Lower"],
                    "ci_hi": r["95% CI Upper"],
                }
            )

    out = pd.DataFrame(rows)
    out["metric"] = pd.Categorical(out["metric"], categories=METRIC_ORDER, ordered=True)
    out = out.sort_values(["model", "metric"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    disp = out.copy()
    disp["value"] = disp.apply(
        lambda r: f"{r['point']:.4f} ({r['ci_lo']:.4f}, {r['ci_hi']:.4f})", axis=1
    )
    print(
        disp.pivot(index="model", columns="metric", values="value")
        .reindex(columns=METRIC_ORDER)
        .to_string()
    )
    print(f"\n{len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
