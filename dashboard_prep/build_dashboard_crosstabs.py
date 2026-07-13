#!/usr/bin/env python3
"""
build_dashboard_crosstabs.py
============================
Precompute cohort crosstabs into a pickle the dashboard reads directly, so the
Flask app never needs NUFORC_enriched.parquet or the training environment.

Each value is a DataFrame indexed by category level, with columns:

    Not Dramatic | Dramatic | Total | Not Dramatic_% | Dramatic_%

and a Total row appended. This matches the shape circ_milan's update_bar
expects, so the plotting code carries over unchanged.

NUFORC_enriched.parquet is features only — the dramatic label is derived
downstream — so the label is joined back on by index here.

Usage
-----
    python modeling/build_dashboard_crosstabs.py
"""

import pickle
from pathlib import Path

import pandas as pd

SRC = Path("data/processed/NUFORC_enriched.parquet")
Y_SRC = Path("data/processed/y_dramatic.parquet")
OUT = Path("../flask_apps/dusc_nuforc_dash/data/freq_cols.pkl")

OUTCOME = "dramatic"
LABELS = {0: "Not Dramatic", 1: "Dramatic"}

# Low-cardinality only. State has hundreds of levels and would be unreadable as
# a bar chart; Country and Shape are capped to their most common levels below.
FEATURES = [
    "Shape",
    "shape_group",
    "Country",
    "occurred_year",
    "occurred_month",
    "occurred_hour",
    "is_night",
    "is_weekend",
    "has_media",
    "in_cluster",
    "exp_certain",
]

TOP_N = 15  # cap high-cardinality nominal features to their most common levels

# Ordinal features read in their natural order, not by volume: trimming hour to
# the 15 busiest and sorting by count would hide the night-time peak, which is
# the whole point of plotting it.
ORDINAL = {"occurred_year", "occurred_month", "occurred_hour"}


def crosstab(df, feature):
    ct = pd.crosstab(df[feature], df[OUTCOME])
    ct = ct.rename(columns=LABELS)

    for lab in LABELS.values():
        if lab not in ct.columns:
            ct[lab] = 0
    ct = ct[list(LABELS.values())]

    ct["Total"] = ct.sum(axis=1)

    if feature in ORDINAL:
        ct = ct.sort_index()
    elif len(ct) > TOP_N:
        ct = ct.nlargest(TOP_N, "Total")

    ct.loc["Total"] = ct.sum()

    for lab in LABELS.values():
        ct[f"{lab}_%"] = (ct[lab] / ct["Total"] * 100).round(2)

    return ct


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SRC)
    n_total = len(df)

    y = pd.read_parquet(Y_SRC).squeeze()

    # Join on index, never on row order: step_05/06 may have dropped rows, and a
    # positional assignment would silently mislabel every crosstab.
    df[OUTCOME] = y.reindex(df.index)
    df = df[df[OUTCOME].notna()]
    df[OUTCOME] = df[OUTCOME].astype(int)

    print(f"{len(df):,} of {n_total:,} rows carry a label")
    print(f"positive rate: {df[OUTCOME].mean():.1%}\n")

    freq_cols = {}
    for feat in FEATURES:
        if feat not in df.columns:
            print(f"  skip: {feat} not in data")
            continue
        ct = crosstab(df, feat)
        freq_cols[feat] = ct
        print(f"  {feat}: {len(ct) - 1} levels")

    with open(OUT, "wb") as f:
        pickle.dump(freq_cols, f)

    print(f"\n{len(freq_cols)} crosstabs -> {OUT}")


if __name__ == "__main__":
    main()
