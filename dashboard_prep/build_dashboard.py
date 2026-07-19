#!/usr/bin/env python3
"""
build_dashboard.py
==================
Collapse save_predictions.py's outputs into the single frame the dashboard
reads: one column per model, one row per test observation, the true label, the
grouping features the subgroup panel needs, and the geography the map panel
needs.

Five model columns, not eight. Only cat_feats_and_text was retrained without
occurred_year — the tabular cat/lr models were fit WITH the year and were never
re-run, and cat_text_only never had the year to begin with. Their "_noyear"
files are byte-identical copies. Emitting all eight would put three identical
pairs in the model dropdown, and a reader toggling
"Logistic Regression (with year)" against "(year ablated)" would see two curves
lying exactly on top of each other and conclude the year does not matter. That
is the opposite of the finding.

So the one real ablation — cat_feats_and_text with vs. without occurred_year —
is the only pair here, and it stands alone.

Why the ablation exists: the NUFORC dramatic-flag rate swings 5-fold across the
study period (3.6% in 2022 vs 18.0% in 2024-25) with no corresponding change in
the phenomena reported, so occurred_year proxies the editorial regime that
labeled a report rather than anything about the sighting.

All models were fit on different feature sets but scored on the SAME test rows,
split from the same y, so the indices align across every column.
"""

import json
from pathlib import Path

import pandas as pd

PRED_ROOT = Path("models/predictions")
ENRICHED = Path("data/processed/NUFORC_enriched.parquet")
OUT_DIR = Path("../flask_apps/dusc_nuforc_dash/data")

# (output column, source subdirectory, model key in the filenames)
SPECS = [
    ("model_lr", "full_text_clean", "lr"),
    ("model_cat", "full_text_clean", "cat"),
    ("model_cat_text_only", "full_text_clean", "cat_text_only"),
    ("model_cat_feats_and_text", "full_text_clean", "cat_feats_and_text"),
    (
        "model_cat_feats_and_text_noyear",
        "full_text_clean_noyear",
        "cat_feats_and_text",
    ),
]

# save_predictions.py keys thresholds by display name; the dashboard needs them
# keyed by the column it plots.
THRESH_KEYS = {
    "lr": "Logistic Regression",
    "cat": "CatBoost Tabular (SMOTE)",
    "cat_text_only": "CatBoost Text Only",
    "cat_feats_and_text": "CatBoost Feats + Text",
}

# Grouping features for the subgroup panel — the same eleven the cohort
# crosstabs use, so the two panels describe the same cuts of the data.
#
# Geography is deliberately NOT in this list. GROUP_COLS is mirrored against
# build_dashboard_crosstabs.py's FEATURES so the subgroup panel and the cohort
# bars describe identical cuts, and latitude is not a cut. Appending it here
# would silently desynchronize the two panels and put a continuous coordinate
# in a categorical dropdown.
GROUP_COLS = [
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

################################################################################
# Geography for the map panel.
#
# Column names are taken from step_04_nuforc_analytics.py's output_cols, which
# is the definitive list of what NUFORC_enriched.parquet carries. Note the
# mixed casing: latitude/longitude are lowercase (step_04 coerces them numeric
# under those names), while City/State/Country retain their original capitals.
#
# cluster_id caveat: step_04 fits DBSCAN on Country == "USA" AND latitude
# notna() only, then left-merges back. Every non-US and every ungeocoded row
# therefore has cluster_id = NaN by construction, not by accident. Cluster-level
# aggregation is a US-only view and cannot be the map's primary mode if non-US
# reports are to appear at all.
#
# Coordinates are written RAW. City-centroid geocodes stack points exactly on
# top of one another, but jitter belongs at render time in the map callback — a
# file perturbed for legibility is no longer the file the model scored.
################################################################################
GEO_COLS = [
    "latitude",
    "longitude",
    "City",
    "State",
    "cluster_id",
    "Link",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = {}
    thresholds = {}
    y_test = None

    for col, subdir, model in SPECS:
        pdir = PRED_ROOT / subdir

        frames[col] = pd.read_parquet(pdir / f"y_prob_{model}.parquet").squeeze()

        raw = json.loads((pdir / "model_thresholds.json").read_text())
        thresholds[col] = raw[THRESH_KEYS[model]]

        # Every variant must share one test split, or the columns do not
        # describe the same rows and the whole frame is a lie.
        yt = pd.read_parquet(pdir / "y_test.parquet").squeeze()
        if y_test is None:
            y_test = yt
        else:
            assert y_test.index.equals(yt.index), (
                f"{subdir} test split does not match the first variant — the "
                "runs did not use the same split, so they cannot share a frame."
            )

    df = pd.DataFrame(frames)

    # Align on index, never on row order.
    df["y_val"] = y_test.reindex(df.index)
    assert df["y_val"].notna().all(), "label/probability index mismatch"

    ############################################################################
    # Grouping and geography columns.
    #
    # Joined ON INDEX from the enriched frame — the test rows are a subset of it,
    # and a positional assignment would silently mislabel every subgroup.
    #
    # Country appears in GROUP_COLS and is reused by the map for hover text, so
    # it is not repeated in GEO_COLS. Deduped below to keep one column per name.
    ############################################################################
    have = pd.read_parquet(ENRICHED).columns

    use_groups = [c for c in GROUP_COLS if c in have]
    for c in GROUP_COLS:
        if c not in have:
            print(f"  skip (group): {c} not in {ENRICHED.name}")

    use_geo = [c for c in GEO_COLS if c in have]
    for c in GEO_COLS:
        if c not in have:
            print(f"  skip (geo): {c} not in {ENRICHED.name}")

    if not {"latitude", "longitude"}.issubset(use_geo):
        raise SystemExit(
            f"{ENRICHED.name} has no latitude/longitude — the map panel cannot "
            "be built. Check that step_04_nuforc_analytics.py ran and that its "
            "output_cols still carries the coordinates."
        )

    # One read, one reindex. Order preserved: groups first, then geo, minus any
    # name already taken (Country).
    use = use_groups + [c for c in use_geo if c not in use_groups]
    joined = pd.read_parquet(ENRICHED, columns=use).reindex(df.index)

    ############################################################################
    # Index-alignment guard, scoped to the GROUPING columns only.
    #
    # A test row with no grouping values at all means the index did not align
    # and every subgroup would be misattributed — fatal. A test row with null
    # coordinates is ordinary: reports are ungeocodable for mundane reasons, and
    # folding geo into this check would turn each one into a crash.
    ############################################################################
    unmatched = joined[use_groups].isna().all(axis=1).sum()
    if unmatched:
        raise SystemExit(
            f"{unmatched} test rows have no match in {ENRICHED} — the index does "
            "not align, so subgroups cannot be attributed to the right reports."
        )

    df = pd.concat([df, joined], axis=1)

    ############################################################################
    # Report geographic coverage rather than filtering on it.
    #
    # Rows with null coordinates STAY in models.csv. The map callback drops them
    # and shows the count, so the map's n and every other panel's n can be
    # reconciled by the reader. Dropping here would make them disagree silently.
    ############################################################################
    n_geo = int(df["latitude"].notna().sum())
    n_total = len(df)
    print(f"\n  mappable: {n_geo:,} of {n_total:,} test rows ({n_geo / n_total:.1%})")

    if "cluster_id" in df.columns:
        n_clustered = int(df["cluster_id"].notna().sum())
        n_clusters = int(df["cluster_id"].nunique(dropna=True))
        print(
            f"  in a DBSCAN cluster: {n_clustered:,} rows across {n_clusters:,} "
            "clusters (US-only by construction)"
        )

    df.to_csv(OUT_DIR / "models.csv", index=False)
    (OUT_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    model_cols = [c for c, _, _ in SPECS]
    print(f"\n{df.shape[0]:,} rows x {df.shape[1]} cols -> {OUT_DIR / 'models.csv'}")
    print(
        f"  {len(model_cols)} model columns, {len(use_groups)} grouping columns, "
        f"{len([c for c in use_geo if c not in use_groups])} geo columns\n"
    )
    print(df[model_cols + ["y_val"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
