#!/usr/bin/env python3
"""
nuforc_pipeline.py
==================
Replication + extension of Posard et al. (2023) "Not the X-Files" RAND report.

Reads the feature-engineered parquet produced by nlp_feature_engineer_nuforc.py
and runs spatial clustering, descriptive statistics, regression modeling, and
chi-square tests before exporting the final enriched dataset.

Steps
-----
  1. Load enriched parquet
  2. DBSCAN spatial clustering (Python-native proxy for Kulldorff scan stats)
  3. Descriptive statistics (yearly breakdown, shape counts, report lag)
  4. Negative Binomial regression (mirrors Table 3.1 from RAND)
  5. Chi-square tests (shape group vs. explanation / night / media)
  6. Export final parquet + CSV with cluster columns appended

Requirements
------------
    pip install pandas numpy scikit-learn statsmodels scipy loguru typer --break-system-packages
"""

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import statsmodels.formula.api as smf
from scipy import stats
import typer
from loguru import logger

app = typer.Typer()


# #############################################################################
# Main
# #############################################################################


@app.command()
def main(
    input_parquet: str = "./data/processed/nuforc_engineered.parquet",
    output_parquet: str = "./data/processed/NUFORC_enriched.parquet",
):
    """
    Run spatial clustering, analytics, and regression on the enriched NUFORC
    dataset. Appends cluster_id and in_cluster columns and exports final outputs.
    """
    logger.info(f"Running script: {os.path.basename(__file__)}")

    # #########################################################################
    # Step 1 -- Load enriched parquet from nlp_feature_engineer_nuforc.py
    # #########################################################################
    logger.info("STEP 1 -- Load enriched parquet")

    try:
        df = pd.read_parquet(input_parquet)
    except Exception as e:
        logger.error(f"Failed to load parquet: {e}")
        raise

    # Ensure lat/lon are numeric -- may be object dtype if loaded from CSV
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    logger.info(f"  Rows loaded : {len(df):,}")
    logger.info(f"  Columns     : {len(df.columns)}")

    # #########################################################################
    # Step 2 -- DBSCAN spatial clustering
    #
    # RAND used Kulldorff scan statistics (requires SaTScan software).
    # DBSCAN with haversine distance is a Python-native proxy for detecting
    # spatial hotspots of UAP sightings.
    #
    # eps         = 30 km  (matches the RAND paper's 30-km MOA proximity threshold)
    # min_samples = 3      (minimum sightings to form a cluster)
    # cluster_id  = NaN    means the record is noise (not in any cluster)
    # #########################################################################
    logger.info("STEP 2 -- Spatial clustering (DBSCAN, eps=30km)")

    # Filter to geocoded US records only for clustering
    geo_us = df[(df["Country"] == "USA") & df["latitude"].notna()].copy()

    EARTH_RADIUS_KM = 6371.0
    EPS_KM = 30.0
    coords_rad = np.radians(geo_us[["latitude", "longitude"]].values)

    db = DBSCAN(
        eps=EPS_KM / EARTH_RADIUS_KM,
        min_samples=3,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords_rad)

    # Replace DBSCAN noise label (-1) with NaN before saving — -1 is a sentinel
    # value with no ordinal meaning and would be misleading as a numeric feature.
    cluster_labels = db.labels_.astype(float)
    cluster_labels[cluster_labels == -1] = np.nan

    geo_us["cluster_id"] = cluster_labels  # NaN = noise (not in any cluster)
    geo_us["in_cluster"] = (db.labels_ >= 0).astype(int)

    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    in_cluster_pct = geo_us["in_cluster"].mean() * 100
    logger.info(f"  Geocoded US records   : {len(geo_us):,}")
    logger.info(f"  DBSCAN clusters found : {n_clusters}")
    logger.info(
        f"  Records in a cluster  : {geo_us['in_cluster'].sum():,} ({in_cluster_pct:.1f}%)"
    )

    # Merge cluster columns back to full dataframe.
    # Non-US and ungeocodeable rows receive NaN / 0 respectively.
    df = df.merge(
        geo_us[["cluster_id", "in_cluster"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    df["cluster_id"] = df["cluster_id"].fillna(np.nan)  # NaN = no cluster
    df["in_cluster"] = df["in_cluster"].fillna(0).astype(int)

    # Summarise the top 10 largest clusters for logging
    # pct_certain = fraction of sightings with a definitive (non-?) explanation
    top_clusters = (
        geo_us[geo_us["cluster_id"].notna()]
        .groupby("cluster_id")
        .agg(
            n_sightings=("cluster_id", "count"),
            lat_center=("latitude", "mean"),
            lon_center=("longitude", "mean"),
            top_shape=("Shape", lambda x: x.value_counts().index[0]),
            pct_certain=("exp_certain", "mean"),
        )
        .sort_values("n_sightings", ascending=False)
        .head(10)
    )
    logger.info(f"  Top 10 spatial clusters:\n{top_clusters.to_string()}")

    # #########################################################################
    # Step 3 -- Descriptive statistics
    # #########################################################################
    logger.info("STEP 3 -- Descriptive statistics")

    # Yearly breakdown -- mirrors Table A.1 from RAND
    # pct_certain = % of sightings with a definitive NUFORC explanation
    yearly = (
        df.groupby("occurred_year")
        .agg(
            n_sightings=("occurred_year", "count"),
            pct_certain=("exp_certain", "mean"),
            pct_night=("is_night", "mean"),
            pct_weekend=("is_weekend", "mean"),
            pct_media=("has_media", "mean"),
            pct_in_cluster=("in_cluster", "mean"),
        )
        .reset_index()
    )
    for col in [
        "pct_certain",
        "pct_night",
        "pct_weekend",
        "pct_media",
        "pct_in_cluster",
    ]:
        yearly[col] = (yearly[col] * 100).round(1)

    logger.info(f"  Sightings by year:\n{yearly.to_string(index=False)}")
    logger.info(
        f"  Shape group counts:\n{df['shape_group'].value_counts().to_string()}"
    )
    logger.info(
        f"  Top explanation categories:\n{df['Explanation'].value_counts().head(12).to_string()}"
    )

    lag = pd.to_numeric(df["report_lag_days"], errors="coerce")
    logger.info(
        f"  Report lag (days) -- median: {lag.median():.0f}, "
        f"mean: {lag.mean():.1f}, max: {lag.max():.0f}"
    )

    # #########################################################################
    # Step 4 -- Chi-square tests
    #
    # Tests whether shape group is independent of explanation certainty, time
    # of night, and media presence. All are expected to be significant given
    # known patterns (e.g. luminous shapes skew heavily nocturnal).
    # #########################################################################
    logger.info("STEP 5 -- Chi-square: shape group vs. explanation type")

    for description, col in [
        ("exp_certain", "exp_certain"),
        ("is_night", "is_night"),
        ("has_media", "has_media"),
    ]:
        ct = pd.crosstab(df["shape_group"], df[col])
        chi2_val, p_val, dof, _ = stats.chi2_contingency(ct)
        logger.info(
            f"  Shape group vs. {description}: chi2={chi2_val:.1f}, df={dof}, p={p_val:.4f}"
        )

    # Night-time rate broken down by shape group
    night_by_shape = df.groupby("shape_group")["is_night"].agg(["mean", "count"])
    night_by_shape["pct_night"] = (night_by_shape["mean"] * 100).round(1)
    logger.info(
        f"  Night-time rate by shape group:\n"
        f"{night_by_shape[['pct_night','count']].sort_values('pct_night', ascending=False).to_string()}"
    )

    # #########################################################################
    # Step 5 -- Export final enriched dataset
    #
    # Selects the modeling-ready columns and appends the cluster columns
    # computed in Step 2. Summary is retained for CatBoost text_features.
    # summary_clean is the stopword-filtered version for the same purpose.
    # exp_any is intentionally excluded -- it is a deterministic combination
    # of the individual exp_* flags and adds no new information.
    # #########################################################################
    logger.info("STEP 6 -- Export")

    output_cols = [
        "Link",
        "Occurred",
        "Reported",
        "report_lag_days",
        "City",
        "State",
        "Country",
        "latitude",
        "longitude",
        "Shape",
        "shape_group",
        "Explanation",
        "exp_drone",
        "exp_rocket",
        "exp_balloon",
        "exp_aircraft",
        "exp_starlink",
        "exp_lantern",
        "exp_satellite",
        "exp_certain",
        "has_media",
        "occurred_year",
        "occurred_month",
        "occurred_hour",
        "is_night",
        "is_weekend",
        "location_count_total",
        "summary_token_count",
        "days_since_uap_event",
        "cluster_id",
        "in_cluster",
        "summary_clean",
        "Summary",
    ]

    df_out = df[[c for c in output_cols if c in df.columns]].copy()
    df_out.to_parquet(output_parquet, index=True)
    logger.info(
        f"  Enriched parquet saved : {output_parquet}  ({len(df_out):,} rows, {len(df_out.columns)} cols)"
    )

    output_csv = output_parquet.replace(".parquet", ".csv")
    df_out.to_csv(output_csv, index=True, encoding="utf-8-sig")
    logger.info(f"  Enriched CSV saved     : {output_csv}")

    logger.info("nuforc_pipeline complete.")


if __name__ == "__main__":
    app()
