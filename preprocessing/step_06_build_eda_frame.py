from pathlib import Path
import csv
import sys

csv.field_size_limit(sys.maxsize)

import typer
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import PROCESSED_DATA_DIR

app = typer.Typer()


################################################################################
# Helpers
################################################################################


def _read_frame(path: Path, label: str) -> pd.DataFrame:
    """Read parquet or CSV based on the file suffix."""

    suffix = path.suffix.lower()

    if suffix == ".csv":
        logger.warning(
            f"Reading {label} from CSV. Parquet is preferred: CSV round-trips of "
            f"narrative text can shift rows on embedded newlines or quotes, and "
            f"dtypes are inferred rather than preserved."
        )
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise typer.BadParameter(
        f"Unsupported input format '{suffix}' for {label}: {path}. "
        f"Use .parquet or .csv."
    )


def _write_frame(df: pd.DataFrame, path: Path, fmt: str) -> Path:
    """Write the frame as parquet, CSV, or both. Returns the primary path."""

    fmt = fmt.lower()
    valid = {"parquet", "csv", "both"}
    if fmt not in valid:
        raise typer.BadParameter(
            f"--output-format must be one of {sorted(valid)}, got '{fmt}'."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")

    written = []

    if fmt in {"parquet", "both"}:
        pq_path = stem.with_suffix(".parquet")
        df.to_parquet(pq_path)
        written.append(pq_path)

    if fmt in {"csv", "both"}:
        csv_path = stem.with_suffix(".csv")
        # index=True so report_id survives the round trip when it is the index.
        df.to_csv(csv_path, index=df.index.name is not None)
        written.append(csv_path)
        logger.warning(
            "CSV written. Treat parquet as the source of truth: CSV loses dtypes "
            "and is fragile around embedded newlines in the narrative columns."
        )

    for p in written:
        logger.success(f"Wrote {p} {df.shape}")

    return written[0]


def _align_key(df: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    """Return a frame with `key` guaranteed to be a column, not the index."""

    if df.index.name == key:
        df = df.reset_index()

    if key not in df.columns:
        raise typer.BadParameter(
            f"'{key}' not found in the {label} frame as a column or index. "
            f"Available: {sorted(df.columns.tolist())[:25]}"
        )

    return df


################################################################################
# Main
################################################################################


@app.command()
def main(
    raw_path: Path = Path("./data/raw/nuforc_data.parquet"),
    model_path: Path = PROCESSED_DATA_DIR / "df_final.parquet",
    enriched_path: Path = PROCESSED_DATA_DIR / "NUFORC_enriched.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "df_eda.parquet",
    output_format: str = "parquet",
    join_key: str = "report_id",
    how: str = "inner",
    keep_index: int = 1,
    geo_cols: str = "latitude,longitude,cluster_id,in_cluster",
):
    """
    Build the EDA dataframe.

    Joins the raw NUFORC report frame to the engineered modeling frame on
    join_key, producing a single wide frame that carries both the original
    report fields (for interpretation and reporting) and the engineered
    features and labels (for analysis).

    Columns present in both frames are kept from the raw side and suffixed
    _model on the processed side, so nothing is silently dropped and the
    provenance of every column stays legible.

    Geography is joined back separately from the enriched frame. latitude and
    longitude are listed in core.constants.drop_vars, so step_05 removes them
    before the modeling frame is built and they cannot be recovered from
    df_final. They do survive in NUFORC_enriched.parquet, which is step_04's
    output and therefore upstream of the drop. Pass --geo-cols "" to skip.

    Inputs may be .parquet or .csv, detected by suffix. Output format is set by
    --output-format: parquet (default), csv, or both. The suffix on
    --output-path is replaced to match, so only the stem matters.
    """

    ############################################################################
    # Step 1. Load both frames
    ############################################################################

    logger.info(f"Raw frame:       {raw_path}")
    df_raw = _read_frame(raw_path, "raw")
    logger.info(f"  shape: {df_raw.shape}")

    logger.info(f"Modeling frame:  {model_path}")
    df_model = _read_frame(model_path, "processed")
    logger.info(f"  shape: {df_model.shape}")

    ############################################################################
    # Step 2. Align the join key
    # The key lives on the index in the processed frame and as a column in the
    # raw frame, so a bare .join() would match the raw key against the
    # processed INDEX. Normalize both sides to a column and merge explicitly.
    ############################################################################

    left = _align_key(df_raw, join_key, "raw")
    right = _align_key(df_model, join_key, "processed")

    # Keys must share a dtype or the merge silently returns nothing.
    if left[join_key].dtype != right[join_key].dtype:
        logger.warning(
            f"Join key dtype mismatch: raw={left[join_key].dtype}, "
            f"processed={right[join_key].dtype}. Casting both to str."
        )
        left[join_key] = left[join_key].astype(str)
        right[join_key] = right[join_key].astype(str)

    overlap = sorted(set(left.columns) & set(right.columns) - {join_key})
    if overlap:
        logger.info(f"{len(overlap)} overlapping columns suffixed _model: {overlap}")

    ############################################################################
    # Step 3. Merge
    ############################################################################

    df_eda = left.merge(right, on=join_key, how=how, suffixes=("", "_model"))

    n_dupe = int(df_eda[join_key].duplicated().sum())
    if n_dupe:
        logger.warning(
            f"{n_dupe} duplicate {join_key} values in the result. One side has a "
            f"non-unique key, so this fanned out rather than joining 1:1."
        )

    ############################################################################
    # Step 3b. Join geography back from the enriched frame
    #
    # Only columns MISSING from df_eda are pulled. City and State, for example,
    # are in drop_vars but also exist on the raw side, so requesting them here
    # would collide with columns already present. Skipping rather than
    # suffixing keeps a single unambiguous City column instead of a City and a
    # City_geo that a reader has to choose between.
    ############################################################################

    wanted = [c.strip() for c in geo_cols.split(",") if c.strip()]

    if wanted:
        if not Path(enriched_path).exists():
            logger.warning(
                f"{enriched_path} not found. Skipping the geography join; "
                f"latitude and longitude will be absent from the EDA frame."
            )
        else:
            logger.info(f"Geography frame: {enriched_path}")
            df_geo = _align_key(
                _read_frame(enriched_path, "enriched"), join_key, "enriched"
            )

            if df_geo[join_key].dtype != df_eda[join_key].dtype:
                df_geo[join_key] = df_geo[join_key].astype(str)
                df_eda[join_key] = df_eda[join_key].astype(str)

            absent = [c for c in wanted if c not in df_geo.columns]
            if absent:
                logger.warning(
                    f"Not in {Path(enriched_path).name}, skipping: "
                    f"{', '.join(absent)}"
                )

            collide = [c for c in wanted if c in df_eda.columns]
            if collide:
                logger.info(
                    f"Already present in the EDA frame, not re-joined: "
                    f"{', '.join(collide)}"
                )

            take = [
                c for c in wanted if c in df_geo.columns and c not in df_eda.columns
            ]

            if take:
                # LEFT join, never inner: a report with no geography must keep
                # its row here. An inner join would quietly shrink the EDA
                # frame to the geocoded subset, and every downstream count
                # would then describe a filtered corpus without saying so.
                before = len(df_eda)
                df_eda = df_eda.merge(
                    df_geo[[join_key] + take], on=join_key, how="left"
                )
                assert len(df_eda) == before, "geography join changed the row count"

                logger.success(f"Joined geography: {', '.join(take)}")
                for c in take:
                    n_null = int(df_eda[c].isna().sum())
                    logger.info(
                        f"  {c}: {n_null:,} null "
                        f"({n_null / len(df_eda):.1%} of rows)"
                    )

    print()
    print("=" * 60)
    print("EDA frame")
    print("=" * 60)
    print(f"  raw rows:          {len(left)}")
    print(f"  processed rows:    {len(right)}")
    print(f"  joined rows:       {len(df_eda)}")
    print(f"  columns:           {df_eda.shape[1]}")
    print(f"  raw coverage:      {len(df_eda) / len(left):.2%}")
    print(f"  processed coverage:{len(df_eda) / len(right):.2%}")
    print()

    ############################################################################
    # Step 4. Write
    ############################################################################

    if keep_index:
        df_eda = df_eda.set_index(join_key)

    _write_frame(df_eda, output_path, output_format)

    logger.success("EDA frame complete.")


if __name__ == "__main__":
    app()
