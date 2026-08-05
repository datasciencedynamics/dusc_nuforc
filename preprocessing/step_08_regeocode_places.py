#!/usr/bin/env python3
"""
Re-geocode NUFORC City/State pairs against the Census Gazetteer.

The existing coordinates have two defects: roughly 37 percent of rows have no
coordinates at all, and among rows that do, some matched a same-named place in
the wrong state or country (Alma GA resolving to Alma, Quebec, for example).

The Gazetteer keys on place name plus USPS state code, so both defects are
addressed in one pass. Places and county subdivisions are both loaded, with
places taking priority, which covers incorporated cities, CDPs, and townships.

Reads and writes parquet or CSV, chosen by file extension.
Downloads are cached under --gaz-dir and skipped on later runs.
"""

from pathlib import Path
from typing import Optional
import io
import re
import urllib.request
import zipfile

import pandas as pd
import typer

app = typer.Typer(add_completion=False, help=__doc__)

GAZ_BASE = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer"

# Suffixes the Gazetteer appends to NAME that never appear in report City fields.
# Order matters: longer phrases stripped before shorter ones.
SUFFIXES = [
    "metropolitan government (balance)",
    "consolidated government (balance)",
    "unified government (balance)",
    "metro government (balance)",
    "urban county government",
    "consolidated government",
    "metropolitan government",
    "unified government",
    "metro government",
    "municipality",
    "township",
    "village",
    "borough",
    "county",
    "city and",
    "town",
    "city",
    "cdp",
    "ccd",
    "ut",
    "(balance)",
]

# Gazetteer placeholder rows that are not real places
JUNK_KEYS = {
    "county subdivisions not defined",
    "municipio subdivision not defined",
    "not defined",
}

ABBREV = [
    (r"^st\.?\s+", "saint "),
    (r"^ste\.?\s+", "sainte "),
    (r"^ft\.?\s+", "fort "),
    (r"^mt\.?\s+", "mount "),
    (r"^pt\.?\s+", "point "),
    (r"^n\.?\s+", "north "),
    (r"^s\.?\s+", "south "),
    (r"^e\.?\s+", "east "),
    (r"^w\.?\s+", "west "),
]


# ------------------------------------------------------------------ io helpers


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read parquet or CSV based on suffix."""
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".csv", ".txt", ".tsv"):
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"unsupported input extension: {suf}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write parquet or CSV based on suffix. Index is preserved for parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        df.to_parquet(path)
    elif suf == ".csv":
        df.to_csv(path)
    else:
        raise ValueError(f"unsupported output extension: {suf}")


# --------------------------------------------------------------- normalization


def normalize(name) -> str:
    """Lowercase, strip punctuation, expand abbreviations, drop legal suffixes."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = s.replace("&", "and")
    for pat, rep in ABBREV:
        s = re.sub(pat, rep, s)
    # suffixes can stack, e.g. "Autaugaville CCD" then nothing left to strip
    changed = True
    while changed:
        changed = False
        for suf in SUFFIXES:
            if s.endswith(" " + suf):
                s = s[: -(len(suf) + 1)].strip()
                changed = True
                break
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ------------------------------------------------------------------- gazetteer


def fetch_gazetteer(kind: str, vintage: str, gaz_dir: Path, force: bool) -> Path:
    """Download and extract one Gazetteer national file. Returns the .txt path."""
    stem = f"{vintage}_Gaz_{kind}_national"
    txt = gaz_dir / f"{stem}.txt"
    if txt.exists() and not force:
        typer.echo(f"  have   {txt.name}")
        return txt

    url = f"{GAZ_BASE}/{vintage}_Gazetteer/{stem}.zip"
    gaz_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"  get    {url}")
    with urllib.request.urlopen(url) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(gaz_dir)
    if not txt.exists():
        cands = list(gaz_dir.glob(f"*Gaz_{kind}_national*.txt"))
        if not cands:
            raise FileNotFoundError(f"no extracted txt for {stem}")
        txt = cands[0]
    typer.echo(f"  got    {txt.name} ({len(payload) / 1e6:.1f} MB)")
    return txt


def load_gazetteer(path: Path, source: str) -> pd.DataFrame:
    """Read one Gazetteer file into name/state/lat/lon/land-area/source."""
    # Recent vintages are UTF-8. Fall back to latin-1 for older files.
    for enc in ("utf-8", "latin-1"):
        try:
            gz = pd.read_csv(path, sep="\t", dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(f"could not decode {path}")

    gz.columns = [c.strip() for c in gz.columns]

    need = {"USPS", "NAME", "INTPTLAT", "INTPTLONG"}
    missing = need - set(gz.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "state": gz["USPS"].str.strip().str.upper(),
            "raw_name": gz["NAME"].str.strip(),
            "lat": pd.to_numeric(gz["INTPTLAT"], errors="coerce"),
            "lon": pd.to_numeric(gz["INTPTLONG"], errors="coerce"),
            "aland": pd.to_numeric(gz.get("ALAND"), errors="coerce"),
            "source": source,
        }
    )
    out["key"] = out["raw_name"].map(normalize)
    out = out[
        (out["key"] != "")
        & ~out["key"].isin(JUNK_KEYS)
        & out["lat"].notna()
        & out["lon"].notna()
    ]
    return out


def build_lookup(frames) -> pd.DataFrame:
    """Collapse to one row per (state, key). Places win; largest land area breaks ties."""
    gz = pd.concat(frames, ignore_index=True)
    gz["prio"] = (gz["source"] == "place").astype(int)
    gz = gz.sort_values(["prio", "aland"], ascending=[False, False])
    gz = gz.drop_duplicates(subset=["state", "key"], keep="first")
    return gz.set_index(["state", "key"])[["lat", "lon", "raw_name", "source"]]


# ------------------------------------------------------------------------ main


@app.command()
def main(
    input_path: Path = typer.Option(..., "--input-path", help="parquet or CSV with City/State/Country"),
    output_path: Path = typer.Option(..., "--output-path", help="where to write the result"),
    gaz_dir: Path = typer.Option(Path("./geo_maps/gazetteer"), "--gaz-dir"),
    vintage: str = typer.Option("2023", "--vintage", help="Gazetteer year"),
    city_col: str = typer.Option("City", "--city-col"),
    state_col: str = typer.Option("State", "--state-col"),
    country_col: str = typer.Option("Country", "--country-col"),
    lat_col: str = typer.Option("latitude", "--lat-col"),
    lon_col: str = typer.Option("longitude", "--lon-col"),
    us_values: str = typer.Option(
        "US,USA,United States,United States of America",
        "--us-values",
        help="comma separated Country values treated as US",
    ),
    keep_original: bool = typer.Option(
        True,
        "--keep-original/--no-keep-original",
        help="preserve the old coordinates as <lat_col>_old / <lon_col>_old",
    ),
    unmatched_csv: Optional[Path] = typer.Option(
        None, "--unmatched-csv", help="optional dump of City/State pairs that did not match"
    ),
    force_download: bool = typer.Option(False, "--force-download"),
):
    """Backfill and correct coordinates from the Census Gazetteer."""

    typer.echo("Gazetteer files")
    places = load_gazetteer(
        fetch_gazetteer("place", vintage, gaz_dir, force_download), "place"
    )
    cousubs = load_gazetteer(
        fetch_gazetteer("cousubs", vintage, gaz_dir, force_download), "cousub"
    )
    lookup = build_lookup([places, cousubs])
    typer.echo(f"  lookup rows: {len(lookup):,}")

    typer.echo(f"\nReading {input_path}")
    df = read_table(input_path, dtype={state_col: str})
    typer.echo(f"  rows: {len(df):,}")

    for col in (city_col, state_col, country_col):
        if col not in df.columns:
            typer.echo(f"  ERROR: no column named {col!r}", err=True)
            typer.echo(f"  available: {list(df.columns)[:40]}", err=True)
            raise typer.Exit(1)

    us_set = {v.strip().lower() for v in us_values.split(",")}
    is_us = df[country_col].astype(str).str.strip().str.lower().isin(us_set)
    typer.echo(f"  US rows: {is_us.sum():,}  non-US rows: {(~is_us).sum():,}")

    if lat_col in df.columns and lon_col in df.columns:
        had_before = df[lat_col].notna() & df[lon_col].notna()
    else:
        df[lat_col] = pd.NA
        df[lon_col] = pd.NA
        had_before = pd.Series(False, index=df.index)
    typer.echo(f"  coordinates before: {had_before.sum():,} ({100 * had_before.mean():.1f}%)")

    keys = pd.MultiIndex.from_arrays(
        [
            df[state_col].astype(str).str.strip().str.upper(),
            df[city_col].map(normalize),
        ]
    )
    matched = lookup.reindex(keys)
    matched.index = df.index

    new_lat = matched["lat"].where(is_us)
    new_lon = matched["lon"].where(is_us)
    hit = new_lat.notna() & new_lon.notna()

    if keep_original:
        df[f"{lat_col}_old"] = df[lat_col]
        df[f"{lon_col}_old"] = df[lon_col]

    df[lat_col] = new_lat.where(hit, df[lat_col])
    df[lon_col] = new_lon.where(hit, df[lon_col])

    df["geocode_source"] = "original"
    df.loc[hit, "geocode_source"] = f"gazetteer_{vintage}"
    had_after = df[lat_col].notna() & df[lon_col].notna()
    df.loc[~had_after, "geocode_source"] = "none"

    typer.echo("\nResult")
    typer.echo(f"  gazetteer matches: {hit.sum():,} ({100 * hit.mean():.1f}% of all rows)")
    if is_us.sum():
        typer.echo(
            f"  US rows matched:   {hit[is_us].sum():,} of {is_us.sum():,} "
            f"({100 * hit[is_us].mean():.1f}%)"
        )
    typer.echo(f"  coordinates after: {had_after.sum():,} ({100 * had_after.mean():.1f}%)")
    typer.echo(f"  newly geocoded:    {(~had_before & had_after).sum():,}")
    typer.echo(f"  coords replaced:   {(had_before & hit).sum():,}")
    typer.echo(f"  still missing:     {(~had_after).sum():,}")

    miss = df[is_us & ~hit]
    if len(miss):
        typer.echo("\nTop unmatched US City/State pairs")
        top = miss.groupby([city_col, state_col]).size().sort_values(ascending=False).head(20)
        for (c, s), n in top.items():
            typer.echo(f"  {n:>4}  {c}, {s}")

    if unmatched_csv is not None and len(miss):
        pairs = (
            miss.groupby([city_col, state_col])
            .size()
            .rename("n")
            .reset_index()
            .sort_values("n", ascending=False)
        )
        unmatched_csv.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(unmatched_csv, index=False)
        typer.echo(f"\nUnmatched pairs written to {unmatched_csv}")

    write_table(df, output_path)
    typer.echo(f"\nWrote {output_path}")


if __name__ == "__main__":
    app()