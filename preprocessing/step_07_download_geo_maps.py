#!/usr/bin/env python3
"""
Fetch the geospatial layers the NUFORC notebooks depend on.

These files are gitignored, so run this once after cloning. Idempotent by
default: anything already on disk is skipped unless --force is passed.
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master"
TIGER_BASE = "https://www2.census.gov/geo/tiger"

# shp/shx/dbf are required to read a layer, prj carries the CRS,
# cpg is an encoding hint that does not exist for every theme
NE_EXTS = ["shp", "shx", "dbf", "prj", "cpg"]
NE_REQUIRED = {"shp", "shx", "dbf"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-path",
        required=True,
        help="root directory for downloaded layers, e.g. ./geo_maps",
    )
    p.add_argument(
        "--world-subdir",
        default="world",
        help="subdirectory under output-path for Natural Earth layers",
    )
    p.add_argument(
        "--us-subdir",
        default="us",
        help="subdirectory under output-path for TIGER layers",
    )
    p.add_argument(
        "--ne-scale",
        default="110m",
        choices=["110m", "50m", "10m"],
        help="Natural Earth resolution",
    )
    p.add_argument(
        "--ne-themes",
        nargs="+",
        default=[
            "admin_0_countries",
            "admin_0_boundary_lines_land",
        ],
        help="Natural Earth cultural theme names, without the scale prefix",
    )
    p.add_argument(
        "--tiger-vintage",
        default="2023",
        help="Census TIGER year",
    )
    p.add_argument(
        "--tiger-layers",
        nargs="+",
        default=["STATE"],
        help="TIGER layer directories to pull, e.g. STATE COUNTY",
    )
    p.add_argument(
        "--force",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 to re-download files that already exist",
    )
    p.add_argument(
        "--verify",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 to read every downloaded shapefile back through geopandas",
    )
    return p.parse_args()


def fetch(url, dest, force=False):
    """Download url to dest. Returns True if a file is on disk afterward."""
    if dest.exists() and not force:
        print(f"  have   {dest.name}")
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  got    {dest.name}")
        return True
    except Exception as exc:
        print(f"  FAILED {dest.name}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def fetch_zip(url, dest_dir, force=False):
    """Download a zip and extract it into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(url).stem
    marker = dest_dir / f"{stem}.shp"

    if marker.exists() and not force:
        print(f"  have   {stem}/")
        return True

    try:
        with urllib.request.urlopen(url) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(dest_dir)
        print(f"  got    {stem}/ ({len(payload) / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  FAILED {stem}: {exc}")
        return False


def download_natural_earth(dest, scale, themes, force):
    print(f"Natural Earth ({scale}) -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for theme in themes:
        name = f"ne_{scale}_{theme}"
        for ext in NE_EXTS:
            url = f"{NE_BASE}/{scale}_cultural/{name}.{ext}"
            landed = fetch(url, dest / f"{name}.{ext}", force)
            if not landed and ext in NE_REQUIRED:
                ok = False
    return ok


def download_tiger(dest, vintage, layers, force):
    print(f"\nCensus TIGER ({vintage}) -> {dest}")
    ok = True
    for layer in layers:
        fname = f"tl_{vintage}_us_{layer.lower()}.zip"
        url = f"{TIGER_BASE}/TIGER{vintage}/{layer}/{fname}"
        if not fetch_zip(url, dest, force):
            ok = False
    return ok


def verify(root):
    print("\nVerifying with geopandas")
    try:
        import geopandas as gpd
    except ImportError:
        print("  geopandas not installed, skipping read check")
        return True

    ok = True
    shapefiles = sorted(root.rglob("*.shp"))
    if not shapefiles:
        print("  no shapefiles found")
        return False

    for shp in shapefiles:
        try:
            gdf = gpd.read_file(shp)
            crs = gdf.crs.to_string() if gdf.crs else "no CRS"
            print(f"  {shp.relative_to(root)}: {len(gdf)} rows, {crs}")
        except Exception as exc:
            print(f"  {shp.relative_to(root)}: unreadable ({exc})")
            ok = False
    return ok


def main():
    args = parse_args()
    root = Path(args.output_path).expanduser().resolve()
    force = bool(args.force)

    ok = download_natural_earth(
        root / args.world_subdir, args.ne_scale, args.ne_themes, force
    )
    ok &= download_tiger(
        root / args.us_subdir, args.tiger_vintage, args.tiger_layers, force
    )

    if args.verify:
        ok &= verify(root)

    print("\nDone." if ok else "\nDone with errors, see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())