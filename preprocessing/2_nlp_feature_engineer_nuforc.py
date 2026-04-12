#!/usr/bin/env python3
"""
nlp_feature_engineer_nuforc.py
===============================
Reads the enriched NUFORC parquet (output of data_gen.py) and produces a
feature-engineered parquet and CSV for downstream CatBoost modeling.

Output columns
--------------
  Temporal    : occurred_year, occurred_month, occurred_day, occurred_hour,
                report_lag_days, is_night, is_weekend
  Location    : location_count_total, latitude, longitude
  Text        : summary_token_count, summary_clean
  Shape       : shape_group
  Explanation : exp_drone, exp_rocket, exp_balloon, exp_aircraft,
                exp_starlink, exp_lantern, exp_satellite, exp_certain
  Media       : has_media
  UAP Context : days_since_uap_event
  Cluster     : cluster_id, in_cluster  (added by nuforc_pipeline.py)
"""

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

app = typer.Typer()


# #############################################################################
# Constants
# #############################################################################

TOKEN_RE = re.compile(r"[a-z0-9']+")

# Valid US state and Canadian province codes for state validation
VALID_US_STATES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}

VALID_CA_PROVINCES = {
    "AB",
    "BC",
    "MB",
    "NB",
    "NL",
    "NS",
    "NT",
    "NU",
    "ON",
    "PE",
    "QC",
    "SK",
    "YT",
}

SHAPE_MAP = {
    "Orb": "luminous",
    "Light": "luminous",
    "Fireball": "luminous",
    "Flash": "luminous",
    "Circle": "circular",
    "Sphere": "circular",
    "Oval": "circular",
    "Disk": "classic_ufo",
    "Saucer": "classic_ufo",
    "Triangle": "angular",
    "Rectangle": "angular",
    "Diamond": "angular",
    "Chevron": "angular",
    "Boomerang": "angular",
    "Cigar": "elongated",
    "Cylinder": "elongated",
    "Torpedo": "elongated",
    "Cone": "elongated",
    "Formation": "multi_object",
    "Changing": "other",
    "Unknown": "other",
    "Other": "other",
}

MANUAL_OVERRIDES = {
    ("new york", "NY"): (40.7128, -74.0060),
    ("new york city", "NY"): (40.7128, -74.0060),
    ("nyc", "NY"): (40.7128, -74.0060),
    ("brooklyn", "NY"): (40.6782, -73.9442),
    ("queens", "NY"): (40.7282, -73.7949),
    ("bronx", "NY"): (40.8448, -73.8648),
    ("manhattan", "NY"): (40.7831, -73.9712),
    ("staten island", "NY"): (40.5795, -74.1502),
    ("los angeles", "CA"): (34.0522, -118.2437),
    ("san francisco", "CA"): (37.7749, -122.4194),
    ("malibu", "CA"): (34.0259, -118.7798),
    ("los alamitos", "CA"): (33.7976, -118.0723),
    ("key largo", "FL"): (25.0865, -80.4473),
    ("panama city beach", "FL"): (30.1766, -85.8055),
    ("port st. lucie", "FL"): (27.2930, -80.3503),
    ("port st lucie", "FL"): (27.2930, -80.3503),
    ("sedona", "AZ"): (34.8697, -111.7609),
    ("snohomish", "WA"): (47.9129, -122.0982),
    ("port orchard", "WA"): (47.5401, -122.6326),
    ("bemidji", "MN"): (47.4736, -94.8803),
    ("ocean city", "MD"): (38.3365, -75.0849),
    ("ocean city", "NJ"): (39.2776, -74.5746),
    ("st. george", "UT"): (37.1041, -113.5841),
    ("red lion", "PA"): (39.8990, -76.6035),
    ("mechanicsburg", "PA"): (40.2140, -77.0086),
    ("quakertown", "PA"): (40.4418, -75.3418),
    ("cumming", "GA"): (34.2073, -84.1402),
    ("blairsville", "GA"): (34.8762, -83.9585),
    ("tomball", "TX"): (30.0971, -95.6160),
    ("lamar", "CO"): (38.0875, -102.6213),
    ("london", "KY"): (37.1290, -84.0830),
    ("lander", "WY"): (42.8330, -108.7307),
    ("youngsville", "LA"): (30.1024, -91.9982),
    ("freetown", "MA"): (41.7640, -71.0173),
    ("dennis", "MA"): (41.7354, -70.1940),
    ("pawleys island", "SC"): (33.4374, -79.1203),
    ("irmo", "SC"): (34.0818, -81.1820),
    ("birch tree", "MO"): (36.9892, -91.5048),
    ("almont", "MI"): (42.9215, -83.0436),
    ("laughlin", "NV"): (35.1678, -114.5733),
    ("collins", "GA"): (32.1851, -82.1146),
}

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "see",
    "saw",
    "seen",
    "look",
    "looked",
    "looking",
    "thing",
    "things",
    "something",
    "went",
    "go",
    "gone",
    "got",
    "come",
    "came",
}


# #############################################################################
# UAP Disclosure Events
#
# Ordered list of major U.S. government UAP disclosure events used to compute
# `days_since_uap_event`: the number of days elapsed between a sighting and
# the most recent disclosure event that preceded it. NUFORC reporting volume
# spikes following congressional hearings and DoD report releases, so this
# feature captures how much public UAP discourse was active at the time a
# report was filed. A short lag may indicate media-driven reporting; a long
# lag may indicate a more isolated, organic observation.
#
# All dates are sourced from official U.S. government records, congressional
# archives, or C-SPAN broadcast records. Citations per event:
#
#   2021-06-25  ODNI Preliminary Assessment: Unidentified Aerial Phenomena.
#               Released per Section 1603, William M. (Mac) Thornberry
#               National Defense Authorization Act for FY2021. 144 UAP
#               incidents examined, largely unexplained.
#               Source: dni.gov/files/ODNI/documents/assessments/
#                       Prelimary-Assessment-UAP-20210625.pdf
#
#   2022-05-17  U.S. House of Representatives, Permanent Select Committee on
#               Intelligence, Subcommittee on Counterterrorism,
#               Counterintelligence, and Counterproliferation: first public
#               congressional UAP hearing in over 50 years. Witnesses:
#               Under Secretary of Defense Ronald S. Moultrie and Deputy
#               Director of Naval Intelligence Scott Bray.
#               Source: intelligence.house.gov; c-span.org
#
#   2022-07-15  DoD formally establishes AARO (All-domain Anomaly Resolution
#               Office) to replace the UAP Task Force and AOIMSG, per NDAA
#               FY2022 (50 U.S.C. § 3373). Deputy Secretary of Defense
#               Kathleen Hicks signed the directive.
#               Source: defense.gov press release, July 20 2022
#
#   2023-01-12  ODNI submits FY2022 Annual Report on Unidentified Aerial
#               Phenomena to Congress (510 total cases as of August 2022,
#               drafted in partnership with AARO). 171 of 366 new cases
#               remain uncharacterized.
#               Source: dni.gov/files/ODNI/documents/assessments/
#                       2022-Annual-Report-UAP.pdf
#
#   2023-04-19  U.S. Senate Armed Services Subcommittee on Emerging Threats
#               and Capabilities hearing on AARO, chaired by Senator Kirsten
#               Gillibrand. First dedicated Senate hearing focused on AARO
#               operations and reporting.
#               Source: armed-services.senate.gov; c-span.org
#
#   2023-07-26  U.S. House Oversight and Accountability Subcommittee on
#               National Security, the Border, and Foreign Affairs hearing:
#               "Unidentified Anomalous Phenomena: Implications on National
#               Security, Public Safety, and Government Transparency."
#               David Grusch (whistleblower, former NGA/NRO), David Fravor
#               (former Navy Commander), and Ryan Graves (former Navy pilot)
#               testified. Grusch alleged secret UAP retrieval programs.
#               Highest single-event public media impact in the dataset window.
#               Source: oversight.house.gov; c-span.org broadcast record
#
#   2023-08-30  AARO public-facing website (aaro.mil) officially launched,
#               enabling broader public awareness and providing a reporting
#               pathway for current/former USG personnel. Required by NDAA
#               FY2022 but delayed by Pentagon red tape (Politico, Aug 2023).
#               Source: defense.gov press release; Politico Aug 10 2023
#
#   2023-09-14  NASA UAP Independent Study Team final report released.
#               Focused on identifying optimal data streams for UAP case
#               resolution. NASA also announced creation of a UAP Research
#               Director position. Did not address extraterrestrial origin.
#               Source: nasa.gov/feature/nasa-releases-independent-study-
#                       team-s-report-on-unidentified-anomalous-phenomena
#
#   2024-03-06  DoD clears for public release: "Report on the Historical
#               Record of U.S. Government Involvement with Unidentified
#               Anomalous Phenomena (UAP) Volume I." AARO reviewed all
#               official USG UAP investigatory efforts since 1945 across ~30
#               interviews and classified/unclassified archives. Found "no
#               empirical evidence" of extraterrestrial technology.
#               Source: media.defense.gov/2024/Mar/08/2003409233/
#                       DOPSR-CLEARED-508-COMPLIANT-HRRV1-08-MAR-2024-FINAL.PDF
#
#   2024-11-13  U.S. House Oversight and Accountability joint subcommittee
#               hearing: "Unidentified Anomalous Phenomena: Exposing the
#               Truth." Witnesses: Retired Rear Admiral Tim Gallaudet, Luis
#               Elizondo (former DoD AATIP), Michael Gold (former NASA), and
#               Michael Shellenberger. Shellenberger introduced a 12-page
#               document alleging the "Immaculate Constellation" program.
#               Source: oversight.house.gov/hearing/
#                       unidentified-anomalous-phenomena-exposing-the-truth;
#                       burchett.house.gov press release Nov 8 2024
#
#   2024-11-14  DoD and ODNI deliver AARO FY2024 Annual Report on UAP to
#               Congress (fourth annual report; third authored by AARO).
#               Covers May 2023 – June 2024: 757 new reports received, 21
#               cases merit further analysis, 3 incidents involve military
#               aircrews being "trailed or shadowed" by UAP.
#               Source: defense.gov; dni.gov/files/ODNI/documents/assessments/
#                       DOD-AARO-Consolidated-Annual-Report-on-UAP-Nov2024.pdf
# #############################################################################

UAP_EVENTS = [
    pd.Timestamp("2021-06-25"),  # ODNI preliminary UAP assessment
    pd.Timestamp("2022-05-17"),  # First public congressional UAP hearing in 50+ years
    pd.Timestamp("2022-07-15"),  # AARO established by DoD
    pd.Timestamp("2023-01-12"),  # ODNI/AARO FY2022 annual report to Congress
    pd.Timestamp("2023-04-19"),  # Senate Armed Services subcommittee hearing on AARO
    pd.Timestamp("2023-07-26"),  # Grusch whistleblower testimony (highest media impact)
    pd.Timestamp("2023-08-30"),  # AARO public website launched
    pd.Timestamp("2023-09-14"),  # NASA UAP independent study report released
    pd.Timestamp("2024-03-06"),  # AARO Historical Record Report Vol 1
    pd.Timestamp("2024-11-13"),  # House Oversight hearing "Exposing the Truth"
    pd.Timestamp("2024-11-14"),  # AARO FY2024 annual report released (757 cases)
]


# #############################################################################
# Date / time helpers
# #############################################################################


def to_datetime_safe(value):
    """Convert a value to datetime, handling Timestamps, strings, and NaT."""
    if value is None or value == "" or pd.isnull(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def to_date_safe(value):
    """Convert a value to a date object, handling Timestamps, strings, and NaT."""
    if value is None or value == "" or pd.isnull(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date()
    dt = to_datetime_safe(value)
    return dt.date() if dt else None


# #############################################################################
# Geocoding
# #############################################################################


def build_geocode_lookups():
    """
    Build city-state and city-only lookup dicts from geonamescache for all
    countries (not just US), injecting manual overrides for cities missing
    or misnamed in the cache. The city-only fallback retains the highest-
    population match per city name to reduce ambiguity.
    Returns (city_state_lookup, city_only_lookup).
    """
    import geonamescache

    gc = geonamescache.GeonamesCache()
    all_cities = gc.get_cities()

    city_state_lookup = {}
    city_only_lookup = {}
    city_pop_cache = {}

    for v in all_cities.values():
        state = v.get("admin1code", "").upper()
        name = v["name"].lower()
        lat, lon = float(v["latitude"]), float(v["longitude"])
        pop = v.get("population", 0) or 0
        city_state_lookup[(name, state)] = (lat, lon)
        if name not in city_pop_cache or pop > city_pop_cache[name]:
            city_pop_cache[name] = pop
            city_only_lookup[name] = (lat, lon)

    # Inject manual overrides for common cities missing from geonamescache
    for (city_lower, state), coords in MANUAL_OVERRIDES.items():
        city_state_lookup[(city_lower, state)] = coords
        if city_lower not in city_only_lookup:
            city_only_lookup[city_lower] = coords

    return city_state_lookup, city_only_lookup


def geocode_city(city, state, country, city_state_lookup, city_only_lookup):
    """
    Return (latitude, longitude, geocode_method) for a city/state/country.
    Geocodes records from all countries using geonamescache lookups.
    Tries city+state match first, falls back to city-only if no state match.
    Returns empty strings and 'missing' if city is absent, or 'unmatched'
    if no lookup entry is found.
    """
    if not city:
        return "", "", "missing"
    c = str(city).strip().lower()
    s = str(state).strip().upper()
    if (c, s) in city_state_lookup:
        lat, lon = city_state_lookup[(c, s)]
        return lat, lon, "state_match"
    if c in city_only_lookup:
        lat, lon = city_only_lookup[c]
        return lat, lon, "city_fallback"
    return "", "", "unmatched"


def validate_state(state, country):
    s = str(state or "").strip().upper()
    if country == "USA" and s not in VALID_US_STATES:
        return "Unknown"
    if country == "Canada" and s not in VALID_CA_PROVINCES:
        return "Unknown"
    return str(state or "").strip()


# #############################################################################
# Feature engineering helpers
# #############################################################################


def normalize_location_part(value, fallback):
    """Lowercase, strip, and collapse whitespace for a location string."""
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    return re.sub(r"\s+", " ", text)


def build_location_key(city, state, country):
    """Build a pipe-delimited city|state|country key for location grouping."""
    return "|".join(
        [
            normalize_location_part(city, "unknown_city"),
            normalize_location_part(state, "unknown_state"),
            normalize_location_part(country, "unknown_country"),
        ]
    )


def tokenize_summary(summary):
    """
    Light tokenization for CatBoost text_features input.
    Removes stopwords, short tokens, and pure digits.
    CatBoost handles its own internal tokenization; this is NOT TF-IDF.
    """
    tokens = []
    for raw in TOKEN_RE.findall((summary or "").lower()):
        token = raw.strip("'")
        if len(token) < 2 or token.isdigit() or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def explanation_flags(explanation):
    """
    Binary flags derived from NUFORC's Explanation column.
    Returns a dict of exp_* features indicating likely mundane explanations.
    exp_certain = 1 if the explanation has no trailing '?' (definitive call).
    txt_* keyword flags are intentionally excluded -- summary_clean passed to
    CatBoost via text_features already captures those patterns without
    double-counting.
    """
    e = str(explanation or "").lower()
    return {
        "exp_drone": 1 if "drone" in e else 0,
        "exp_rocket": 1 if "rocket" in e else 0,
        "exp_balloon": 1 if "balloon" in e else 0,
        "exp_aircraft": 1 if "aircraft" in e else 0,
        "exp_starlink": 1 if "starlink" in e else 0,
        "exp_lantern": 1 if "lantern" in e else 0,
        "exp_satellite": 1 if "satellite" in e else 0,
        "exp_certain": 1 if (e and not e.endswith("?")) else 0,
    }


# #############################################################################
# Core engineering function
# #############################################################################


def engineer_rows(rows, city_state_lookup, city_only_lookup):
    """
    Two-pass feature engineering over a list of row dicts.

    Pass 1: compute location counts and cache datetime/token fields per row.
    Pass 2: build the enriched output dict for each row using cached fields.

    Returns a list of enriched row dicts.
    """
    location_counts = Counter()

    # Pass 1 -- accumulate location counts and cache expensive computations
    for row in rows:
        occurred_dt = to_datetime_safe(row.get("Occurred"))
        loc_key = build_location_key(
            row.get("City"), row.get("State"), row.get("Country")
        )

        row["_occurred_dt"] = occurred_dt
        row["_loc_key"] = loc_key
        row["_tokens"] = tokenize_summary(row.get("Summary", ""))

        location_counts[loc_key] += 1

    # Pass 2 -- build enriched output rows
    enriched = []
    for row in rows:
        occurred_dt = row["_occurred_dt"]
        reported_date = to_date_safe(row.get("Reported"))
        tokens = row["_tokens"]
        loc_key = row["_loc_key"]
        country = str(row.get("Country") or "").strip()

        # Geocode using raw state value for best lookup match
        lat, lon, geo_method = geocode_city(
            row.get("City"),
            row.get("State"),  # raw, not validated
            country,
            city_state_lookup,
            city_only_lookup,
        )

        # Validate state -- flag invalid US/Canada codes as 'Unknown' to
        # prevent foreign region names from leaking into the State column
        state = validate_state(row.get("State"), country)

        out = {
            # Passthrough identity and raw columns
            "report_id": row.get("report_id", ""),
            "Link": row.get("Link", ""),
            "Occurred": occurred_dt.isoformat(sep=" ") if occurred_dt else "",
            "Reported": reported_date.isoformat() if reported_date else "",
            "City": row.get("City", ""),
            "State": state,
            "Country": country,
            "Shape": row.get("Shape", ""),
            "Summary": row.get("Summary", ""),
            "Explanation": row.get("Explanation", ""),
            "Media": row.get("Media", ""),
            # Temporal features
            "occurred_year": occurred_dt.year if occurred_dt else "",
            "occurred_month": occurred_dt.month if occurred_dt else "",
            "occurred_day": occurred_dt.day if occurred_dt else "",
            "occurred_hour": occurred_dt.hour if occurred_dt else "",
            "is_night": (
                1
                if occurred_dt and (occurred_dt.hour >= 20 or occurred_dt.hour <= 5)
                else 0
            ),
            "is_weekend": 1 if occurred_dt and occurred_dt.weekday() in (5, 6) else 0,
            "report_lag_days": (
                # Clamp to [0, 365] -- negatives indicate data entry errors;
                # values > 365 are likely stale retrospective reports and
                # would distort any model using this as a feature.
                min(max((reported_date - occurred_dt.date()).days, 0), 365)
                if occurred_dt and reported_date
                else ""
            ),
            # location_count_total: number of times this exact city|state|country
            # combination appears across the full dataset. Serves as a proxy for
            # reporting density by location — high values indicate population centers
            # or known UAP hotspots with frequent filings; low values indicate
            # geographically rare or isolated reports. Distinct from cluster_id /
            # in_cluster which measure spatial proximity within 30 km; a city can
            # have high location_count_total without belonging to any DBSCAN cluster
            # if its reports are temporally spread out or geocoded to different
            # coordinates. Computed in Pass 1 of engineer_rows before any filtering.
            "location_count_total": location_counts[loc_key],
            # Geocoding (geocode_method retained for logging; dropped before save)
            "latitude": lat,
            "longitude": lon,
            "geocode_method": geo_method,
            # Shape group (collapsed from 20+ NUFORC shapes into 7 categories)
            "shape_group": SHAPE_MAP.get(str(row.get("Shape", "")).strip(), "other"),
            # Text features -- summary_clean passed to CatBoost via text_features
            "summary_token_count": len(tokens),
            "summary_clean": " ".join(tokens),
            # Media flag
            "has_media": 1 if row.get("Media") else 0,
            # Days since the most recent major UAP disclosure event prior to
            # this sighting. Captures reporting surges driven by congressional
            # hearings and DoD report releases. NUFORC volume is known to
            # spike following high-profile events. A short lag suggests the
            # report may be media-influenced; a long lag suggests a more
            # isolated, organic observation. Empty string if sighting predates
            # all known events. See UAP_EVENTS constant above for full source
            # citations per event.
            "days_since_uap_event": (
                min(
                    (occurred_dt.date() - e.date()).days
                    for e in UAP_EVENTS
                    if e.date() <= occurred_dt.date()
                )
                if occurred_dt
                and any(e.date() <= occurred_dt.date() for e in UAP_EVENTS)
                else ""
            ),
        }

        # Explanation flags (individual category + certainty; exp_any excluded
        # as it is a deterministic combination of the individual flags)
        out.update(explanation_flags(row.get("Explanation", "")))

        enriched.append(out)

    return enriched


# #############################################################################
# I/O helpers
# #############################################################################


def write_json(payload, output_path):
    """Write a dict to a JSON file at output_path."""
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# #############################################################################
# Main
# #############################################################################


@app.command()
def main(
    input_parquet: str = "./data/raw/nuforc_data.parquet",
    output_parquet: str = "./data/processed/nuforc_engineered.parquet",
    output_metadata: str = "./data/processed/nuforc_feature_metadata.json",
):
    """
    Engineer features from the raw NUFORC parquet and write enriched outputs.
    Produces both a parquet and a CSV at the same path (different extensions).
    geocode_method is used for logging only and is dropped before saving.
    """
    logger.info(f"Running script: {os.path.basename(__file__)}")

    # Step 1 -- Load raw parquet produced by data_gen.py
    logger.info("STEP 1 -- Load parquet")
    df = pd.read_parquet(input_parquet)
    raw_rows = df.reset_index().to_dict(orient="records")
    logger.info(f"  Rows read : {len(raw_rows):,}")

    # Step 2 -- Build offline geocode lookup tables
    logger.info("STEP 2 -- Build geocode lookups")
    city_state_lookup, city_only_lookup = build_geocode_lookups()

    # Step 3 -- Run two-pass feature engineering
    logger.info("STEP 3 -- Engineer features")
    enriched = engineer_rows(raw_rows, city_state_lookup, city_only_lookup)

    # Log geocode coverage stats
    geo_counts = Counter(r["geocode_method"] for r in enriched)
    total = sum(geo_counts.values())
    geocoded = geo_counts.get("state_match", 0) + geo_counts.get("city_fallback", 0)
    logger.info(f"  Geocode breakdown : {dict(geo_counts)}")
    logger.info(
        f"  Geocode rate : {geocoded / total * 100:.1f}%" if total else "  No rows"
    )

    # Step 4 -- Write outputs
    logger.info("STEP 4 -- Write outputs")
    df_out = pd.DataFrame(enriched).set_index("report_id")

    # Cast lat/lon to numeric (empty strings from non-US rows become NaN)
    df_out["latitude"] = pd.to_numeric(df_out["latitude"], errors="coerce")
    df_out["longitude"] = pd.to_numeric(df_out["longitude"], errors="coerce")

    # Drop geocode_method -- used for logging only, not a modeling feature
    df_out = df_out.drop(columns=["geocode_method"], errors="ignore")

    df_out.to_parquet(output_parquet)
    output_csv = output_parquet.replace(".parquet", ".csv")
    # utf-8-sig adds a BOM so Excel opens non-ASCII characters correctly
    df_out.to_csv(output_csv, encoding="utf-8-sig")
    logger.info(f"  Enriched parquet saved : {output_parquet}  ({len(df_out):,} rows)")
    logger.info(f"  Enriched CSV saved     : {output_csv}")

    # Write metadata sidecar
    metadata = {
        "row_count": len(df_out),
        "output_columns": list(df_out.columns),
        "geocode_counts": dict(geo_counts),
    }
    write_json(metadata, Path(output_metadata))
    logger.info(f"  Metadata saved         : {output_metadata}")
    logger.info("nlp_feature_engineer_nuforc complete.")


if __name__ == "__main__":
    app()
