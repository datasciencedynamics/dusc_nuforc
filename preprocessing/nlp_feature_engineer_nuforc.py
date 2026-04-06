#!/usr/bin/env python3

import csv
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import typer

app = typer.Typer()

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")


EXCEL_EPOCH = datetime(1899, 12, 30)
XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TOKEN_RE = re.compile(r"[a-z0-9']+")

# NOTE: summary_clean is retained for CatBoost's text_features input.
# TF-IDF columns have been removed — CatBoost handles text internally
# when Summary (or summary_clean) is passed via text_features=[...].
APPEND_BASE_COLUMNS = [
    "index",
    "location_key",
    "location_count_total",
    "location_count_am",
    "location_count_pm",
    "occurred_year",
    "occurred_month",
    "occurred_day",
    "occurred_hour",
    "occurred_ampm",
    "summary_char_count",
    "summary_token_count",
    "summary_unique_token_count",
    "summary_clean",  # pass this column to CatBoost as text_features
]

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


def excel_serial_to_datetime(value):
    if value in (None, ""):
        return None
    try:
        return EXCEL_EPOCH + timedelta(days=float(value))
    except (TypeError, ValueError):
        return None


def excel_serial_to_date(value):
    dt = excel_serial_to_datetime(value)
    return dt.date() if dt else None


def col_to_index(cell_ref):
    letters = []
    for char in cell_ref:
        if char.isalpha():
            letters.append(char)
        else:
            break
    value = 0
    for char in letters:
        value = (value * 26) + (ord(char.upper()) - 64)
    return value - 1


def load_shared_strings(workbook_zip):
    shared_strings_path = "xl/sharedStrings.xml"
    if shared_strings_path not in workbook_zip.namelist():
        return []
    shared_strings_root = ET.fromstring(workbook_zip.read(shared_strings_path))
    values = []
    for item in shared_strings_root.findall("a:si", XML_NS):
        text = "".join(node.text or "" for node in item.iterfind(".//a:t", XML_NS))
        values.append(text)
    return values


def iter_sheet_rows(workbook_path):
    with zipfile.ZipFile(workbook_path) as workbook_zip:
        shared_strings = load_shared_strings(workbook_zip)
        sheet_root = ET.fromstring(workbook_zip.read("xl/worksheets/sheet1.xml"))
        rows = sheet_root.find("a:sheetData", XML_NS).findall("a:row", XML_NS)

        headers = None
        data_index = 0
        for row in rows:
            values = {}
            for cell in row.findall("a:c", XML_NS):
                value_node = cell.find("a:v", XML_NS)
                value = "" if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value != "":
                    value = shared_strings[int(value)]
                values[col_to_index(cell.attrib["r"])] = value

            ordered = [
                values.get(idx, "") for idx in range(max(values) + 1 if values else 0)
            ]
            if headers is None:
                headers = ordered
                continue

            record = {
                headers[idx]: ordered[idx] if idx < len(ordered) else ""
                for idx in range(len(headers))
            }
            record["index"] = data_index
            data_index += 1
            yield record


def normalize_location_part(value, fallback):
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    return re.sub(r"\s+", " ", text)


def build_location_key(record):
    city = normalize_location_part(record.get("City"), "unknown_city")
    state = normalize_location_part(record.get("State"), "unknown_state")
    country = normalize_location_part(record.get("Country"), "unknown_country")
    return f"{city}|{state}|{country}"


def tokenize_summary(summary):
    """
    Light tokenization used only to produce summary_clean for CatBoost's
    text_features input and to compute token count statistics.
    CatBoost handles its own internal tokenization and embedding;
    we are NOT producing TF-IDF features here.
    """
    tokens = []
    for raw_token in TOKEN_RE.findall((summary or "").lower()):
        token = raw_token.strip("'")
        if len(token) < 2:
            continue
        if token.isdigit():
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def engineer_rows(rows):
    location_counts = Counter()
    location_am_counts = Counter()
    location_pm_counts = Counter()

    # First pass: accumulate location counts
    for row in rows:
        occurred_dt = excel_serial_to_datetime(row.get("Occurred"))
        if occurred_dt is None:
            ampm = "unknown"
        elif occurred_dt.hour < 12:
            ampm = "am"
        else:
            ampm = "pm"

        location_key = build_location_key(row)
        row["_occurred_dt"] = occurred_dt
        row["_ampm"] = ampm
        row["_location_key"] = location_key

        location_counts[location_key] += 1
        if ampm == "am":
            location_am_counts[location_key] += 1
        elif ampm == "pm":
            location_pm_counts[location_key] += 1

        tokens = tokenize_summary(row.get("Summary", ""))
        row["_summary_tokens"] = tokens

    # Second pass: build engineered rows
    engineered = []
    for row in rows:
        occurred_dt = row["_occurred_dt"]
        reported_date = excel_serial_to_date(row.get("Reported"))
        tokens = row["_summary_tokens"]

        engineered_row = dict(row)

        # Datetime features
        engineered_row["Occurred"] = (
            occurred_dt.isoformat(sep=" ") if occurred_dt else ""
        )
        engineered_row["Reported"] = reported_date.isoformat() if reported_date else ""
        engineered_row["occurred_year"] = occurred_dt.year if occurred_dt else ""
        engineered_row["occurred_month"] = occurred_dt.month if occurred_dt else ""
        engineered_row["occurred_day"] = occurred_dt.day if occurred_dt else ""
        engineered_row["occurred_hour"] = occurred_dt.hour if occurred_dt else ""
        engineered_row["occurred_ampm"] = row["_ampm"]

        # Location features
        engineered_row["location_key"] = row["_location_key"]
        engineered_row["location_count_total"] = location_counts[row["_location_key"]]
        engineered_row["location_count_am"] = location_am_counts[row["_location_key"]]
        engineered_row["location_count_pm"] = location_pm_counts[row["_location_key"]]

        # Text stat features (signal features, not TF-IDF)
        engineered_row["summary_clean"] = " ".join(tokens)
        engineered_row["summary_token_count"] = len(tokens)
        engineered_row["summary_unique_token_count"] = len(set(tokens))
        engineered_row["summary_char_count"] = len((row.get("Summary") or "").strip())

        # Remove transient keys
        for transient_key in [
            "_occurred_dt",
            "_ampm",
            "_location_key",
            "_summary_tokens",
        ]:
            engineered_row.pop(transient_key, None)

        engineered.append(engineered_row)

    metadata = {
        "row_count": len(rows),
        "note": (
            "TF-IDF features removed. Pass 'summary_clean' or raw 'Summary' "
            "directly to CatBoost via text_features=[...]. CatBoost handles "
            "tokenization and text embedding internally."
        ),
    }
    return engineered, metadata


def select_output_rows(rows, output_mode):
    if output_mode == "full" or not rows:
        return rows
    slim_rows = []
    for row in rows:
        slim_rows.append({col: row.get(col, "") for col in APPEND_BASE_COLUMNS})
    return slim_rows


def write_csv(rows, output_path):
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload, output_path):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@app.command()
def main(
    input_workbook: str = "./data/raw/NUFORC_DATA.xlsx",
    output_csv: str = "./data/processed/nuforc_engineered_features.csv",
    output_metadata: str = "./data/processed/nuforc_feature_metadata.json",
    output_mode: str = "full",
):
    """
    Feature engineering for the NUFORC workbook.

    Produces structured, temporal, location, and lightweight text stat features.
    TF-IDF columns have been removed. Pass 'summary_clean' to CatBoost via
    text_features=[...] and let CatBoost handle tokenization internally.

    Args:
        input_workbook (str): Path to the source .xlsx workbook.
        output_csv (str): Destination for the enriched CSV output.
        output_metadata (str): Destination for run metadata.
        output_mode (str): 'full' or 'append' (append-ready column subset).
    """
    input_path = Path(input_workbook)
    output_csv_path = Path(output_csv)
    output_metadata_path = Path(output_metadata)

    raw_rows = list(iter_sheet_rows(input_path))
    engineered_rows, metadata = engineer_rows(raw_rows)
    output_rows = select_output_rows(engineered_rows, output_mode)

    metadata["output_mode"] = output_mode
    metadata["output_row_count"] = len(output_rows)
    if output_rows:
        metadata["output_columns"] = list(output_rows[0].keys())

    write_csv(output_rows, output_csv_path)
    write_json(metadata, output_metadata_path)
    print(f"Wrote {len(output_rows)} rows to {output_csv_path}")
    print(f"Wrote metadata to {output_metadata_path}")


if __name__ == "__main__":
    app()
