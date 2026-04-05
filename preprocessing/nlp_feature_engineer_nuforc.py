#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


EXCEL_EPOCH = datetime(1899, 12, 30)
XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TOKEN_RE = re.compile(r"[a-z0-9']+")
MIN_DOC_FREQUENCY = 5
MAX_DOC_FREQUENCY_RATIO = 0.50
DEFAULT_VOCAB_SIZE = 75
APPEND_BASE_COLUMNS = [
    "index",
    "location_key",
    "location_count_total",
    "location_count_am",
    "location_count_pm",
    "summary_clean",
    "summary_top_keywords",
]

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
    "see", "saw", "seen", "look", "looked", "looking",
    "thing", "things", "something",
    "went", "go", "gone", "got", "come", "came",
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

            ordered = [values.get(idx, "") for idx in range(max(values) + 1 if values else 0)]
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


def choose_vocabulary(document_frequency, corpus_frequency, document_count, vocab_size):
    candidates = []
    for token, doc_freq in document_frequency.items():
        if doc_freq < MIN_DOC_FREQUENCY:
            continue
        if doc_freq / document_count > MAX_DOC_FREQUENCY_RATIO:
            continue
        score = corpus_frequency[token] * math.log((1 + document_count) / (1 + doc_freq))
        candidates.append((score, token))
    candidates.sort(reverse=True)
    return [token for _, token in candidates[:vocab_size]]


def engineer_rows(rows, vocab_size):
    location_counts = Counter()
    location_am_counts = Counter()
    location_pm_counts = Counter()
    document_frequency = Counter()
    corpus_frequency = Counter()

    for row in rows:
        occurred_dt = excel_serial_to_datetime(row.get("Occurred"))
        reported_date = excel_serial_to_date(row.get("Reported"))

        if occurred_dt is None:
            ampm = "unknown"
        elif occurred_dt.hour < 12:
            ampm = "am"
        else:
            ampm = "pm"

        location_key = build_location_key(row)
        row["_occurred_dt"] = occurred_dt
        row["_reported_date"] = reported_date
        row["_ampm"] = ampm
        row["_location_key"] = location_key

        location_counts[location_key] += 1
        if ampm == "am":
            location_am_counts[location_key] += 1
        elif ampm == "pm":
            location_pm_counts[location_key] += 1

        tokens = tokenize_summary(row.get("Summary", ""))
        row["_summary_tokens"] = tokens
        term_counts = Counter(tokens)
        row["_term_counts"] = term_counts
        for token in term_counts:
            document_frequency[token] += 1
        corpus_frequency.update(tokens)

    vocabulary = choose_vocabulary(
        document_frequency, corpus_frequency, len(rows), vocab_size
    )
    idf = {
        token: math.log((1 + len(rows)) / (1 + document_frequency[token])) + 1.0
        for token in vocabulary
    }

    engineered = []
    for row in rows:
        occurred_dt = row["_occurred_dt"]
        reported_date = row["_reported_date"]
        term_counts = row["_term_counts"]
        total_tokens = len(row["_summary_tokens"])

        tfidf_scores = {}
        for token in vocabulary:
            if total_tokens == 0 or term_counts[token] == 0:
                tfidf_scores[token] = 0.0
                continue
            tf = term_counts[token] / total_tokens
            tfidf_scores[token] = round(tf * idf[token], 6)

        top_keywords = [
            token
            for token, score in sorted(
                tfidf_scores.items(), key=lambda item: item[1], reverse=True
            )
            if score > 0
        ][:5]

        engineered_row = dict(row)

        engineered_row["Occurred_raw"] = row.get("Occurred", "")
        engineered_row["Reported_raw"] = row.get("Reported", "")
        engineered_row["Occurred"] = occurred_dt.isoformat(sep=" ") if occurred_dt else ""
        engineered_row["Reported"] = reported_date.isoformat() if reported_date else ""

        engineered_row["occurred_iso"] = occurred_dt.isoformat(sep=" ") if occurred_dt else ""
        engineered_row["occurred_year"] = occurred_dt.year if occurred_dt else ""
        engineered_row["occurred_month"] = occurred_dt.month if occurred_dt else ""
        engineered_row["occurred_day"] = occurred_dt.day if occurred_dt else ""
        engineered_row["occurred_hour"] = occurred_dt.hour if occurred_dt else ""
        engineered_row["occurred_ampm"] = row["_ampm"]

        engineered_row["location_key"] = row["_location_key"]
        engineered_row["location_count_total"] = location_counts[row["_location_key"]]
        engineered_row["location_count_am"] = location_am_counts[row["_location_key"]]
        engineered_row["location_count_pm"] = location_pm_counts[row["_location_key"]]

        engineered_row["summary_clean"] = " ".join(row["_summary_tokens"])
        engineered_row["summary_token_count"] = total_tokens
        engineered_row["summary_unique_token_count"] = len(set(row["_summary_tokens"]))
        engineered_row["summary_char_count"] = len((row.get("Summary") or "").strip())
        engineered_row["summary_top_keywords"] = " ".join(top_keywords)

        for token in vocabulary:
            engineered_row[f"tfidf__{token}"] = tfidf_scores[token]

        for transient_key in [
            "_occurred_dt",
            "_reported_date",
            "_ampm",
            "_location_key",
            "_summary_tokens",
            "_term_counts",
        ]:
            engineered_row.pop(transient_key, None)

        engineered.append(engineered_row)

    metadata = {
        "row_count": len(rows),
        "vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
        "top_corpus_terms": corpus_frequency.most_common(25),
    }
    return engineered, metadata


def select_output_rows(rows, output_mode):
    if output_mode == "full" or not rows:
        return rows

    tfidf_columns = [column for column in rows[0] if column.startswith("tfidf__")]
    selected_columns = APPEND_BASE_COLUMNS + tfidf_columns

    slim_rows = []
    for row in rows:
        slim_rows.append({column: row.get(column, "") for column in selected_columns})
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


def parse_args():
    parser = argparse.ArgumentParser(description="Feature engineering for the NUFORC workbook.")
    parser.add_argument("input_workbook", type=Path, help="Path to the source .xlsx workbook")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("nuforc_engineered_features.csv"),
        help="Destination for the enriched CSV output",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("nuforc_feature_metadata.json"),
        help="Destination for vocabulary and run metadata",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="How many TF-IDF summary terms to keep as explicit numeric features",
    )
    parser.add_argument(
        "--output-mode",
        choices=["full", "append"],
        default="full",
        help="Write the full engineered dataset or only the append-ready subset of columns.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_rows = list(iter_sheet_rows(args.input_workbook))
    engineered_rows, metadata = engineer_rows(raw_rows, args.vocab_size)
    output_rows = select_output_rows(engineered_rows, args.output_mode)

    metadata["output_mode"] = args.output_mode
    metadata["output_row_count"] = len(output_rows)
    if output_rows:
        metadata["output_columns"] = list(output_rows[0].keys())

    write_csv(output_rows, args.output_csv)
    write_json(metadata, args.output_metadata)
    print(f"Wrote {len(output_rows)} rows to {args.output_csv}")
    print(f"Wrote metadata to {args.output_metadata}")


if __name__ == "__main__":
    main()
