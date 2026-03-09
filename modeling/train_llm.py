################################################################################
# train_llm.py
# Zero-shot / few-shot LLM prediction of NUFORC "dramatic" flag from summary.
# Uses Groq API (free tier) with async concurrent requests for speed.
################################################################################

import os
import json
import re
import asyncio
from pathlib import Path

import httpx
import pandas as pd
import numpy as np
import typer
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)

load_dotenv()

app = typer.Typer()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ZERO_SHOT_TEMPLATE = """\
You are an expert analyst reviewing civilian UAP (unidentified aerial \
phenomenon) sighting reports submitted to NUFORC (National UFO Reporting Center).

Based solely on the witness description below, predict whether this report \
would be flagged as dramatic or unusual by a NUFORC editor.

Respond with ONLY a single number between 0 and 1 representing your confidence \
that this report is dramatic or unusual. No explanation, no text, just the number.

Witness description: {summary}"""

FEW_SHOT_HEADER = """\
You are an expert analyst reviewing civilian UAP (unidentified aerial \
phenomenon) sighting reports submitted to NUFORC (National UFO Reporting Center).

Below are examples of reports and whether they were flagged as dramatic (1) \
or not (0) by a NUFORC editor. Use these examples to calibrate your prediction.

{examples}
Now predict the following report. Respond with ONLY a single number between \
0 and 1 representing your confidence that this report is dramatic or unusual. \
No explanation, no text, just the number.

Witness description: {summary}"""

FEW_SHOT_EXAMPLE_TEMPLATE = "Report: {summary}\nDramatic: {label}\n"


def build_prompt(
    summary: str,
    prompt_type: str,
    few_shot_examples: list[dict] | None = None,
) -> str:
    """Build prompt string based on prompt_type."""
    if prompt_type == "few_shot" and few_shot_examples:
        examples_str = "\n".join(
            FEW_SHOT_EXAMPLE_TEMPLATE.format(summary=ex["summary"], label=ex["label"])
            for ex in few_shot_examples
        )
        return FEW_SHOT_HEADER.format(examples=examples_str, summary=summary)
    return ZERO_SHOT_TEMPLATE.format(summary=summary)


def sample_few_shot_examples(
    X: pd.DataFrame,
    y: pd.Series,
    n: int,
    text_col: str,
    random_state: int,
) -> list[dict]:
    """Sample balanced few-shot examples (equal pos/neg where possible)."""
    rng = np.random.default_rng(random_state)
    n_pos = n // 2
    n_neg = n - n_pos

    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    chosen_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    chosen_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)

    examples = []
    for idx in list(chosen_pos) + list(chosen_neg):
        examples.append(
            {"summary": str(X.loc[idx, text_col]), "label": int(y.loc[idx])}
        )
    rng.shuffle(examples)
    return examples


async def call_llm_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    prompt: str,
    model: str,
    max_retries: int = 5,
) -> float:
    """Async Groq API call with semaphore-controlled concurrency."""
    async with semaphore:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            try:
                response = await client.post(
                    GROQ_URL, json=payload, headers=headers, timeout=30
                )

                if response.status_code == 429:
                    wait = 2**attempt
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"].strip()
                print(f"DEBUG raw: {repr(raw)}")
                match = re.search(r"\d+\.?\d*", raw)
                if match:
                    prob = float(match.group())
                    return max(0.0, min(1.0, prob))
                return 0.5

            except (httpx.TimeoutException, httpx.RequestError) as e:
                print(f"DEBUG timeout/request error (attempt {attempt}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    return 0.5
            except Exception as e:
                print(f"DEBUG exception: {type(e).__name__}: {e}")
                return 0.5

    return 0.5


async def run_predictions(
    ids: list,
    prompts: list,
    cache: dict,
    api_key: str,
    model: str,
    max_workers: int,
    cache_path: Path,
) -> list:
    """Run all predictions concurrently with progress bar."""

    semaphore = asyncio.Semaphore(max_workers)
    probs = [None] * len(ids)
    new_calls = 0

    pending_indices = [i for i, rid in enumerate(ids) if str(rid) not in cache]
    cached_count = len(ids) - len(pending_indices)
    print(f"Cached: {cached_count:,}  |  Pending API calls: {len(pending_indices):,}\n")

    async with httpx.AsyncClient() as client:

        async def fetch_one(i: int):
            nonlocal new_calls
            rid = ids[i]
            prob = await call_llm_async(client, semaphore, api_key, prompts[i], model)
            cache[str(rid)] = prob
            new_calls += 1
            return i, prob

        tasks = [fetch_one(i) for i in pending_indices]

        for coro in async_tqdm.as_completed(
            tasks, total=len(tasks), desc="LLM predictions"
        ):
            i, prob = await coro
            probs[i] = prob

            if new_calls % 50 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)

    for i, rid in enumerate(ids):
        if probs[i] is None:
            probs[i] = cache[str(rid)]

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    print(f"\nNew API calls made: {new_calls:,}")
    return probs


@app.command()
def main(
    features_path: str = "./data/processed/X.parquet",
    labels_path: str = "./data/processed/y_dramatic.parquet",
    output_path: str = "./models/train/llm/llm_dramatic_preds.parquet",
    model: str = "llama3-8b-8192",
    sample_n: int = 0,
    max_workers: int = 10,
    text_col: str = "summary",
    random_state: int = 222,
    cache_path: str = "./models/train/llm/llm_cache.json",
    prompt_type: str = "zero_shot",
    few_shot_n: int = 5,
    splits_dir: str = "./models/train/splits",
):
    """
    Zero-shot or few-shot LLM prediction of NUFORC dramatic flag from summary.
    Uses canonical test split from splits_dir when available, else falls back
    to random sample_n. Uses Groq API with async concurrency.
    """

    print("\n" + "#" * 80)
    print("LLM Prediction: dramatic flag from summary text (Groq)")
    print(f"Model: {model}  |  Prompt: {prompt_type}  |  Max workers: {max_workers}")
    if prompt_type == "few_shot":
        print(f"Few-shot examples: {few_shot_n}")
    print("#" * 80 + "\n")

    ############################################################################
    # Check API key
    ############################################################################

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
        raise typer.Exit(1)

    ############################################################################
    # Load data
    ############################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path).squeeze()

    if text_col not in X.columns:
        print(f"ERROR: '{text_col}' not found in X. Columns: {X.columns.tolist()}")
        raise typer.Exit(1)

    X = X.loc[y.index]

    ############################################################################
    # Subset fallback for few-shot pool if no train split exists
    ############################################################################

    splits_path = Path(splits_dir)
    test_indices_file = splits_path / "test_indices.parquet"
    train_indices_file = splits_path / "train_indices.parquet"

    if train_indices_file.exists():
        train_idx = pd.read_parquet(train_indices_file)["index"]
        X_train_pool = X.loc[train_idx]
        y_train_pool = y.loc[train_idx]
    elif test_indices_file.exists():
        test_idx = pd.read_parquet(test_indices_file)["index"]
        X_train_pool = X.loc[test_idx]
        y_train_pool = y.loc[test_idx]
        print(
            "WARNING: No train split found — sampling few-shot examples from test split."
        )
    else:
        X_train_pool = X
        y_train_pool = y
        print(
            "WARNING: No splits found — sampling few-shot examples from full dataset."
        )

    ############################################################################
    # Build few-shot examples from train split (not test)
    ############################################################################

    few_shot_examples = None
    if prompt_type == "few_shot":
        few_shot_examples = sample_few_shot_examples(
            X_train_pool, y_train_pool, few_shot_n, text_col, random_state
        )
        print(
            f"Sampled {len(few_shot_examples)} few-shot examples from train split "
            f"({sum(e['label'] for e in few_shot_examples)} positive).\n"
        )

    ############################################################################
    # Load splits — build split dict for all three
    ############################################################################

    split_data = {}
    for split_name in ["train", "valid", "test"]:
        split_file = splits_path / f"{split_name}_indices.parquet"
        if split_file.exists():
            idx = pd.read_parquet(split_file)["index"]
            split_data[split_name] = (X.loc[idx], y.loc[idx])
        else:
            # fallback: only test available or no splits at all
            if split_name == "test":
                split_data[split_name] = (X_test, y_test)

    if not split_data:
        print("ERROR: No split index files found and no fallback available.")
        raise typer.Exit(1)

    ############################################################################
    # Load cache
    ############################################################################

    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache):,} cached predictions.")
    else:
        cache = {}

    ############################################################################
    # Run predictions and compute metrics per split
    ############################################################################

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for split_name, (X_split, y_split) in split_data.items():
        print(
            f"\n--- Split: {split_name.upper()} ({len(y_split):,} rows, "
            f"{y_split.mean()*100:.1f}% positive) ---"
        )

        ids = X_split.index.tolist()
        prompts = [
            build_prompt(
                summary=(
                    str(X_split.loc[rid, text_col])
                    if pd.notna(X_split.loc[rid, text_col])
                    else ""
                ),
                prompt_type=prompt_type,
                few_shot_examples=few_shot_examples,
            )
            for rid in ids
        ]

        probs = asyncio.run(
            run_predictions(
                ids, prompts, cache, api_key, model, max_workers, cache_file
            )
        )

        y_true = y_split.values
        y_prob = np.array(probs)
        y_pred = (y_prob >= 0.5).astype(int)

        ap = average_precision_score(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Precision/PPV",
                    "Average Precision",
                    "Sensitivity",
                    "AUC ROC",
                    "Brier Score",
                ],
                "Value": [prec, ap, rec, auc, brier],
            }
        )

        print("\n" + "*" * 80)
        print(f"Report Model Metrics: LLM ({model}) — {prompt_type} — {split_name}")
        print(metrics.to_string(index=False))
        print("*" * 80)

        split_results = pd.DataFrame(
            {"split": split_name, "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred},
            index=y_split.index,
        )
        all_results.append(split_results)

    ############################################################################
    # Save all predictions
    ############################################################################

    results = pd.concat(all_results)
    results.to_parquet(output_file)
    print(f"\nPredictions saved to: {output_file}")


if __name__ == "__main__":
    app()
