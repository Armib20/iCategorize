"""
Evaluate LLM classifier accuracy on a labeled CSV.

Usage (from repo root, env must have OPENAI_API_KEY):

    python -m evaluation.eval_llm \
        --csv data/samples/test_products.csv \
        --model gpt-4o-mini

Outputs overall accuracy and a simple breakdown of mistakes.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import Counter

from ..llm.classifier import classify_llm


def run_eval(csv_path: pathlib.Path, model: str, debug: bool = False):
    total = 0
    correct = 0
    errors: Counter[str] = Counter()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            name = row["product_name"].strip()
            truth = row["ground_truth"].strip()
            pred = classify_llm(name, model=model, debug=debug)
            if pred == truth:
                correct += 1
                if debug:
                    print(f"✅ CORRECT: {name}")
            else:
                errors[f"{truth} → {pred}"] += 1
                print(f"❌ {name}: {pred} (expected {truth})")

    acc = correct / total * 100
    print("\n========================")
    print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    if errors:
        print("\nTop mistakes:")
        for pair, cnt in errors.most_common(10):
            print(f"  {pair}  ×{cnt}")


def _cli():
    parser = argparse.ArgumentParser(description="Evaluate LLM FDA classifier")
    parser.add_argument("--csv", default="data/samples/test_products.csv", help="CSV with product_name,ground_truth")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    run_eval(pathlib.Path(args.csv), args.model, args.debug)


if __name__ == "__main__":
    _cli()
