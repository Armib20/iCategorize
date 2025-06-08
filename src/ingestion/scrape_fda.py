"""
Scrape FDA Product Categories & Products page and persist as JSON/CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

FDA_URL = "https://www.fda.gov/product-categories-and-products#_top"


def fetch_html(url: str = FDA_URL) -> str:
    """Download the FDA page and return raw HTML."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_categories(html: str) -> Dict[str, List[str]]:
    """Parse top‑level <h3> headings and the following <ul><li> lists.

    Returns a mapping {category_name: [product1, product2, ...]}.
    """
    soup = BeautifulSoup(html, "html.parser")
    categories: Dict[str, List[str]] = {}

    for h3 in soup.find_all("h3"):
        cat_name = h3.get_text(strip=True)
        ul = h3.find_next_sibling("ul")
        if not ul:
            # Occasionally the FDA page has an <h3> without a list; skip.
            continue
        items = [li.get_text(strip=True) for li in ul.find_all("li")]
        categories[cat_name] = items

    return categories


def save_json(data: Dict[str, List[str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(data: Dict[str, List[str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "product"])
        for category, products in data.items():
            for product in products:
                writer.writerow([category, product])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape FDA product categories and products")
    parser.add_argument("--out", default="data/interim/fda_categories.json", help="Path to output JSON file")
    parser.add_argument("--csv", default="data/interim/fda_categories.csv", help="Optional CSV companion file")
    args = parser.parse_args()

    html = fetch_html()
    cats = parse_categories(html)

    save_json(cats, pathlib.Path(args.out))
    save_csv(cats, pathlib.Path(args.csv))

    print(f"✅ Saved {len(cats)} categories → {args.out} (JSON) and {args.csv} (CSV)")


if __name__ == "__main__":
    main()
