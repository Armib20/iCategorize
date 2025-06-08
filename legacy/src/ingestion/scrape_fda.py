"""
Scrape FDA Product Categories & Products page and persist as JSON/CSV.

Key features
------------
* Browser‑like headers + retries to dodge FDA bot wall.
* Local disk cache after first successful pull (override with `--force-refresh`).
* **Table‑aware parser**: the FDA page lists categories inside HTML tables, not bullet lists. We now scan
  every `<table>` that has exactly two `<td>` cells per `<tr>`:
    • **Column 1** → product category name.
    • **Column 2** → the description / example products.
* If a category repeats later in the doc we merge the descriptions.
* Validation step — raises if < 10 categories parsed (a good sanity floor).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FDA_URL = "https://www.fda.gov/product-categories-and-products#_top"
CACHE_PATH = pathlib.Path("data/interim/fda_page.html")

# ---------------------------------------------------------------------------
#  HTTP helpers
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fda.gov/",
    "DNT": "1",
}


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[403, 404, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


# ---------------------------------------------------------------------------
#  Download / caching
# ---------------------------------------------------------------------------

def fetch_html(url: str = FDA_URL, *, force: bool = False) -> str:
    if not force and CACHE_PATH.exists():
        return CACHE_PATH.read_text(encoding="utf-8")

    session = build_session()
    for attempt in range(5):
        resp = session.get(url, timeout=30)
        ok = resp.status_code == 200 and "abuse-detection-apology" not in resp.text
        if ok:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(resp.text, encoding="utf-8")
            return resp.text
        wait = 2 ** attempt
        print(f"⚠️  Attempt {attempt+1}: status={resp.status_code}. Retrying in {wait}s…", file=sys.stderr)
        time.sleep(wait)

    if CACHE_PATH.exists():
        print("⚠️  Using cached FDA HTML due to repeated failures.", file=sys.stderr)
        return CACHE_PATH.read_text(encoding="utf-8")

    raise RuntimeError("Unable to download FDA page — blocked by bot detector.")


# ---------------------------------------------------------------------------
#  Parsing logic (table‑aware)
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Collapse whitespace and strip footnote markers."""
    text = re.sub(r"\s+", " ", text).strip()
    # Drop inline citations like "(e.g., …)"? keep for now; just remove footnote numbers.
    return re.sub(r"\(\s*\d+\s*\)", "", text)


def parse_categories(html: str) -> Dict[str, List[str]]:
    """Return mapping {category: [description, …]}. Merges duplicate categories."""
    soup = BeautifulSoup(html, "html.parser")
    mapping: Dict[str, List[str]] = defaultdict(list)

    # FDA tables are class="table table-bordered" but we fall back to all tables.
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 2:
                continue  # skip header rows or malformed rows
            cat = clean(cells[0].get_text(" ", strip=True))
            desc = clean(cells[1].get_text(" ", strip=True))

            # Skip empty or obvious non‑data rows
            if not cat or cat.lower().startswith("attachment"):
                continue
            mapping[cat].append(desc)

    return mapping


# ---------------------------------------------------------------------------
#  Writers
# ---------------------------------------------------------------------------

def save_json(data: Dict[str, List[str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(data: Dict[str, List[str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "description"])
        for category, descs in data.items():
            for desc in descs:
                writer.writerow([category, desc])


# ---------------------------------------------------------------------------
#  CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape FDA product categories and products")
    parser.add_argument("--out", default="data/interim/fda_categories.json", help="Output JSON")
    parser.add_argument("--csv", default="data/interim/fda_categories.csv", help="Output CSV")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and re‑scrape")
    args = parser.parse_args()

    html = fetch_html(force=args.force_refresh)
    cats = parse_categories(html)

    if len(cats) < 10:
        raise RuntimeError("Parsed fewer than 10 product categories — scraping likely failed. Try --force-refresh or inspect cached HTML in data/interim/fda_page.html.")

    save_json(cats, pathlib.Path(args.out))
    save_csv(cats, pathlib.Path(args.csv))

    print(f"✅ Saved {len(cats)} categories → {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
