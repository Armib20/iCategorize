"""
Heuristic helpers: text cleanup + fuzzy string matching baseline.

Usage example (run from repo root):

```python
from heuristics import classify_heuristic
print(classify_heuristic("Organic Granny Smith Apples 88ct"))
``` 
Returns `(best_category, confidence_score)` where confidence is the RapidFuzz
ratio (0‑100).

This module intentionally keeps zero external state so it can be imported by
both training notebooks and production code.
"""

from __future__ import annotations

import json
import pathlib
import re
from functools import lru_cache
from typing import Dict, List, Tuple, Set

from rapidfuzz import fuzz, process

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root two levels up
FDA_JSON = ROOT / "data" / "interim" / "fda_categories.json"

# ---------------------------------------------------------------------------
#  Normalization helpers
# ---------------------------------------------------------------------------

_re_units = re.compile(
    r"\b(lbs?|pounds?|kg|kgs|g|grams?|oz|ounces?|ct|count|box|case|pkgs?|pkg|pack|bag|ea|each|#|num)\b",
    flags=re.I,
)
_re_non_alphanum = re.compile(r"[^a-z0-9\s]")
_re_multispace = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lower-case, drop punctuation, and strip common size keywords."""
    s = s.lower()
    s = _re_non_alphanum.sub(" ", s)
    s = _re_units.sub(" ", s)
    s = _re_multispace.sub(" ", s).strip()
    return s

# ---------------------------------------------------------------------------
#  FDA vocabulary utilities
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_fda() -> Dict[str, List[str]]:
    if not FDA_JSON.exists():
        raise FileNotFoundError(
            f"{FDA_JSON} not found. Run `python -m ingestion.scrape_fda` first."
        )
    with FDA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _alias_map() -> Dict[str, str]:
    """Return {alias_normalized -> canonical_category}"""
    mapping = _load_fda()
    aliases: Dict[str, str] = {}
    
    for cat, descs in mapping.items():
        # Include the category name itself and extract key terms from it
        aliases[normalize(cat)] = cat
        
        # Extract key terms from category names (words before first comma/parenthesis)
        cat_main = re.split(r'[,(\[]', cat)[0].strip()
        for word in cat_main.split():
            word_norm = normalize(word)
            if len(word_norm) > 2:
                aliases[word_norm] = cat
        
        # Process descriptions more carefully
        for desc in descs:
            # Split by common delimiters and extract meaningful tokens
            parts = re.split(r'[;,\(\)]', desc)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                # Extract individual words and short phrases
                words = part.split()
                for i, word in enumerate(words):
                    word_norm = normalize(word)
                    if len(word_norm) > 2 and word_norm not in {'and', 'the', 'for', 'with', 'all', 'are', 'any', 'may'}:
                        aliases[word_norm] = cat
                
                # Also add 2-word phrases for compound terms
                for i in range(len(words) - 1):
                    phrase = f"{words[i]} {words[i+1]}"
                    phrase_norm = normalize(phrase)
                    if len(phrase_norm) > 4:
                        aliases[phrase_norm] = cat
    
    # Manual mappings for common terms that might be missed
    manual_mappings = {
        'milk': 'Milk, milk-based drinks, e.g., instant breakfast, meal replacement, cocoa',
        'butter': 'Butter, margarine, oil, shortening',
        'nuts': 'Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole',
        'pecans': 'Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole',
        'almonds': 'Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole',
        'walnuts': 'Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole',
        'popcorn': 'All varieties, chips, pretzels, popcorns, extruded snacks, fruit-based snacks (e.g., fruit chips), grain-based snack mixes',
        'chips': 'All varieties, chips, pretzels, popcorns, extruded snacks, fruit-based snacks (e.g., fruit chips), grain-based snack mixes',
        'pickles': 'Pickles, all types',
        'juice': 'Juices, nectars, fruit drinks',
        'orange juice': 'Juices, nectars, fruit drinks',
        'fruit juice': 'Juices, nectars, fruit drinks',
        'bread': 'Breads {excluding sweet quick type}, rolls',
        'yogurt': 'Yogurt',
        'cheese': 'Cheese, all others except those listed as separate categories--including cream cheese and cheese spread',
        'sauce': 'Major main entree sauces, e.g., spaghetti sauce',
        'bbq sauce': 'Barbecue sauce, hollandaise sauce, tartar sauce, other sauces for dipping (e.g., mustard sauce, sweet and sour sauce), all dips (e.g., bean dips, dairy-based dips, salsa)',
        'barbecue sauce': 'Barbecue sauce, hollandaise sauce, tartar sauce, other sauces for dipping (e.g., mustard sauce, sweet and sour sauce), all dips (e.g., bean dips, dairy-based dips, salsa)',
        'pasta': 'Pastes, plain',
        'pasta salad': 'Pasta or potato salad',
        'potato salad': 'Pasta or potato salad',
        'salad': 'All other salads, e.g., egg, fish, shellfish, bean, fruit, or vegetable salads',
        'ice cream': 'ice cream, ice milk, frozen yogurt, sherbet: all types, bulk and novelties e.g., bars, sandwiches, cones)',
        'frozen yogurt': 'ice cream, ice milk, frozen yogurt, sherbet: all types, bulk and novelties e.g., bars, sandwiches, cones)',
    }
    
    for term, category in manual_mappings.items():
        aliases[normalize(term)] = category
    
    return aliases


# ---------------------------------------------------------------------------
#  Enhanced candidate generation
# ---------------------------------------------------------------------------

def get_candidates(product_name: str, *, min_candidates: int = 1, max_candidates: int = 5) -> Set[str]:
    """Find relevant FDA categories using multiple strategies.
    
    This function combines several approaches:
    1. Direct heuristic classification
    2. Word-level fuzzy matching against aliases
    3. Product type detection against category names
    4. Fallback fuzzy matching against all categories
    """
    candidates: Set[str] = set()
    normalized_name = normalize(product_name)
    words = normalized_name.split()
    
    # Strategy 1: Direct heuristic classification
    best_heuristic, score = classify_heuristic(product_name)
    if best_heuristic != "MISC":
        candidates.add(best_heuristic)
    
    # Strategy 2: Word-level matching against aliases
    alias_map = _alias_map()
    alias_keys = list(alias_map.keys())
    
    # Try individual words
    for word in words:
        if len(word) > 2:  # Skip very short words
            matches = process.extract(word, alias_keys, limit=2, scorer=fuzz.ratio)
            for match, match_score, _ in matches:
                if match_score > 85:
                    candidates.add(alias_map[match])
    
    # Try full normalized name against aliases
    full_matches = process.extract(normalized_name, alias_keys, limit=3, scorer=fuzz.token_set_ratio)
    for match, match_score, _ in full_matches:
        if match_score > 75:
            candidates.add(alias_map[match])
    
    # Strategy 3: Product type detection
    categories = list(_load_fda().keys())
    common_product_types = {
        "juice", "milk", "bread", "butter", "yogurt", "ice cream", "cream",
        "sauce", "paste", "pickles", "salad", "nuts", "cheese", "eggs",
        "fruits", "vegetables", "meat", "fish", "chicken", "beef", "pork",
        "pasta", "cereal", "cake", "cookies", "candy", "chocolate", "soup"
    }
    
    for word in words:
        if word in common_product_types:
            matches = process.extract(word, categories, limit=2, scorer=fuzz.token_set_ratio)
            for match, match_score, _ in matches:
                if match_score > 70:
                    candidates.add(match)
    
    # Strategy 4: Fallback - if we don't have enough candidates, broaden the search
    if len(candidates) < min_candidates:
        # Try fuzzy matching the full product name against all category names
        broad_matches = process.extract(normalized_name, categories, limit=3, scorer=fuzz.token_set_ratio)
        for match, match_score, _ in broad_matches:
            if match_score > 60:
                candidates.add(match)
    
    # Limit the number of candidates to avoid overwhelming the LLM
    if len(candidates) > max_candidates:
        # Keep the best ones based on fuzzy score against the original product name
        candidate_scores = []
        for candidate in candidates:
            score = fuzz.token_set_ratio(normalized_name, normalize(candidate))
            candidate_scores.append((candidate, score))
        
        # Sort by score and keep top max_candidates
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = {candidate for candidate, _ in candidate_scores[:max_candidates]}
    
    return candidates


# ---------------------------------------------------------------------------
#  Fuzzy matching baseline
# ---------------------------------------------------------------------------

def classify_heuristic(raw_name: str, *, threshold: int = 90) -> Tuple[str, int]:
    """Return (best_category, score).

    We first look for an *exact* normalized alias match (fast). If that fails we
    do a fuzzy match across *category names only*. Empirically this is ~20×
    faster than matching against every alias while still covering most typos.
    """
    name = normalize(raw_name)

    # 1️⃣ exact alias → category shortcut
    alias_map = _alias_map()
    if name in alias_map:
        return alias_map[name], 100

    # 2️⃣ fuzzy against canonical category names
    cat_names = list(_load_fda().keys())
    best, score, _ = process.extractOne(
        name,
        cat_names,
        scorer=fuzz.token_set_ratio,
    )
    if score >= threshold:
        return best, score

    return "MISC", score
