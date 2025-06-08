"""
AI-native product classifier using semantic understanding rather than manual mappings.

This approach leverages the LLM's built-in knowledge to understand product semantics
and match them to FDA categories without relying on pre-built alias dictionaries.
"""
from __future__ import annotations

import argparse
import functools
import json
import pathlib
from typing import Dict, List
import re

import dotenv
import openai

# Load env vars from .env if present – no error if file is missing.
dotenv.load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parents[1]
FDA_JSON = ROOT / "data" / "fda_categories.json"


@functools.lru_cache(maxsize=1)
def _load_fda_categories() -> Dict[str, List[str]]:
    """Load FDA categories and descriptions."""
    try:
        with FDA_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Try alternative path
        alt_path = pathlib.Path("data/fda_categories.json")
        if alt_path.exists():
            with alt_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}



def classify_llm_semantic(product_name: str, *, model: str = "gpt-4o", temperature: float = 0.0, debug: bool = False) -> str:
    """
    AI-native classification using semantic understanding.
    
    Instead of pre-filtering candidates, we let the AI use its semantic knowledge
    to understand the product and find the best category match.
    """
    if debug:
        print(f"\n🤖 AI-native classification: {product_name}")
    
    categories = _load_fda_categories()
    category_names = list(categories.keys())
    
    # Let the AI understand the product semantically and choose the best category
    system_prompt = """You are an expert food product classifier with deep understanding of food categories and product semantics.

Your task: Given a product name, determine which FDA food category it belongs to.

CRITICAL GUIDELINES:
1. ALWAYS read product descriptions carefully to distinguish target audience
2. Many categories have INFANT/BABY versions and GENERAL consumption versions - choose correctly:
   - "Juices, all varieties" = ONLY for babies/infants up to 12 months
   - "Juices, nectars, fruit drinks" = for general consumption (adults, children, etc.)
3. Look for size indicators (64oz, gallon, etc.) - large sizes are NEVER for infants
4. Look for terms like "conventional", "organic", "family size" - these indicate general consumption
5. If no age specification is given and it's a standard product, assume general consumption
6. Be precise about product types (grated cheese vs regular cheese, different milk types, etc.)

Process:
1. Understand what the product actually IS and WHO it's for
2. Consider product variants and target audience
3. Match to the most appropriate FDA category based on semantic meaning
4. If genuinely unsure, return 'MISC'

Respond with ONLY the exact category name or 'MISC'."""

    user_prompt = f"""Product: {product_name}

Available FDA Categories:
{json.dumps(category_names, indent=2)}

Category:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=256,
        messages=messages,
    )
    
    choice = resp.choices[0].message.content.strip()
    
    if debug:
        print(f"🎯 AI choice: '{choice}'")
    
    # Validate the response
    if choice.upper() == "MISC":
        return "MISC"
    
    if choice in category_names:
        return choice
    
    # If AI returned something close but not exact, find the best match
    from rapidfuzz import fuzz, process
    best_match, score, _ = process.extractOne(choice, category_names, scorer=fuzz.ratio)
    
    if score > 95:
        if debug:
            print(f"📝 Corrected '{choice}' → '{best_match}' (score: {score})")
        return best_match
    
    if debug:
        print(f"⚠️ AI returned unexpected category: '{choice}'")
    return "MISC"


def classify_llm_hybrid(product_name: str, *, model: str = "gpt-4o", temperature: float = 0.0, debug: bool = False) -> str:
    """
    Hybrid approach: Use AI to generate candidates, then AI to choose the best one.
    This reduces token usage while maintaining semantic understanding.
    """
    if debug:
        print(f"\n🔄 Hybrid classification: {product_name}")
    
    categories = _load_fda_categories()
    category_names = list(categories.keys())
    
    # Step 1: Ask AI to identify the most relevant categories (candidate generation)
    candidate_prompt = f"""Given this product name, identify the 3-5 most relevant FDA food categories:

Product: {product_name}

IMPORTANT CONSIDERATIONS:
- Distinguish between infant/baby products and general consumption products
- Large package sizes (64oz, gallon, etc.) indicate general consumption, NOT infant products
- Terms like "conventional", "organic", "family size" indicate general consumption
- "Juices, all varieties" is ONLY for baby/infant juices (up to 12 months)
- "Juices, nectars, fruit drinks" is for general consumption juices
- Consider product size, packaging, and target audience

Think about:
- What type of food/product this is
- WHO the target consumer is (infant vs general public)
- Similar products and their categories  
- Product variants (e.g., "Whole Milk" → milk categories)

Available categories:
{json.dumps(category_names, indent=2)}

Respond with a JSON array of the most relevant category names (3-5 max):"""

    client = openai.OpenAI()
    
    # Get candidates from AI
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=512,
        messages=[{"role": "user", "content": candidate_prompt}]
    )
    
    try:
        candidates_raw = resp.choices[0].message.content.strip()
        # Clean up the response to extract JSON
        if "```json" in candidates_raw:
            candidates_raw = candidates_raw.split("```json")[1].split("```")[0]
        elif "```" in candidates_raw:
            candidates_raw = candidates_raw.split("```")[1].split("```")[0]
        
        candidates = json.loads(candidates_raw)
        
        # Validate candidates exist in our category list
        valid_candidates = [c for c in candidates if c in category_names]
        
        if not valid_candidates:
            if debug:
                print("⚠️ No valid candidates from AI, falling back to semantic approach")
            return classify_llm_semantic(product_name, model=model, temperature=temperature, debug=debug)
        
        if debug:
            print(f"🎯 AI-generated candidates: {valid_candidates}")
        
        # Step 2: Ask AI to choose the best from candidates
        choice_prompt = f"""Choose the single best category for this product from the given candidates:

Product: {product_name}

Available candidates: {json.dumps(valid_candidates)}

CRITICAL DECISION FACTORS:
- Target audience: Is this for infants/babies (up to 12 months) or general consumption?
- Package size: Large sizes (64oz, gallon) are NEVER for infants
- Product descriptors: "conventional", "organic", "family size" = general consumption
- Default assumption: Unless clearly marked for babies/infants, assume general consumption

Instructions:
- Choose the most specific and appropriate category from the candidates above
- Distinguish carefully between infant/baby categories and general consumption categories
- Consider package size and target audience indicators
- Only return 'MISC' if the product really doesn't fit any of the candidates
- Be precise - if there's a specific match, choose it over general categories

Respond with only the exact category name:"""
        
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=256,
            messages=[{"role": "user", "content": choice_prompt}]
        )
        
        choice = resp.choices[0].message.content.strip()
        
        if debug:
            print(f"🎯 Final AI choice: '{choice}'")
        
        if choice.upper() == "MISC":
            return "MISC"
        
        if choice in valid_candidates:
            return choice
        
        # Fuzzy match against candidates
        from rapidfuzz import fuzz, process
        best_match, score, _ = process.extractOne(choice, valid_candidates, scorer=fuzz.ratio)
        
        if score > 90:
            if debug:
                print(f"📝 Corrected '{choice}' → '{best_match}' (score: {score})")
            return best_match
        
        return "MISC"
        
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        if debug:
            print(f"⚠️ Error parsing AI candidates: {e}, falling back to semantic approach")
        return classify_llm_semantic(product_name, model=model, temperature=temperature, debug=debug)


# For backwards compatibility, default to the hybrid approach
def classify_llm(product_name: str, *, model: str = "gpt-4o", temperature: float = 0.0, debug: bool = False) -> str:
    """Main classification function - uses hybrid AI approach by default."""
    return classify_llm_hybrid(product_name, model=model, temperature=temperature, debug=debug)


def _cli():
    parser = argparse.ArgumentParser(description="AI-native FDA classifier")
    parser.add_argument("product_name", help="Product name to classify")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--method", choices=["hybrid", "semantic"], default="hybrid", 
                       help="Classification method: hybrid (2-step AI) or semantic (1-step AI)")
    args = parser.parse_args()

    if args.method == "semantic":
        answer = classify_llm_semantic(args.product_name, model=args.model, debug=args.debug)
    else:
        answer = classify_llm_hybrid(args.product_name, model=args.model, debug=args.debug)
    
    print(answer)


if __name__ == "__main__":
    _cli()
