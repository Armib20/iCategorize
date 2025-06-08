"""
Reasoning system for the AI agent.

Handles intent analysis, explanation generation, and multi-step planning.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple, Any, Optional

import openai


class AgentReasoner:
    """
    Handles agent reasoning including intent analysis and explanation generation.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = openai.OpenAI()
    
    def analyze_intent(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze user intent and extract entities.
        
        Returns:
            (intent, entities) tuple
        """
        prompt = f"""Analyze this user input and determine the intent and extract entities.

User input: "{user_input}"

Possible intents:
- classify_single: User wants to classify one product
- classify_batch: User wants to classify multiple products  
- feedback: User is providing feedback/correction
- explain: User wants explanation of a classification
- export: User wants to export results
- help: User needs help or general conversation

Extract entities like:
- product_name: Single product name
- product_names: List of product names
- correction: {{product_name, wrong_category, correct_category}}

Respond with JSON: {{"intent": "...", "entities": {{...}}}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result.get("intent", "help"), result.get("entities", {})
            
        except (json.JSONDecodeError, Exception) as e:
            # If OpenAI fails for any reason, fallback to pattern matching
            # This ensures the agent always works even without OpenAI access
            return self._fallback_intent_analysis(user_input)
    
    def _fallback_intent_analysis(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Simple pattern-based fallback for intent analysis."""
        user_lower = user_input.lower().strip()
        
        # Check for explicit classification requests
        if any(word in user_lower for word in ["classify", "categorize", "what category"]):
            # Look for product names in quotes or after "classify"
            product_match = re.search(r'["\']([^"\']+)["\']', user_input)
            if product_match:
                return "classify_single", {"product_name": product_match.group(1)}
            
            # Look for product after "classify" keyword
            classify_match = re.search(r'classify\s+(.+)', user_lower)
            if classify_match:
                product_name = classify_match.group(1).strip()
                if product_name and product_name not in ["these", "this", "that"]:
                    return "classify_single", {"product_name": product_name}
            
            # Look for lists
            if any(word in user_lower for word in ["list", "these", "batch"]):
                return "classify_batch", {"product_names": self._extract_product_list(user_input)}
            
            return "classify_single", {"product_name": ""}
        
        # Check for feedback
        if any(word in user_lower for word in ["wrong", "incorrect", "should be", "actually"]):
            return "feedback", {}
        
        # Check for explanation requests
        if any(word in user_lower for word in ["why", "explain", "how", "reasoning"]):
            return "explain", {}
        
        # Check for simple product names (if it looks like a food product)
        if len(user_input.split()) <= 4 and not any(word in user_lower for word in ["help", "stats", "export", "show", "tell"]):
            # Common food product indicators
            food_indicators = [
                "milk", "bread", "cheese", "butter", "yogurt", "juice", "water", "soda",
                "cereal", "pasta", "rice", "beans", "nuts", "fruit", "apple", "banana",
                "chicken", "beef", "fish", "pork", "egg", "flour", "sugar", "salt",
                "oil", "sauce", "honey", "jam", "tea", "coffee", "wine", "beer",
                "chocolate", "candy", "cookie", "cake", "ice", "cream", "pickle",
                "organic", "fresh", "frozen", "canned", "whole", "skim", "raw"
            ]
            
            # If input contains food-related words or looks like a product name
            if (any(indicator in user_lower for indicator in food_indicators) or
                re.match(r'^[a-zA-Z\s\d\-&\'\.\,]+$', user_input)):
                return "classify_single", {"product_name": user_input.strip()}
        
        return "help", {}
    
    def _extract_product_list(self, text: str) -> List[str]:
        """Extract a list of products from text."""
        # Simple extraction - look for comma-separated items
        patterns = [
            r':\s*([^.!?]+)',  # After colon
            r'these:\s*([^.!?]+)',  # After "these:"
            r'\[(.*?)\]',  # In brackets
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                items = [item.strip().strip('"\'') for item in match.group(1).split(',')]
                return [item for item in items if len(item) > 2]
        
        return []
    
    def explain_classification(
        self, 
        product_name: str, 
        category: str
    ) -> Tuple[str, List[str], float]:
        """
        Generate explanation for why a product was classified in a category.
        
        Returns:
            (reasoning, alternatives, confidence)
        """
        prompt = f"""Explain why the product "{product_name}" was classified as "{category}".

Provide:
1. Clear reasoning for this classification
2. 2-3 alternative categories that could apply
3. Confidence score (0.0-1.0)

Be concise but informative. Focus on the key characteristics that led to this classification.

Respond with JSON:
{{
    "reasoning": "...",
    "alternatives": ["category1", "category2"],
    "confidence": 0.85
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return (
                result.get("reasoning", "Classification based on product characteristics."),
                result.get("alternatives", []),
                result.get("confidence", 0.8)
            )
            
        except (json.JSONDecodeError, Exception):
            # Fallback explanation
            return (
                f"This product was classified as '{category}' based on its characteristics and typical usage patterns.",
                [],
                0.8
            )
    
    def detailed_explanation(self, result) -> str:
        """Generate a detailed explanation of a classification result."""
        explanation = f"""## Detailed Classification Analysis

**Product:** {result.product_name}  
**Category:** {result.category}  
**Confidence:** {result.confidence:.1%}  
**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

### Reasoning
{result.reasoning}

### Process
I analyzed the product name "{result.product_name}" and considered various factors:
- Product type and characteristics
- Typical usage patterns
- FDA category definitions
- Similar products and their classifications

"""
        
        if result.alternatives:
            explanation += f"""### Alternative Categories Considered
{', '.join(result.alternatives)}

These were considered but didn't match as closely as the selected category.
"""
        
        explanation += """
### Confidence Assessment
This confidence score reflects how certain I am about this classification based on the product name and available information."""
        
        return explanation
    
    def generate_suggestions(
        self, 
        context: str, 
        recent_classifications: List
    ) -> List[str]:
        """Generate helpful suggestions for the user."""
        suggestions = [
            "Ask me to classify another product",
            "Tell me if any classifications seem wrong",
            "Request detailed explanations"
        ]
        
        # Context-aware suggestions
        if len(recent_classifications) > 5:
            suggestions.append("Ask me to export your results")
        
        if context and "batch" in context.lower():
            suggestions.append("Process another batch of products")
        
        return suggestions
    
    def plan_response(
        self, 
        intent: str, 
        entities: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Plan the agent's response based on intent and context.
        
        Returns response plan with actions to take.
        """
        plan = {
            "primary_action": intent,
            "requires_classification": intent in ["classify_single", "classify_batch"],
            "requires_explanation": intent in ["explain", "classify_single"],
            "requires_learning": intent == "feedback",
            "entities": entities,
            "fallback_needed": False
        }
        
        # Validate entities for the intent
        if intent == "classify_single" and not entities.get("product_name"):
            plan["fallback_needed"] = True
            plan["fallback_message"] = "Please provide a product name to classify."
        
        elif intent == "classify_batch" and not entities.get("product_names"):
            plan["fallback_needed"] = True  
            plan["fallback_message"] = "Please provide a list of products to classify."
        
        return plan 