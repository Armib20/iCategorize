"""
Simplified AI Agent for Product Classification

This is a streamlined version focusing on core FDA classification functionality.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pathlib
import os
import openai
from .classifier import classify_llm_hybrid, classify_llm_semantic, classify_llm

# Try multiple possible paths for the FDA categories file
def _get_fda_json_path():
    """Get the path to FDA categories JSON with fallbacks for different environments."""
    
    # Try the most likely paths in order
    possible_paths = [
        # Development: relative to repo root
        pathlib.Path("data/fda_categories.json"),
        # Streamlit Cloud: absolute deployment path
        pathlib.Path("/mount/src/icategorize/data/fda_categories.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # If none found, return the first path for error reporting
    return possible_paths[0]

FDA_JSON = _get_fda_json_path()


@dataclass
class ClassificationResult:
    """Result of a product classification."""
    product_name: str
    category: str
    confidence: float
    reasoning: str
    alternatives: List[str]
    timestamp: datetime
    session_id: str


@dataclass
class AgentResponse:
    """Complete response from the agent."""
    message: str
    results: List[ClassificationResult]
    suggestions: List[str]
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


class SimplifiedProductClassificationAgent:
    """
    Simplified AI Agent for intelligent product classification.
    
    This agent can:
    - Classify single products or batches
    - Explain its reasoning
    - Handle ambiguous cases
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        enable_learning: bool = True
    ):
        self.model = model
        self.enable_learning = enable_learning
        self.session_id = str(uuid.uuid4())
        
        # Simple memory for this session
        self.classification_history: List[ClassificationResult] = []
        
        # Load FDA categories for reference
        self.fda_categories = self._load_fda_categories()
        
    def _load_fda_categories(self, fda_data_path: Optional[str] = None):
        """Load FDA categories and descriptions."""
        if fda_data_path:
            fda_path = pathlib.Path(fda_data_path)
        else:
            # Default path relative to the project root
            fda_path = pathlib.Path("data/fda_categories.json")

        if not fda_path.exists():
            raise FileNotFoundError(f"FDA data file not found at {fda_path}")

        try:
            with fda_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # Try to find the file again with fresh path resolution
            fda_json_path = _get_fda_json_path()
            try:
                with fda_json_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                # If still not found, return empty dict but log the issue
                print(f"Warning: FDA categories file not found at {fda_json_path}")
                print(f"Current working directory: {pathlib.Path.cwd()}")
                print(f"Script location: {pathlib.Path(__file__).resolve()}")
                return {}
    
    def classify_product(
        self, 
        product_name: str, 
        *, 
        explain: bool = True,
        method: str = "hybrid"
    ) -> ClassificationResult:
        """
        Classify a single product with optional explanation.
        
        Args:
            product_name: The product name to classify
            explain: Whether to generate reasoning
            method: Classification method ('hybrid' or 'semantic')
            
        Returns:
            ClassificationResult with category and reasoning
        """
        # Choose classification method
        if method == "semantic":
            category = classify_llm_semantic(product_name, model=self.model)
        else:
            category = classify_llm_hybrid(product_name, model=self.model)
        
        # Generate reasoning if requested
        reasoning = ""
        alternatives = []
        confidence = 0.9  # Default confidence
        
        if explain:
            reasoning, alternatives, confidence = self._generate_explanation(
                product_name, category
            )
        
        result = ClassificationResult(
            product_name=product_name,
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            timestamp=datetime.now(),
            session_id=self.session_id
        )
        
        # Store in memory
        self.classification_history.append(result)
        
        return result
    
    def classify_batch(
        self, 
        product_names: List[str],
        *, 
        explain: bool = False,
        method: str = "hybrid"
    ) -> List[ClassificationResult]:
        """Classify multiple products efficiently."""
        results = []
        
        for product_name in product_names:
            result = self.classify_product(
                product_name, 
                explain=explain, 
                method=method
            )
            results.append(result)
            
        return results
    
    def chat(self, user_input: str) -> AgentResponse:
        """
        Main chat interface for the agent.
        
        Handles natural language requests like:
        - "Classify this product: Organic Honey 12oz"
        - "What category is whole milk?"
        - "Process this list: [product1, product2, ...]"
        """
        user_input = user_input.strip()
        
        # Simple intent detection
        if any(keyword in user_input.lower() for keyword in ["classify", "category", "what is", "categorize"]):
            # Extract product name
            product_name = self._extract_product_name(user_input)
            
            if not product_name:
                return AgentResponse(
                    message="I'd be happy to classify a product! Please provide the product name.",
                    results=[],
                    suggestions=["Try: 'Classify Organic Honey 12oz'"],
                    needs_clarification=True,
                    clarification_question="What product would you like me to classify?"
                )
            
            result = self.classify_product(product_name, explain=True)
            
            message = f"I classified '{product_name}' as:\n\n"
            message += f"**Category:** {result.category}\n"
            message += f"**Confidence:** {result.confidence:.1%}\n\n"
            message += f"**Reasoning:** {result.reasoning}\n"
            
            if result.alternatives:
                message += f"\n**Alternative categories considered:** {', '.join(result.alternatives[:3])}"
            
            return AgentResponse(
                message=message,
                results=[result],
                suggestions=[
                    "Ask me to classify another product",
                    "Tell me if this classification seems wrong",
                    "Ask me to explain my reasoning in more detail"
                ]
            )
        
        else:
            # General response
            return AgentResponse(
                message="I'm an FDA product classification assistant. I can help you classify food products into FDA categories. Try asking me to classify a specific product! Say Classify [product name] to get started.",
                results=[],
                suggestions=[
                    "Classify [product name]",
                    "What category is [product name]?",
                    "Categorize this product: [product name]"
                ]
            )
    
    def _extract_product_name(self, user_input: str) -> str:
        """Extract product name from user input."""
        # Simple extraction - look for patterns
        patterns = [
            "classify ",
            "classify this product:",
            "classify this:",
            "what category is ",
            "categorize ",
            "categorize this product:",
            "categorize this:"
        ]
        
        lower_input = user_input.lower()
        for pattern in patterns:
            if pattern in lower_input:
                # Extract everything after the pattern
                start_idx = lower_input.index(pattern) + len(pattern)
                product_name = user_input[start_idx:].strip()
                # Remove common endings
                product_name = product_name.rstrip("?.")
                return product_name
        
        # If no pattern found, return the whole input (might be just a product name)
        return user_input
    
    def _generate_explanation(
        self, 
        product_name: str, 
        category: str
    ) -> Tuple[str, List[str], float]:
        """Generate explanation for classification."""
        # Simple explanation generation
        reasoning = f"Based on the product name '{product_name}', this appears to be classified as '{category}' because it matches the characteristics of this FDA category."
        
        # Generate some alternative categories
        alternatives = []
        if self.fda_categories:
            category_names = list(self.fda_categories.keys())
            # Remove the selected category and pick a few others
            other_categories = [c for c in category_names if c != category][:3]
            alternatives = other_categories
        
        # Improved confidence calculation
        confidence = self._calculate_confidence(product_name, category)
        
        return reasoning, alternatives, confidence
    
    def _calculate_confidence(self, product_name: str, category: str) -> float:
        """Calculate classification confidence based on various factors."""
        base_confidence = 0.85
        
        # Reduce confidence for MISC category
        if category == "MISC":
            return 0.3
        
        # Increase confidence for specific matches
        product_lower = product_name.lower()
        
        # High confidence indicators
        high_confidence_patterns = {
            "Juices, nectars, fruit drinks": ["juice", "nectar", "fruit drink"],
            "Juices, all varieties": ["baby juice", "infant juice", "gerber"],
            "Cheese, grated hard, e.g., Parmesan, Romano": ["grated", "parmesan", "romano"],
            "Pickles, all types": ["pickle", "pickled"],
            "Milk, milk-based drinks, e.g., instant breakfast, meal replacement, cocoa": ["milk", "dairy"]
        }
        
        if category in high_confidence_patterns:
            if any(pattern in product_lower for pattern in high_confidence_patterns[category]):
                base_confidence = min(0.95, base_confidence + 0.1)
        
        # Reduce confidence for ambiguous cases
        ambiguous_indicators = ["variety", "mixed", "assorted", "multi"]
        if any(indicator in product_lower for indicator in ambiguous_indicators):
            base_confidence *= 0.8
        
        # Reduce confidence for very short product names (likely incomplete)
        if len(product_name.split()) <= 1:
            base_confidence *= 0.7
        
        # Increase confidence for products with size/packaging info
        if any(indicator in product_lower for indicator in ['oz', 'lb', 'gallon', 'quart', 'pkg', 'pack']):
            base_confidence = min(0.95, base_confidence + 0.05)
        
        return round(base_confidence, 2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "total_classifications": len(self.classification_history),
            "session_id": self.session_id,
            "unique_categories": len(set(r.category for r in self.classification_history)),
            "avg_confidence": sum(r.confidence for r in self.classification_history) / len(self.classification_history) if self.classification_history else 0
        }
        
    def export_results(self, format: str = "json") -> str:
        """Export classification results."""
        if format == "json":
            data = [
                {
                    "product_name": r.product_name,
                    "category": r.category,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "alternatives": r.alternatives,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.classification_history
            ]
            return json.dumps(data, indent=2)
        else:
            # CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["product_name", "category", "confidence", "reasoning", "alternatives", "timestamp"])
            
            for r in self.classification_history:
                writer.writerow([
                    r.product_name,
                    r.category,
                    r.confidence,
                    r.reasoning,
                    ", ".join(r.alternatives),
                    r.timestamp.isoformat()
                ])
            
            return output.getvalue()


# For backward compatibility
ProductClassificationAgent = SimplifiedProductClassificationAgent 