"""
Simplified AI Agent for Product Classification

This is a streamlined version focusing on core FDA classification functionality.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pathlib

from ..llm.classifier import classify_llm_hybrid, classify_llm_semantic

# Get the FDA categories path
ROOT = pathlib.Path(__file__).resolve().parents[2]
FDA_JSON = ROOT / "data" / "interim" / "fda_categories.json"


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
        
    def _load_fda_categories(self) -> Dict[str, List[str]]:
        """Load FDA categories and descriptions."""
        try:
            with FDA_JSON.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
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
                message="I'm an FDA product classification assistant. I can help you classify food products into FDA categories. Try asking me to classify a specific product!",
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
        
        # Simple confidence based on category match
        confidence = 0.95 if category != "MISC" else 0.6
        
        return reasoning, alternatives, confidence
    
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