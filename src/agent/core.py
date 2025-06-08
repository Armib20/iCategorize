"""
Core AI Agent for Product Classification

This is the main agent class that orchestrates classification, reasoning,
memory, and tool usage to provide an intelligent product categorization service.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..llm.classifier import classify_llm_hybrid, classify_llm_semantic
from .memory import ConversationMemory
from .tools import ToolRegistry
from .reasoning import AgentReasoner


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


class ProductClassificationAgent:
    """
    AI Agent for intelligent product classification.
    
    This agent can:
    - Classify single products or batches
    - Learn from user feedback
    - Explain its reasoning
    - Handle ambiguous cases
    - Integrate with external systems
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        memory_size: int = 100,
        enable_learning: bool = True
    ):
        self.model = model
        self.enable_learning = enable_learning
        self.session_id = str(uuid.uuid4())
        
        # Initialize components
        self.memory = ConversationMemory(max_size=memory_size)
        self.tools = ToolRegistry()
        self.reasoner = AgentReasoner(model=model)
        
        # Agent state
        self.context = {}
        self.user_preferences = {}
        
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
        
        # Store in memory for learning
        self.memory.add_classification(result)
        
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
        - "I think that classification was wrong"
        - "Process this list: [product1, product2, ...]"
        """
        # Store user input in memory
        self.memory.add_user_message(user_input)
        
        # Use reasoner to understand intent and plan response
        intent, entities = self.reasoner.analyze_intent(user_input)
        
        if intent == "classify_single":
            product_name = entities.get("product_name", "")
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
                message += f"\n**Alternative categories considered:** {', '.join(result.alternatives)}"
            
            return AgentResponse(
                message=message,
                results=[result],
                suggestions=[
                    "Ask me to classify another product",
                    "Tell me if this classification seems wrong",
                    "Ask me to explain my reasoning in more detail"
                ]
            )
            
        elif intent == "classify_batch":
            product_names = entities.get("product_names", [])
            if not product_names:
                return AgentResponse(
                    message="I can classify multiple products at once! Please provide a list of product names.",
                    results=[],
                    suggestions=["Try: 'Classify these: Milk, Bread, Apples'"],
                    needs_clarification=True
                )
            
            results = self.classify_batch(product_names, explain=False)
            
            message = f"I classified {len(results)} products:\n\n"
            for i, result in enumerate(results, 1):
                message += f"{i}. **{result.product_name}** → {result.category}\n"
            
            return AgentResponse(
                message=message,
                results=results,
                suggestions=[
                    "Ask for detailed explanations of any classification",
                    "Request corrections if any seem wrong",
                    "Ask me to export these results"
                ]
            )
            
        elif intent == "feedback":
            # Handle user corrections and learning
            correction = entities.get("correction", {})
            if correction:
                self._learn_from_feedback(correction)
                message = "Thank you for the feedback! I've noted this correction and will improve."
            else:
                message = "I'd love to learn from your feedback! Please tell me what was incorrect."
                
            return AgentResponse(
                message=message,
                results=[],
                suggestions=[
                    "Try: 'That product should be in the Dairy category'",
                    "Or: 'Product X is actually category Y'"
                ]
            )
            
        elif intent == "explain":
            # Provide detailed explanations
            product_name = entities.get("product_name", "")
            if product_name:
                # Find recent classification or reclassify
                recent_result = self.memory.find_recent_classification(product_name)
                if recent_result:
                    explanation = self._generate_detailed_explanation(recent_result)
                    return AgentResponse(
                        message=explanation,
                        results=[recent_result],
                        suggestions=[]
                    )
            
            message = "I can explain my classification reasoning! Which product would you like me to explain?"
            return AgentResponse(
                message=message,
                results=[],
                suggestions=[],
                needs_clarification=True
            )
            
        else:
            # General conversation or help
            message = """I'm an AI agent specialized in product classification! Here's what I can do:

🏷️ **Classify Products**: "Classify Organic Honey 12oz"
📝 **Batch Processing**: "Classify these: Milk, Bread, Apples"  
🤔 **Explain Reasoning**: "Why did you choose that category?"
📚 **Learn from Feedback**: "That should be in the Dairy category"
📊 **Export Results**: "Export my classifications as CSV"

What would you like to do?"""
            
            return AgentResponse(
                message=message,
                results=[],
                suggestions=[
                    "Classify a product",
                    "Process a list of products", 
                    "Ask about my capabilities"
                ]
            )
    
    def _generate_explanation(
        self, 
        product_name: str, 
        category: str
    ) -> Tuple[str, List[str], float]:
        """Generate reasoning for a classification."""
        return self.reasoner.explain_classification(product_name, category)
    
    def _generate_detailed_explanation(self, result: ClassificationResult) -> str:
        """Generate a detailed explanation of a classification result."""
        return self.reasoner.detailed_explanation(result)
    
    def _learn_from_feedback(self, correction: Dict[str, Any]) -> None:
        """Learn from user corrections to improve future classifications."""
        if not self.enable_learning:
            return
            
        # Store the correction in memory
        self.memory.add_correction(correction)
        
        # Update user preferences if patterns emerge
        self._update_preferences(correction)
    
    def _update_preferences(self, correction: Dict[str, Any]) -> None:
        """Update user preferences based on feedback patterns."""
        # This could implement user-specific classification preferences
        pass
    
    def export_results(self, format: str = "json") -> str:
        """Export classification results in various formats."""
        return self.tools.export_classifications(
            self.memory.get_all_classifications(), 
            format
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance and usage statistics."""
        return {
            "session_id": self.session_id,
            "total_classifications": len(self.memory.classifications),
            "corrections_received": len(self.memory.corrections),
            "categories_used": len(set(r.category for r in self.memory.classifications)),
            "avg_confidence": sum(r.confidence for r in self.memory.classifications) / max(len(self.memory.classifications), 1)
        } 