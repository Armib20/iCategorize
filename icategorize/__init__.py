"""
iCategorize - Intelligent Product Classification

This package provides two main classification approaches:
- FDA Classification: Official FDA product category classification
- Custom Classification: Business-intelligent category discovery and classification
"""

# Import FDA classifier components
from .fda_classifier import SimplifiedProductClassificationAgent, classify_llm_hybrid, classify_llm_semantic, classify_llm

# Import custom classifier components  
from .custom_classifier import CustomProductClassificationAgent, CategoryDiscoveryEngine, ProductKnowledgeGraph

# For backwards compatibility
ProductClassificationAgent = SimplifiedProductClassificationAgent

__all__ = [
    'SimplifiedProductClassificationAgent',
    'ProductClassificationAgent',  # backwards compatibility
    'classify_llm_hybrid',
    'classify_llm_semantic', 
    'classify_llm',
    'CustomProductClassificationAgent',
    'CategoryDiscoveryEngine',
    'ProductKnowledgeGraph'
] 