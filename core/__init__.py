"""
Core FDA Product Classification Components

This module provides the essential components for FDA product classification:
- SimplifiedProductClassificationAgent: Main agent for product classification
- classify_llm_hybrid, classify_llm_semantic: LLM-based classification functions
"""

from .agent import SimplifiedProductClassificationAgent, ClassificationResult, AgentResponse
from .classifier import classify_llm_hybrid, classify_llm_semantic, classify_llm

# For backwards compatibility
ProductClassificationAgent = SimplifiedProductClassificationAgent

__all__ = [
    'SimplifiedProductClassificationAgent',
    'ProductClassificationAgent',  # backwards compatibility
    'ClassificationResult', 
    'AgentResponse',
    'classify_llm_hybrid',
    'classify_llm_semantic',
    'classify_llm'
] 