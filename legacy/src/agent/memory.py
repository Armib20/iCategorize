"""
Memory system for the AI agent.

Handles conversation history, classification results, user feedback,
and learning from interactions.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class Message:
    """A conversation message."""
    role: str  # 'user' or 'agent'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class Correction:
    """User feedback/correction."""
    product_name: str
    wrong_category: str
    correct_category: str
    reasoning: str
    timestamp: datetime


class ConversationMemory:
    """
    Manages agent memory including conversations, classifications, and learning.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        
        # Conversation history
        self.messages: deque = deque(maxlen=max_size)
        
        # Classification results
        self.classifications: List = []
        
        # User corrections for learning
        self.corrections: List[Correction] = []
        
        # User preferences learned over time
        self.user_preferences: Dict[str, Any] = {}
        
        # Performance tracking
        self.stats = {
            "total_interactions": 0,
            "successful_classifications": 0,
            "user_corrections": 0,
            "session_start": datetime.now()
        }
    
    def add_user_message(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a user message to conversation history."""
        message = Message(
            role="user",
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.stats["total_interactions"] += 1
    
    def add_agent_message(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add an agent response to conversation history."""
        message = Message(
            role="agent", 
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
    
    def add_classification(self, result) -> None:
        """Store a classification result."""
        self.classifications.append(result)
        if result.category != "MISC":
            self.stats["successful_classifications"] += 1
    
    def add_correction(self, correction_data: Dict[str, Any]) -> None:
        """Store user feedback/correction."""
        correction = Correction(
            product_name=correction_data.get("product_name", ""),
            wrong_category=correction_data.get("wrong_category", ""),
            correct_category=correction_data.get("correct_category", ""),
            reasoning=correction_data.get("reasoning", ""),
            timestamp=datetime.now()
        )
        self.corrections.append(correction)
        self.stats["user_corrections"] += 1
        
        # Update user preferences based on corrections
        self._update_preferences_from_correction(correction)
    
    def find_recent_classification(self, product_name: str, hours: int = 24) -> Optional:
        """Find a recent classification for a product."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        for result in reversed(self.classifications):
            if (result.product_name.lower() == product_name.lower() and 
                result.timestamp > cutoff):
                return result
        return None
    
    def get_conversation_context(self, last_n: int = 10) -> List[Message]:
        """Get recent conversation context."""
        return list(self.messages)[-last_n:]
    
    def get_classification_history(self, last_n: int = 20) -> List:
        """Get recent classification history."""
        return self.classifications[-last_n:]
    
    def get_all_classifications(self) -> List:
        """Get all classifications in this session."""
        return self.classifications.copy()
    
    def get_user_patterns(self) -> Dict[str, Any]:
        """Analyze user interaction patterns."""
        if not self.classifications:
            return {}
        
        categories_used = {}
        for result in self.classifications:
            categories_used[result.category] = categories_used.get(result.category, 0) + 1
        
        return {
            "most_common_categories": sorted(
                categories_used.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            "avg_confidence": sum(r.confidence for r in self.classifications) / len(self.classifications),
            "correction_rate": len(self.corrections) / max(len(self.classifications), 1)
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from corrections for learning."""
        if not self.corrections:
            return {"corrections": []}
        
        # Analyze correction patterns
        category_corrections = {}
        for correction in self.corrections:
            key = f"{correction.wrong_category} -> {correction.correct_category}"
            category_corrections[key] = category_corrections.get(key, 0) + 1
        
        return {
            "total_corrections": len(self.corrections),
            "correction_patterns": category_corrections,
            "recent_corrections": [asdict(c) for c in self.corrections[-5:]]
        }
    
    def _update_preferences_from_correction(self, correction: Correction) -> None:
        """Update user preferences based on correction."""
        # Track which categories the user prefers for ambiguous cases
        pref_key = f"prefer_{correction.correct_category}_over_{correction.wrong_category}"
        self.user_preferences[pref_key] = self.user_preferences.get(pref_key, 0) + 1
    
    def export_memory(self, format: str = "json") -> str:
        """Export memory data for analysis or backup."""
        data = {
            "session_stats": self.stats,
            "classifications": [asdict(r) for r in self.classifications],
            "corrections": [asdict(c) for c in self.corrections],
            "user_preferences": self.user_preferences,
            "conversation_length": len(self.messages)
        }
        
        if format == "json":
            return json.dumps(data, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_session(self, keep_preferences: bool = True) -> None:
        """Clear session data, optionally keeping user preferences."""
        self.messages.clear()
        self.classifications.clear()
        self.corrections.clear()
        
        if not keep_preferences:
            self.user_preferences.clear()
        
        self.stats = {
            "total_interactions": 0,
            "successful_classifications": 0, 
            "user_corrections": 0,
            "session_start": datetime.now()
        } 