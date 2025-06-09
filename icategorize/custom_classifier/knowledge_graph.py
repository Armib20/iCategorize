"""
Knowledge Graph for Product Classification

This module creates and manages a knowledge graph representation of products,
categories, and their relationships for intelligent classification decisions.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
import pathlib

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai


@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    id: str
    type: str  # 'product', 'category', 'attribute', 'pattern'
    name: str
    properties: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None


@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    source: str
    target: str
    relationship: str  # 'belongs_to', 'similar_to', 'has_attribute', 'related_to'
    weight: float
    properties: Dict[str, Any] = None


class ProductKnowledgeGraph:
    """
    Knowledge graph for product categorization.
    
    This graph captures:
    - Products and their features
    - Categories and subcategories
    - Relationships between products
    - Patterns and rules
    - Semantic embeddings
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.graph = nx.MultiDiGraph()
        self.nodes = {}  # id -> GraphNode
        self.embeddings_cache = {}
        self.tfidf_vectorizer = None
        self.category_embeddings = {}
        
    def build_from_discoveries(
        self, 
        categories: List[Any],  # CategoryInsight objects
        patterns: List[Any],    # ProductPattern objects
        products: List[str]
    ):
        """Build knowledge graph from discovered categories and patterns."""
        print("🔧 Building knowledge graph from discoveries...")
        
        # Step 1: Add category nodes
        self._add_category_nodes(categories)
        
        # Step 2: Add pattern nodes
        self._add_pattern_nodes(patterns)
        
        # Step 3: Add product nodes and relationships
        self._add_product_nodes(products, categories)
        
        # Step 4: Create relationships between categories
        self._create_category_relationships(categories)
        
        # Step 5: Create pattern relationships
        self._create_pattern_relationships(patterns)
        
        # Step 6: Generate embeddings
        self._generate_embeddings()
        
        print(f"✅ Knowledge graph built with {len(self.nodes)} nodes and {self.graph.number_of_edges()} edges")
    
    def _add_category_nodes(self, categories: List[Any]):
        """Add category nodes to the graph."""
        for category in categories:
            node_id = f"category_{category.name.replace(' ', '_').lower()}"
            
            node = GraphNode(
                id=node_id,
                type='category',
                name=category.name,
                properties={
                    'description': category.description,
                    'keywords': category.keywords,
                    'size': category.size,
                    'confidence': category.confidence,
                    'sample_products': category.sample_products
                }
            )
            
            self.nodes[node_id] = node
            # Add node with name property for visualization
            self.graph.add_node(node_id, name=category.name, **node.properties)
    
    def _add_pattern_nodes(self, patterns: List[Any]):
        """Add pattern nodes to the graph."""
        for i, pattern in enumerate(patterns):
            node_id = f"pattern_{pattern.pattern_type}_{i}"
            
            node = GraphNode(
                id=node_id,
                type='pattern',
                name=f"{pattern.pattern_type}: {', '.join(pattern.values)}",
                properties={
                    'pattern_type': pattern.pattern_type,
                    'values': pattern.values,
                    'frequency': pattern.frequency,
                    'examples': pattern.examples
                }
            )
            
            self.nodes[node_id] = node
            # Add node with name property for visualization
            self.graph.add_node(node_id, name=node.name, **node.properties)
    
    def _add_product_nodes(self, products: List[str], categories: List[Any]):
        """Add product nodes and connect them to categories."""
        # Create a mapping of products to categories
        product_to_category = {}
        for category in categories:
            for product in category.sample_products:
                product_to_category[product] = category.name
        
        for i, product in enumerate(products):
            node_id = f"product_{i}"
            
            # Extract features from product name
            features = self._extract_product_features(product)
            
            node = GraphNode(
                id=node_id,
                type='product',
                name=product,
                properties=features
            )
            
            self.nodes[node_id] = node
            # Add node with name property for visualization
            self.graph.add_node(node_id, name=product, **node.properties)
            
            # Connect to category if known
            if product in product_to_category:
                category_name = product_to_category[product]
                category_id = f"category_{category_name.replace(' ', '_').lower()}"
                
                if category_id in self.nodes:
                    edge = GraphEdge(
                        source=node_id,
                        target=category_id,
                        relationship='belongs_to',
                        weight=1.0
                    )
                    self.graph.add_edge(node_id, category_id, 
                                      relationship='belongs_to', weight=1.0)
    
    def _extract_product_features(self, product: str) -> Dict[str, Any]:
        """Extract features from a product name."""
        # Reuse the feature extraction logic from CategoryDiscoveryEngine
        features = {
            'original_name': product,
            'clean_name': product.lower().strip(),
            'length': len(product),
            'word_count': len(product.split()),
        }
        
        # Add more sophisticated feature extraction here
        # This could include NLP, ingredient detection, etc.
        
        return features
    
    def _create_category_relationships(self, categories: List[Any]):
        """Create relationships between categories based on similarity."""
        if not categories or len(categories) < 2:
            print("⚠️ Not enough categories to create relationships")
            return
            
        similarity_matrix = self._calculate_category_similarity(categories)
        
        # Create edges for similar categories
        threshold = 0.3  # Similarity threshold
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i != j and similarity_matrix[i][j] > threshold:
                    category1_id = f"category_{cat1.name.replace(' ', '_').lower()}"
                    category2_id = f"category_{cat2.name.replace(' ', '_').lower()}"
                    
                    self.graph.add_edge(category1_id, category2_id,
                                      relationship='similar_to',
                                      weight=float(similarity_matrix[i][j]))
    
    def _calculate_category_similarity(self, categories: List[Any]) -> np.ndarray:
        """Calculate similarity matrix for categories."""
        if not categories:
            return np.array([[]])
        
        # Create text representations for each category
        texts = []
        for category in categories:
            # Combine category name, description, and keywords
            text_parts = [
                category.name,
                getattr(category, 'description', ''),
                ' '.join(getattr(category, 'keywords', [])),
                ' '.join(getattr(category, 'sample_products', [])[:3])  # First 3 sample products
            ]
            text = ' '.join(filter(None, text_parts))
            texts.append(text)
        
        # Handle case where all texts are empty or only stop words
        if not texts or all(not text.strip() for text in texts):
            print("⚠️ All category texts are empty, returning identity matrix")
            return np.eye(len(categories))
        
        if not self.tfidf_vectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000,
                min_df=1  # Allow single occurrences for small datasets
            )
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError as e:
            if "empty vocabulary" in str(e):
                print("⚠️ Empty vocabulary after TF-IDF processing, returning identity matrix")
                return np.eye(len(categories))
            else:
                raise
        
        return similarity_matrix
    
    def _create_pattern_relationships(self, patterns: List[Any]):
        """Create relationships between patterns and categories/products."""
        # Connect patterns to categories based on shared products
        for i, pattern in enumerate(patterns):
            pattern_id = f"pattern_{pattern.pattern_type}_{i}"
            
            # Find categories that contain products from this pattern
            for category_id, category_node in self.nodes.items():
                if category_node.type == 'category':
                    sample_products = category_node.properties.get('sample_products', [])
                    
                    # Check if any pattern examples are in this category
                    overlap = set(pattern.examples) & set(sample_products)
                    
                    if overlap:
                        # Calculate overlap ratio
                        overlap_ratio = len(overlap) / len(pattern.examples)
                        
                        if overlap_ratio > 0.2:  # At least 20% overlap
                            self.graph.add_edge(pattern_id, category_id,
                                              relationship='influences',
                                              weight=overlap_ratio)
    
    def _generate_embeddings(self):
        """Generate semantic embeddings for nodes."""
        print("🧠 Generating semantic embeddings...")
        
        # Generate embeddings for categories
        for node_id, node in self.nodes.items():
            if node.type == 'category':
                text = f"{node.name} {node.properties.get('description', '')}"
                embedding = self._get_text_embedding(text)
                node.embeddings = embedding
                self.category_embeddings[node_id] = embedding
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using OpenAI's embedding model."""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(1536)  # Default embedding size
    
    def classify_product(
        self, 
        product_name: str, 
        top_k: int = 3
    ) -> List[Tuple[str, float, str]]:
        """
        Classify a product using the knowledge graph.
        
        Returns:
            List of (category_name, confidence, reasoning) tuples
        """
        # Extract features from the product
        product_features = self._extract_product_features(product_name)
        
        # Get embedding for the product
        product_embedding = self._get_text_embedding(product_name)
        
        # Method 1: Semantic similarity with categories
        semantic_scores = self._calculate_semantic_similarity(product_embedding)
        
        # Method 2: Pattern matching
        pattern_scores = self._calculate_pattern_similarity(product_name)
        
        # Method 3: Graph traversal
        graph_scores = self._calculate_graph_similarity(product_name, product_features)
        
        # Combine scores
        combined_scores = self._combine_classification_scores(
            semantic_scores, pattern_scores, graph_scores
        )
        
        # Get top-k results with reasoning
        results = []
        for category_id, score in sorted(combined_scores.items(), 
                                       key=lambda x: x[1], reverse=True)[:top_k]:
            
            category_node = self.nodes[category_id]
            reasoning = self._generate_classification_reasoning(
                product_name, category_node, score, 
                semantic_scores.get(category_id, 0),
                pattern_scores.get(category_id, 0),
                graph_scores.get(category_id, 0)
            )
            
            results.append((category_node.name, score, reasoning))
        
        return results
    
    def _calculate_semantic_similarity(self, product_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate semantic similarity with category embeddings."""
        scores = {}
        
        for category_id, category_embedding in self.category_embeddings.items():
            if category_embedding is not None:
                similarity = cosine_similarity(
                    product_embedding.reshape(1, -1),
                    category_embedding.reshape(1, -1)
                )[0][0]
                scores[category_id] = max(0, similarity)  # Ensure non-negative
        
        return scores
    
    def _calculate_pattern_similarity(self, product_name: str) -> Dict[str, float]:
        """Calculate similarity based on patterns."""
        scores = defaultdict(float)
        product_lower = product_name.lower()
        
        # Check each pattern node
        for node_id, node in self.nodes.items():
            if node.type == 'pattern':
                # Check if any pattern values match the product
                pattern_values = node.properties.get('values', [])
                
                for value in pattern_values:
                    if value.lower() in product_lower:
                        # Find categories influenced by this pattern
                        for target in self.graph.successors(node_id):
                            if self.nodes[target].type == 'category':
                                edge_data = self.graph.get_edge_data(node_id, target)
                                if edge_data and 'weight' in edge_data[0]:
                                    scores[target] += edge_data[0]['weight'] * 0.5
        
        return dict(scores)
    
    def _calculate_graph_similarity(
        self, 
        product_name: str, 
        product_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate similarity using graph structure."""
        scores = defaultdict(float)
        
        # Find similar products in the graph
        similar_products = self._find_similar_products(product_name, top_k=10)
        
        for similar_product_id, similarity in similar_products:
            # Find categories this similar product belongs to
            for target in self.graph.successors(similar_product_id):
                if self.nodes[target].type == 'category':
                    edge_data = self.graph.get_edge_data(similar_product_id, target)
                    if edge_data and edge_data[0].get('relationship') == 'belongs_to':
                        scores[target] += similarity * 0.3
        
        return dict(scores)
    
    def _find_similar_products(self, product_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar products in the graph."""
        similarities = []
        product_embedding = self._get_text_embedding(product_name)
        
        for node_id, node in self.nodes.items():
            if node.type == 'product':
                # Calculate text similarity
                node_embedding = self._get_text_embedding(node.name)
                similarity = cosine_similarity(
                    product_embedding.reshape(1, -1),
                    node_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append((node_id, similarity))
        
        # Return top-k most similar
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _combine_classification_scores(
        self, 
        semantic_scores: Dict[str, float],
        pattern_scores: Dict[str, float],
        graph_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine different scoring methods."""
        all_categories = set()
        all_categories.update(semantic_scores.keys())
        all_categories.update(pattern_scores.keys())
        all_categories.update(graph_scores.keys())
        
        combined_scores = {}
        
        for category_id in all_categories:
            # Weighted combination
            semantic_weight = 0.5
            pattern_weight = 0.3
            graph_weight = 0.2
            
            score = (
                semantic_scores.get(category_id, 0) * semantic_weight +
                pattern_scores.get(category_id, 0) * pattern_weight +
                graph_scores.get(category_id, 0) * graph_weight
            )
            
            combined_scores[category_id] = score
        
        return combined_scores
    
    def _generate_classification_reasoning(
        self,
        product_name: str,
        category_node: GraphNode,
        total_score: float,
        semantic_score: float,
        pattern_score: float,
        graph_score: float
    ) -> str:
        """Generate human-readable reasoning for classification."""
        reasoning_parts = []
        
        # Primary reasoning source
        if semantic_score > pattern_score and semantic_score > graph_score:
            reasoning_parts.append(f"Semantic similarity to '{category_node.name}' category")
        elif pattern_score > graph_score:
            reasoning_parts.append(f"Matches patterns associated with '{category_node.name}'")
        else:
            reasoning_parts.append(f"Similar to known products in '{category_node.name}'")
        
        # Add specific insights
        if semantic_score > 0.3:
            reasoning_parts.append(f"Strong semantic match (score: {semantic_score:.2f})")
        
        if pattern_score > 0.2:
            reasoning_parts.append(f"Pattern-based evidence (score: {pattern_score:.2f})")
        
        if graph_score > 0.1:
            reasoning_parts.append(f"Graph relationship evidence (score: {graph_score:.2f})")
        
        # Add category description insight
        description = category_node.properties.get('description', '')
        if description and len(description) > 10:
            reasoning_parts.append(f"Category description: {description[:100]}...")
        
        return ". ".join(reasoning_parts)
    
    def get_category_insights(self, category_name: str) -> Dict[str, Any]:
        """Get insights about a specific category."""
        category_id = f"category_{category_name.replace(' ', '_').lower()}"
        
        if category_id not in self.nodes:
            return {}
        
        category_node = self.nodes[category_id]
        
        # Find connected patterns
        connected_patterns = []
        for predecessor in self.graph.predecessors(category_id):
            if self.nodes[predecessor].type == 'pattern':
                pattern_node = self.nodes[predecessor]
                connected_patterns.append({
                    'type': pattern_node.properties['pattern_type'],
                    'values': pattern_node.properties['values'],
                    'frequency': pattern_node.properties['frequency']
                })
        
        # Find similar categories
        similar_categories = []
        for successor in self.graph.successors(category_id):
            if (self.nodes[successor].type == 'category' and 
                self.graph.get_edge_data(category_id, successor)[0].get('relationship') == 'similar_to'):
                
                weight = self.graph.get_edge_data(category_id, successor)[0]['weight']
                similar_categories.append({
                    'name': self.nodes[successor].name,
                    'similarity': weight
                })
        
        # Count connected products
        connected_products = sum(1 for pred in self.graph.predecessors(category_id)
                               if self.nodes[pred].type == 'product')
        
        return {
            'name': category_node.name,
            'description': category_node.properties.get('description', ''),
            'keywords': category_node.properties.get('keywords', []),
            'size': category_node.properties.get('size', 0),
            'connected_products': connected_products,
            'connected_patterns': connected_patterns,
            'similar_categories': similar_categories
        }
    
    def save_graph(self, filepath: str = "data/knowledge_graph.pkl"):
        """Save the knowledge graph to file."""
        # Ensure the directory exists
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"✅ Knowledge graph saved to {filepath}")
    
    def load_graph(self, filepath: str = "data/knowledge_graph.pkl"):
        """Load the graph from a file."""
        graph_path = pathlib.Path(filepath)
        if graph_path.exists():
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
            print(f"✅ Knowledge graph loaded from {filepath}")
        else:
            print(f"⚠️ Knowledge graph file not found at {filepath}. A new one will be created.")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for node_id, node in self.nodes.items():
            node_types[node.type] += 1
        
        for source, target, data in self.graph.edges(data=True):
            edge_types[data.get('relationship', 'unknown')] += 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'is_connected': nx.is_weakly_connected(self.graph),
            'density': nx.density(self.graph)
        } 