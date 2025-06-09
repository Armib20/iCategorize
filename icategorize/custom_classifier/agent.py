"""
Custom Classification Agent for iTradenetwork

This agent combines category discovery, knowledge graphs, and LLM reasoning
to create business-relevant categorization for food product listings.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pathlib

from .category_discovery import CategoryDiscoveryEngine, CategoryInsight, ProductPattern
from .knowledge_graph import ProductKnowledgeGraph, GraphNode
from ..fda_classifier.classifier import classify_llm_hybrid, classify_llm_semantic
import openai


@dataclass
class CustomClassificationResult:
    """Result of custom product classification."""
    product_name: str
    primary_category: str
    secondary_categories: List[str]
    confidence: float
    reasoning: str
    business_insights: Dict[str, Any]
    patterns_matched: List[str]
    similar_products: List[str]
    timestamp: datetime
    session_id: str


@dataclass
class CategoryRecommendation:
    """Recommendation for creating a new category."""
    suggested_name: str
    description: str
    business_justification: str
    similar_products: List[str]
    estimated_size: int
    confidence: float


class CustomProductClassificationAgent:
    """
    Advanced classification agent for iTradenetwork that creates and uses
    custom business-relevant categories instead of generic FDA categories.
    
    Features:
    - Automatic category discovery from product data
    - Knowledge graph-based classification
    - Business-intelligent category recommendations
    - Continuous learning and adaptation
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o",
        auto_discover_categories: bool = True,
        min_products_for_category: int = 5,
        enable_knowledge_graph: bool = True
    ):
        self.model = model
        self.auto_discover_categories = auto_discover_categories
        self.min_products_for_category = min_products_for_category
        self.enable_knowledge_graph = enable_knowledge_graph
        
        # Core components
        self.discovery_engine = CategoryDiscoveryEngine(model=model)
        self.knowledge_graph = ProductKnowledgeGraph(model=model) if enable_knowledge_graph else None
        
        # State
        self.custom_categories: List[CategoryInsight] = []
        self.product_patterns: List[ProductPattern] = []
        self.classification_history: List[CustomClassificationResult] = []
        self.category_performance: Dict[str, Dict[str, float]] = {}
        
        # Try to load existing data
        self._load_existing_data()
    
    def bootstrap_from_product_data(
        self, 
        products: List[str],
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Bootstrap the system by analyzing existing product data to discover categories.
        
        Args:
            products: List of product names/descriptions from iTradenetwork
            force_rebuild: Whether to rebuild categories even if they exist
            
        Returns:
            Dictionary with discovered categories and statistics
        """
        if self.custom_categories and not force_rebuild:
            print(f"✅ Using existing {len(self.custom_categories)} categories")
            return self._get_bootstrap_summary()
        
        print(f"🚀 Bootstrapping custom categories from {len(products)} products...")
        
        # Adjust minimum cluster size for small datasets
        min_cluster_size = self.min_products_for_category
        if len(products) < 50:
            min_cluster_size = max(2, min_cluster_size // 2)
            print(f"📉 Small dataset detected, adjusting min_cluster_size to {min_cluster_size}")
        
        # Step 1: Discover categories
        categories = self.discovery_engine.analyze_product_data(
            products,
            min_cluster_size=min_cluster_size,
            max_categories=min(50, len(products) // 2)  # Reasonable max for small datasets
        )
        
        self.custom_categories = categories
        self.product_patterns = self.discovery_engine.product_patterns
        
        # Step 2: Build knowledge graph if enabled
        if self.enable_knowledge_graph and categories:  # Only build if we have categories
            print("🔧 Building knowledge graph from discoveries...")
            try:
                self.knowledge_graph.build_from_discoveries(
                    categories, self.product_patterns, products
                )
            except Exception as e:
                print(f"⚠️ Knowledge graph building failed: {e}")
                print("Continuing without knowledge graph...")
        
        # Step 3: Save discoveries
        self._save_custom_data()
        
        # Step 4: Generate business insights
        insights = self._generate_business_insights()
        
        return {
            'categories_discovered': len(categories),
            'patterns_found': len(self.product_patterns),
            'products_analyzed': len(products),
            'business_insights': insights,
            'category_hierarchy': self.discovery_engine.hierarchy,
            'top_categories': [
                {
                    'name': cat.name,
                    'size': cat.size,
                    'confidence': cat.confidence,
                    'keywords': cat.keywords[:5]
                }
                for cat in sorted(categories, key=lambda x: x.size, reverse=True)[:10]
            ]
        }
    
    def classify_product(
        self, 
        product_name: str, 
        explain: bool = True,
        suggest_new_category: bool = True
    ) -> CustomClassificationResult:
        """
        Classify a product using custom categories and knowledge graph.
        
        Args:
            product_name: Product to classify
            explain: Whether to generate detailed reasoning
            suggest_new_category: Whether to suggest creating new categories for poor matches
            
        Returns:
            Custom classification result with business insights
        """
        # Method 1: Knowledge graph classification (if available)
        graph_results = []
        if self.enable_knowledge_graph and self.knowledge_graph:
            try:
                graph_results = self.knowledge_graph.classify_product(
                    product_name, top_k=5
                )
            except Exception as e:
                print(f"Knowledge graph classification failed: {e}")
        
        # Method 2: Pattern matching with custom categories
        pattern_results = self._classify_using_patterns(product_name)
        
        # Method 3: LLM classification with custom categories
        llm_results = self._classify_using_llm(product_name)
        
        # Combine and rank results
        combined_results = self._combine_classification_methods(
            graph_results, pattern_results, llm_results
        )
        
        # Select best result
        if combined_results:
            primary_category = combined_results[0][0]
            confidence = combined_results[0][1]
            secondary_categories = [result[0] for result in combined_results[1:3]]
        else:
            primary_category = "Uncategorized"
            confidence = 0.1
            secondary_categories = []
        
        # Generate reasoning
        reasoning = ""
        if explain:
            reasoning = self._generate_classification_reasoning(
                product_name, primary_category, combined_results,
                graph_results, pattern_results, llm_results
            )
        
        # Business insights
        business_insights = self._generate_business_insights_for_product(
            product_name, primary_category, combined_results
        )
        
        # Find matched patterns
        patterns_matched = self._find_matched_patterns(product_name)
        
        # Find similar products
        similar_products = self._find_similar_products(product_name, top_k=5)
        
        # Check if we should suggest a new category
        if suggest_new_category and confidence < 0.6:
            category_recommendation = self._suggest_new_category(product_name)
            business_insights['new_category_suggestion'] = category_recommendation
        
        result = CustomClassificationResult(
            product_name=product_name,
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            confidence=confidence,
            reasoning=reasoning,
            business_insights=business_insights,
            patterns_matched=patterns_matched,
            similar_products=similar_products,
            timestamp=datetime.now(),
            session_id="custom_session"
        )
        
        # Store for learning
        self.classification_history.append(result)
        self._update_category_performance(result)
        
        return result
    
    def _classify_using_patterns(self, product_name: str) -> List[Tuple[str, float, str]]:
        """Classify using discovered patterns."""
        scores = {}
        product_lower = product_name.lower()
        
        for category in self.custom_categories:
            score = 0.0
            
            # Check keywords
            for keyword in category.keywords:
                if keyword.lower() in product_lower:
                    score += 0.3
            
            # Check if product is similar to sample products
            for sample in category.sample_products:
                if self._calculate_text_similarity(product_name, sample) > 0.7:
                    score += 0.4
            
            if score > 0:
                scores[category.name] = min(score, 1.0)
        
        # Convert to sorted list
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def _classify_using_llm(self, product_name: str) -> List[Tuple[str, float, str]]:
        """Classify using LLM with custom categories."""
        if not self.custom_categories:
            return []
        
        category_names = [cat.name for cat in self.custom_categories]
        category_descriptions = {
            cat.name: f"{cat.description} (Keywords: {', '.join(cat.keywords[:5])})"
            for cat in self.custom_categories
        }
        
        prompt = f"""Classify this product into the most appropriate business category:

Product: {product_name}

Available Categories:
{chr(10).join(f"- {name}: {desc}" for name, desc in category_descriptions.items())}

Consider:
1. What type of product this is
2. How buyers would search for it
3. Business relevance for food brokers
4. Product characteristics and use cases

Respond with JSON:
{{
  "primary_category": "Most appropriate category name",
  "confidence": 0.85,
  "reasoning": "Brief explanation of why this category fits",
  "alternatives": ["Second choice", "Third choice"]
}}"""

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text)
            
            # Format results
            formatted_results = []
            
            # Primary result
            if result.get('primary_category') in category_names:
                formatted_results.append((
                    result['primary_category'],
                    result.get('confidence', 0.8),
                    result.get('reasoning', '')
                ))
            
            # Alternatives
            for alt in result.get('alternatives', [])[:3]:
                if alt in category_names and alt != result.get('primary_category'):
                    formatted_results.append((alt, 0.6, ''))
            
            return formatted_results
            
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return []
    
    def _combine_classification_methods(
        self,
        graph_results: List[Tuple[str, float, str]],
        pattern_results: List[Tuple[str, float, str]],
        llm_results: List[Tuple[str, float, str]]
    ) -> List[Tuple[str, float]]:
        """Combine results from different classification methods."""
        combined_scores = {}
        
        # Weight the different methods
        graph_weight = 0.4
        pattern_weight = 0.3
        llm_weight = 0.3
        
        # Process graph results
        for category, score, _ in graph_results:
            if category not in combined_scores:
                combined_scores[category] = 0
            combined_scores[category] += score * graph_weight
        
        # Process pattern results
        for category, score in pattern_results:
            if category not in combined_scores:
                combined_scores[category] = 0
            combined_scores[category] += score * pattern_weight
        
        # Process LLM results
        for category, score, _ in llm_results:
            if category not in combined_scores:
                combined_scores[category] = 0
            combined_scores[category] += score * llm_weight
        
        # Sort and return
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        # Simple implementation - could be enhanced with embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _generate_classification_reasoning(
        self,
        product_name: str,
        primary_category: str,
        combined_results: List[Tuple[str, float]],
        graph_results: List[Tuple[str, float, str]],
        pattern_results: List[Tuple[str, float]],
        llm_results: List[Tuple[str, float, str]]
    ) -> str:
        """Generate detailed reasoning for the classification."""
        reasoning_parts = []
        
        # Primary classification source
        if graph_results and graph_results[0][0] == primary_category:
            reasoning_parts.append("Knowledge graph analysis suggests this category")
        elif pattern_results and pattern_results[0][0] == primary_category:
            reasoning_parts.append("Pattern matching indicates this category")
        elif llm_results and llm_results[0][0] == primary_category:
            reasoning_parts.append("AI semantic analysis recommends this category")
        
        # Confidence level
        if combined_results:
            confidence = combined_results[0][1]
            if confidence > 0.8:
                reasoning_parts.append("High confidence classification")
            elif confidence > 0.6:
                reasoning_parts.append("Moderate confidence classification")
            else:
                reasoning_parts.append("Low confidence - consider manual review")
        
        # Category insights
        category = next((cat for cat in self.custom_categories if cat.name == primary_category), None)
        if category:
            reasoning_parts.append(f"Category contains {category.size} similar products")
            if category.keywords:
                matching_keywords = [kw for kw in category.keywords 
                                   if kw.lower() in product_name.lower()]
                if matching_keywords:
                    reasoning_parts.append(f"Matches keywords: {', '.join(matching_keywords)}")
        
        return ". ".join(reasoning_parts)
    
    def _generate_business_insights_for_product(
        self,
        product_name: str,
        primary_category: str,
        results: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Generate business insights for the classified product."""
        insights = {}
        
        # Category popularity
        category = next((cat for cat in self.custom_categories if cat.name == primary_category), None)
        if category:
            insights['category_size'] = category.size
            insights['category_popularity'] = 'High' if category.size > 20 else 'Medium' if category.size > 10 else 'Low'
            insights['sample_products'] = category.sample_products[:3]
        
        # Market positioning
        if results:
            top_confidence = results[0][1]
            if top_confidence > 0.8:
                insights['market_position'] = 'Well-defined category with clear demand'
            elif top_confidence > 0.6:
                insights['market_position'] = 'Established category with good market fit'
            else:
                insights['market_position'] = 'Niche or emerging category'
        
        # Competition level
        num_competitors = len([r for r in results if r[1] > 0.3])
        insights['competition_level'] = 'High' if num_competitors > 5 else 'Medium' if num_competitors > 2 else 'Low'
        
        return insights
    
    def _find_matched_patterns(self, product_name: str) -> List[str]:
        """Find patterns that match the product."""
        matched = []
        product_lower = product_name.lower()
        
        for pattern in self.product_patterns:
            for value in pattern.values:
                if value.lower() in product_lower:
                    matched.append(f"{pattern.pattern_type}: {value}")
        
        return matched[:5]  # Limit to top 5
    
    def _find_similar_products(self, product_name: str, top_k: int = 5) -> List[str]:
        """Find similar products from the database."""
        similarities = []
        
        for category in self.custom_categories:
            for sample in category.sample_products:
                similarity = self._calculate_text_similarity(product_name, sample)
                if similarity > 0.3:
                    similarities.append((sample, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [product for product, _ in similarities[:top_k]]
    
    def _suggest_new_category(self, product_name: str) -> Optional[CategoryRecommendation]:
        """Suggest creating a new category for products that don't fit well."""
        # This is a simplified version - could be much more sophisticated
        
        # Use LLM to suggest a new category
        prompt = f"""This product doesn't fit well into existing categories: {product_name}

Existing categories:
{chr(10).join(f"- {cat.name}" for cat in self.custom_categories[:10])}

Should we create a new category? If so, suggest:

Respond with JSON:
{{
  "should_create": true/false,
  "suggested_name": "New Category Name",
  "description": "What this category would contain",
  "business_justification": "Why this would help buyers/sellers",
  "estimated_products": 5
}}"""

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            
            result = json.loads(result_text)
            
            if result.get('should_create'):
                return CategoryRecommendation(
                    suggested_name=result.get('suggested_name', ''),
                    description=result.get('description', ''),
                    business_justification=result.get('business_justification', ''),
                    similar_products=[product_name],
                    estimated_size=result.get('estimated_products', 5),
                    confidence=0.7
                )
        
        except Exception as e:
            print(f"New category suggestion failed: {e}")
        
        return None
    
    def _update_category_performance(self, result: CustomClassificationResult):
        """Update performance metrics for categories."""
        category = result.primary_category
        
        if category not in self.category_performance:
            self.category_performance[category] = {
                'total_classifications': 0,
                'avg_confidence': 0.0,
                'usage_frequency': 0.0
            }
        
        perf = self.category_performance[category]
        perf['total_classifications'] += 1
        
        # Update running average confidence
        old_avg = perf['avg_confidence']
        new_count = perf['total_classifications']
        perf['avg_confidence'] = (old_avg * (new_count - 1) + result.confidence) / new_count
        
        # Update usage frequency (simplified)
        perf['usage_frequency'] = new_count / len(self.classification_history)
    
    def _generate_business_insights(self) -> Dict[str, Any]:
        """Generate overall business insights about the category system."""
        if not self.custom_categories:
            return {}
        
        # Category size distribution
        sizes = [cat.size for cat in self.custom_categories]
        avg_size = sum(sizes) / len(sizes)
        
        # Most popular categories
        top_categories = sorted(self.custom_categories, key=lambda x: x.size, reverse=True)[:5]
        
        # Pattern analysis
        pattern_types = {}
        for pattern in self.product_patterns:
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
        
        return {
            'total_categories': len(self.custom_categories),
            'avg_category_size': avg_size,
            'largest_category': top_categories[0].name if top_categories else None,
            'top_categories': [{'name': cat.name, 'size': cat.size} for cat in top_categories],
            'pattern_types_found': pattern_types,
            'coverage_estimate': f"{sum(sizes)} products covered across {len(self.custom_categories)} categories"
        }
    
    def _get_bootstrap_summary(self) -> Dict[str, Any]:
        """Get summary of existing bootstrap data."""
        return {
            'categories_available': len(self.custom_categories),
            'patterns_available': len(self.product_patterns),
            'status': 'Using existing categories',
            'top_categories': [
                {'name': cat.name, 'size': cat.size, 'confidence': cat.confidence}
                for cat in sorted(self.custom_categories, key=lambda x: x.size, reverse=True)[:5]
            ]
        }
    
    def _save_custom_data(self):
        """Save custom categories and patterns."""
        self.discovery_engine.export_categories("data/custom_categories.json")
        
        if self.enable_knowledge_graph and self.knowledge_graph:
            self.knowledge_graph.save_graph("data/knowledge_graph.pkl")
    
    def _load_existing_data(self):
        """Load existing categories and knowledge graph."""
        categories_path = pathlib.Path("data/custom_categories.json")
        
        if categories_path.exists():
            try:
                with open(categories_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Recreate CategoryInsight objects
                self.custom_categories = []
                for cat_data in data.get('categories', []):
                    category = CategoryInsight(
                        name=cat_data['name'],
                        description=cat_data['description'],
                        keywords=cat_data['keywords'],
                        sample_products=cat_data['sample_products'],
                        size=cat_data['size'],
                        confidence=cat_data['confidence'],
                        parent_category=cat_data.get('parent_category')
                    )
                    self.custom_categories.append(category)
                
                # Recreate ProductPattern objects
                self.product_patterns = []
                for pattern_data in data.get('patterns', []):
                    pattern = ProductPattern(
                        pattern_type=pattern_data['type'],
                        values=pattern_data['values'],
                        frequency=pattern_data['frequency'],
                        examples=pattern_data['examples']
                    )
                    self.product_patterns.append(pattern)
                
                print(f"✅ Loaded {len(self.custom_categories)} categories and {len(self.product_patterns)} patterns")
                
            except Exception as e:
                print(f"Error loading categories: {e}")
        
        # Load knowledge graph
        if self.enable_knowledge_graph and self.knowledge_graph:
            self.knowledge_graph.load_graph("data/knowledge_graph.pkl")
    
    def get_category_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for improving the category system."""
        recommendations = []
        
        # Find categories that might need splitting
        for category in self.custom_categories:
            if category.size > 50:  # Large category
                recommendations.append({
                    'type': 'split_category',
                    'category': category.name,
                    'reason': f'Large category with {category.size} products could be split for better granularity',
                    'priority': 'medium'
                })
        
        # Find categories that might need merging
        similar_pairs = []
        for i, cat1 in enumerate(self.custom_categories):
            for cat2 in self.custom_categories[i+1:]:
                # Simple similarity check
                common_keywords = set(cat1.keywords) & set(cat2.keywords)
                if len(common_keywords) >= 2:
                    similar_pairs.append((cat1.name, cat2.name))
        
        for cat1, cat2 in similar_pairs[:3]:  # Limit recommendations
            recommendations.append({
                'type': 'merge_categories',
                'categories': [cat1, cat2],
                'reason': 'Categories have overlapping keywords and might be merged',
                'priority': 'low'
            })
        
        return recommendations
    
    def export_categories_for_review(self, filepath: str = "data/categories_for_review.csv") -> str:
        """Export categories in a format suitable for business review."""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Category Name', 'Description', 'Size', 'Confidence', 
                'Keywords', 'Sample Products', 'Business Relevance'
            ])
            
            for category in sorted(self.custom_categories, key=lambda x: x.size, reverse=True):
                writer.writerow([
                    category.name,
                    category.description,
                    category.size,
                    f"{category.confidence:.2f}",
                    '; '.join(category.keywords),
                    '; '.join(category.sample_products[:3]),
                    'High' if category.size > 20 else 'Medium' if category.size > 10 else 'Low'
                ])
        
        return filepath 