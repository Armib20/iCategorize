"""
Custom Category Discovery System

This module analyzes product listing data to automatically discover
business-relevant categories and subcategories for iTradenetwork.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import pathlib
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import openai
from rapidfuzz import fuzz, process


@dataclass
class CategoryInsight:
    """Insight about a discovered category."""
    name: str
    description: str
    keywords: List[str]
    sample_products: List[str]
    size: int
    confidence: float
    parent_category: Optional[str] = None
    subcategories: List[str] = None


@dataclass
class ProductPattern:
    """Pattern discovered in product data."""
    pattern_type: str  # origin, season, processing, size, etc.
    values: List[str]
    frequency: int
    examples: List[str]


class CategoryDiscoveryEngine:
    """
    Engine for discovering custom categories from product listing data.
    
    This uses a multi-step approach:
    1. Text analysis and clustering
    2. Pattern recognition (origin, seasonality, processing)
    3. AI-powered semantic grouping
    4. Hierarchy generation
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.discovered_categories = []
        self.product_patterns = []
        self.hierarchy = {}
        
    def analyze_product_data(
        self, 
        products: List[str], 
        min_cluster_size: int = 5,
        max_categories: int = 50
    ) -> List[CategoryInsight]:
        """
        Analyze product data to discover natural categories.
        
        Args:
            products: List of product names/descriptions
            min_cluster_size: Minimum products per category
            max_categories: Maximum number of categories to discover
            
        Returns:
            List of discovered category insights
        """
        print(f"🔍 Analyzing {len(products)} products for category discovery...")
        
        # Step 1: Text preprocessing and feature extraction
        processed_products = self._preprocess_products(products)
        
        # Step 2: Discover patterns (origin, seasonality, processing, etc.)
        patterns = self._discover_patterns(processed_products)
        self.product_patterns = patterns
        
        # Step 3: Semantic clustering
        clusters = self._perform_semantic_clustering(
            processed_products, 
            min_cluster_size=min_cluster_size,
            max_clusters=max_categories
        )
        
        # Step 4: Generate category insights
        categories = self._generate_category_insights(clusters, patterns)
        
        # Step 5: Create hierarchy
        hierarchy = self._create_category_hierarchy(categories)
        self.hierarchy = hierarchy
        
        self.discovered_categories = categories
        print(f"✅ Discovered {len(categories)} categories")
        
        return categories
    
    def _preprocess_products(self, products: List[str]) -> List[Dict[str, Any]]:
        """Preprocess and extract features from product names."""
        processed = []
        
        for product in products:
            # Extract various features
            features = {
                'original': product,
                'clean': self._clean_product_name(product),
                'origin': self._extract_origin(product),
                'season': self._extract_season_indicators(product),
                'processing': self._extract_processing_level(product),
                'size_indicators': self._extract_size_info(product),
                'quality_indicators': self._extract_quality_indicators(product),
                'main_ingredient': self._extract_main_ingredient(product)
            }
            processed.append(features)
            
        return processed
    
    def _clean_product_name(self, product: str) -> str:
        """Clean and normalize product name."""
        # Remove extra whitespace, special characters
        cleaned = re.sub(r'[^\w\s-]', '', product.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _extract_origin(self, product: str) -> Optional[str]:
        """Extract geographic origin from product name."""
        origins = [
            'california', 'florida', 'texas', 'mexico', 'chile', 'peru',
            'organic', 'local', 'imported', 'domestic', 'usa', 'american',
            'italian', 'spanish', 'french', 'dutch', 'canadian'
        ]
        
        product_lower = product.lower()
        for origin in origins:
            if origin in product_lower:
                return origin.title()
        return None
    
    def _extract_season_indicators(self, product: str) -> List[str]:
        """Extract seasonal indicators."""
        seasons = {
            'spring': ['spring', 'early', 'new'],
            'summer': ['summer', 'peak', 'ripe'],
            'fall': ['fall', 'autumn', 'late'],
            'winter': ['winter', 'storage', 'cold']
        }
        
        found_seasons = []
        product_lower = product.lower()
        
        for season, indicators in seasons.items():
            if any(indicator in product_lower for indicator in indicators):
                found_seasons.append(season)
                
        return found_seasons
    
    def _extract_processing_level(self, product: str) -> str:
        """Determine processing level."""
        product_lower = product.lower()
        
        if any(term in product_lower for term in ['fresh', 'raw', 'whole']):
            return 'fresh'
        elif any(term in product_lower for term in ['frozen', 'freeze']):
            return 'frozen'
        elif any(term in product_lower for term in ['dried', 'dehydrated']):
            return 'dried'
        elif any(term in product_lower for term in ['canned', 'jarred', 'preserved']):
            return 'processed'
        elif any(term in product_lower for term in ['juice', 'puree', 'concentrate']):
            return 'liquid'
        else:
            return 'unknown'
    
    def _extract_size_info(self, product: str) -> Optional[str]:
        """Extract size/packaging information."""
        size_patterns = [
            r'\d+\s*(oz|lb|lbs|pound|pounds|kg|g|gram|grams)',
            r'\d+\s*(count|ct|piece|pieces|pack)',
            r'(small|medium|large|extra\s*large|jumbo|baby)',
            r'\d+\s*(gallon|quart|pint|liter|ml)'
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, product.lower())
            if match:
                return match.group(0)
        return None
    
    def _extract_quality_indicators(self, product: str) -> List[str]:
        """Extract quality/grade indicators."""
        quality_terms = [
            'organic', 'premium', 'grade a', 'choice', 'select', 'prime',
            'artisanal', 'gourmet', 'specialty', 'conventional', 'standard',
            'fresh', 'frozen', 'natural', 'pure', 'authentic', 'traditional',
            'non-gmo', 'fair trade', 'sustainable', 'heirloom', 'wild-caught',
            'free-range', 'grass-fed', 'cage-free', 'kosher', 'halal'
        ]
        
        found_qualities = []
        product_lower = product.lower()
        
        for quality in quality_terms:
            if quality in product_lower:
                found_qualities.append(quality.title())
                
        return found_qualities
    
    def _extract_main_ingredient(self, product: str) -> str:
        """Extract the main ingredient/product type."""
        # This is a simplified version - in practice, you'd use NLP
        # or a more sophisticated ingredient extraction method
        
        # Common ingredient categories
        fruits = ['apple', 'orange', 'banana', 'grape', 'berry', 'citrus', 'stone fruit']
        vegetables = ['lettuce', 'tomato', 'carrot', 'pepper', 'onion', 'potato', 'leafy']
        proteins = ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'turkey']
        dairy = ['milk', 'cheese', 'yogurt', 'butter', 'cream']
        
        product_lower = product.lower()
        
        # Check each category
        for fruit in fruits:
            if fruit in product_lower:
                return 'fruit'
        for vegetable in vegetables:
            if vegetable in product_lower:
                return 'vegetable'
        for protein in proteins:
            if protein in product_lower:
                return 'protein'
        for d in dairy:
            if d in product_lower:
                return 'dairy'
                
        return 'other'
    
    def _discover_patterns(self, processed_products: List[Dict]) -> List[ProductPattern]:
        """Discover patterns across products."""
        patterns = []
        
        # Analyze each feature type
        feature_types = ['origin', 'season', 'processing', 'quality_indicators', 'main_ingredient']
        
        for feature_type in feature_types:
            values = []
            examples = []
            
            for product in processed_products:
                feature_value = product.get(feature_type)
                if feature_value:
                    if isinstance(feature_value, list):
                        values.extend(feature_value)
                        examples.extend([product['original']] * len(feature_value))
                    else:
                        values.append(feature_value)
                        examples.append(product['original'])
            
            # Count frequencies
            value_counts = Counter(values)
            
            # Create patterns for significant values
            for value, count in value_counts.items():
                if count >= 3:  # Minimum frequency
                    pattern = ProductPattern(
                        pattern_type=feature_type,
                        values=[value],
                        frequency=count,
                        examples=[ex for ex, val in zip(examples, values) if val == value][:5]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _perform_semantic_clustering(
        self, 
        processed_products: List[Dict], 
        min_cluster_size: int = 5,
        max_clusters: int = 50
    ) -> List[List[str]]:
        """Perform semantic clustering on products."""
        print(f"🔍 Clustering {len(processed_products)} products...")
        
        # Adjust parameters for small datasets
        if len(processed_products) < 20:
            # For very small datasets, reduce minimum cluster size
            min_cluster_size = max(2, min_cluster_size // 2)
            print(f"📉 Small dataset detected, reducing min_cluster_size to {min_cluster_size}")
        
        # Create text representations with emphasis on quality indicators
        texts = []
        for product in processed_products:
            # Combine various features into a text representation
            # Give more weight to quality indicators by repeating them
            quality_indicators = product.get('quality_indicators', [])
            quality_text = ' '.join(quality_indicators * 3)  # Triple weight for quality
            
            text_parts = [
                product['clean'],
                quality_text,  # Quality indicators get priority
                product.get('origin', ''),
                product.get('processing', ''),
                product.get('main_ingredient', '')
            ]
            texts.append(' '.join(filter(None, text_parts)))
        
        # Filter out empty texts
        non_empty_texts = []
        product_mapping = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text)
                product_mapping.append(i)
        
        if not non_empty_texts:
            print("⚠️ No meaningful text data found for clustering")
            return []
        
        print(f"📝 Processing {len(non_empty_texts)} non-empty product descriptions")
        
        # TF-IDF Vectorization with adjusted parameters for small datasets
        vectorizer = TfidfVectorizer(
            max_features=min(1000, len(non_empty_texts) * 10),  # Adjust max features
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1 if len(non_empty_texts) < 10 else 2  # Allow single occurrences for very small datasets
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(non_empty_texts)
        except ValueError as e:
            if "empty vocabulary" in str(e):
                print("⚠️ Empty vocabulary after processing - creating simple text-based clusters")
                return self._create_fallback_clusters(processed_products, min_cluster_size)
            else:
                raise
        
        # Determine optimal number of clusters
        optimal_clusters = self._find_optimal_clusters(tfidf_matrix, max_clusters)
        print(f"🎯 Using {optimal_clusters} clusters")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group products by cluster (map back to original products)
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            original_idx = product_mapping[i]
            clusters[label].append(processed_products[original_idx]['original'])
        
        # Filter out small clusters
        filtered_clusters = []
        for cluster_products in clusters.values():
            if len(cluster_products) >= min_cluster_size:
                filtered_clusters.append(cluster_products)
                print(f"✅ Found cluster with {len(cluster_products)} products")
            else:
                print(f"⚠️ Filtered out small cluster with {len(cluster_products)} products")
        
        print(f"🎉 Final result: {len(filtered_clusters)} clusters discovered")
        return filtered_clusters

    def _create_fallback_clusters(self, processed_products: List[Dict], min_cluster_size: int) -> List[List[str]]:
        """Create simple clusters based on main ingredient when TF-IDF fails."""
        print("🔄 Creating fallback clusters based on main ingredients...")
        
        ingredient_clusters = defaultdict(list)
        
        for product in processed_products:
            main_ingredient = product.get('main_ingredient', 'other')
            ingredient_clusters[main_ingredient].append(product['original'])
        
        # Filter clusters by minimum size
        filtered_clusters = []
        for ingredient, products in ingredient_clusters.items():
            if len(products) >= min_cluster_size:
                filtered_clusters.append(products)
                print(f"✅ Fallback cluster '{ingredient}': {len(products)} products")
        
        return filtered_clusters
    
    def _find_optimal_clusters(self, tfidf_matrix, max_clusters: int) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        if tfidf_matrix.shape[0] < 4:
            return 2
        
        max_k = min(max_clusters, tfidf_matrix.shape[0] - 1)
        if max_k < 2:
            return 2
        
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, 11))  # Test up to 10 clusters
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, labels)
            silhouette_scores.append(score)
        
        # Find the k with the highest silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        return best_k
    
    def _generate_category_insights(
        self, 
        clusters: List[List[str]], 
        patterns: List[ProductPattern]
    ) -> List[CategoryInsight]:
        """Generate insights for each discovered category using AI."""
        categories = []
        
        for i, cluster_products in enumerate(clusters):
            # Use AI to analyze the cluster and generate category info
            category_info = self._analyze_cluster_with_ai(cluster_products, patterns)
            
            if category_info:
                insight = CategoryInsight(
                    name=category_info['name'],
                    description=category_info['description'],
                    keywords=category_info.get('keywords', []),
                    sample_products=cluster_products[:10],
                    size=len(cluster_products),
                    confidence=category_info.get('confidence', 0.8)
                )
                categories.append(insight)
        
        return categories
    
    def _analyze_cluster_with_ai(
        self, 
        products: List[str], 
        patterns: List[ProductPattern]
    ) -> Optional[Dict[str, Any]]:
        """Use AI to analyze a cluster and generate category information."""
        # Sample products for AI analysis (to avoid token limits)
        sample_products = products[:20]
        
        # Find relevant patterns for this cluster
        relevant_patterns = []
        for pattern in patterns:
            # Check if any pattern examples are in this cluster
            if any(example in products for example in pattern.examples):
                relevant_patterns.append(pattern)
        
        prompt = f"""Analyze these products and create a business-relevant category:

Products ({len(products)} total, showing sample):
{chr(10).join(f"- {p}" for p in sample_products)}

Discovered patterns in this group:
{chr(10).join(f"- {p.pattern_type}: {', '.join(p.values)} (appears {p.frequency} times)" for p in relevant_patterns[:5])}

Based on this analysis, create a category that would be useful for food brokers and buyers. Consider:
1. What makes these products similar?
2. What would buyers search for?
3. How granular should this category be?
4. What are the key distinguishing features?

IMPORTANT GUIDELINES:
- Prioritize quality indicators (organic, premium, fresh) over general product types
- Group products by their most specific shared attribute (organic > fresh > bulk)
- Avoid overly broad categories like "Fresh Produce" - be more specific like "Organic Fresh Produce"
- Consider business context: buyers often search by quality first, then product type
- If products share quality attributes (organic, premium, artisanal), make that the primary category focus

Examples of good categories:
- "Organic Fresh Produce" (not just "Fresh Produce")
- "Premium Frozen Seafood" (not just "Frozen Foods")
- "Bulk Specialty Nuts" (not just "Nuts")

Respond with JSON in this format:
{{
  "name": "Category Name (specific and business-relevant)",
  "description": "Brief description of what defines this category",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "confidence": 0.85,
  "business_relevance": "Why this category matters for buyers/sellers"
}}"""

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up JSON response
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            return json.loads(result)
            
        except Exception as e:
            print(f"Error analyzing cluster: {e}")
            return None
    
    def _create_category_hierarchy(self, categories: List[CategoryInsight]) -> Dict[str, Any]:
        """Create a hierarchy from discovered categories."""
        # Use AI to suggest hierarchical relationships
        hierarchy = {}
        
        # Group similar categories
        for category in categories:
            # Simple grouping by main ingredient/type
            main_type = self._get_category_main_type(category)
            
            if main_type not in hierarchy:
                hierarchy[main_type] = {
                    'subcategories': [],
                    'description': f"All {main_type} related products"
                }
            
            hierarchy[main_type]['subcategories'].append({
                'name': category.name,
                'description': category.description,
                'size': category.size,
                'keywords': category.keywords
            })
        
        return hierarchy
    
    def _get_category_main_type(self, category: CategoryInsight) -> str:
        """Determine the main type for hierarchy grouping."""
        # Analyze category name and keywords
        name_lower = category.name.lower()
        keywords_lower = [k.lower() for k in category.keywords]
        
        # Define main types
        type_mapping = {
            'Fresh Produce': ['fresh', 'fruit', 'vegetable', 'produce', 'organic'],
            'Processed Foods': ['canned', 'frozen', 'dried', 'processed'],
            'Proteins': ['meat', 'poultry', 'seafood', 'fish', 'protein'],
            'Dairy': ['dairy', 'milk', 'cheese', 'yogurt'],
            'Beverages': ['juice', 'drink', 'beverage', 'liquid'],
            'Specialty': ['gourmet', 'artisanal', 'specialty', 'premium']
        }
        
        # Check for matches
        for main_type, indicators in type_mapping.items():
            if any(ind in name_lower for ind in indicators) or \
               any(ind in ' '.join(keywords_lower) for ind in indicators):
                return main_type
        
        return 'Other'
    
    def export_categories(self, filepath: str = "data/custom_categories.json"):
        """
        Exports the discovered categories and patterns to a JSON file.
        
        Args:
            filepath: Path to the output JSON file.
        """
        # Convert CategoryInsight objects to dictionaries
        categories_data = []
        for cat in self.discovered_categories:
            categories_data.append({
                'name': cat.name,
                'description': cat.description,
                'keywords': cat.keywords,
                'sample_products': cat.sample_products,
                'size': cat.size,
                'confidence': cat.confidence,
                'parent_category': cat.parent_category,
                'subcategories': cat.subcategories or []
            })
        
        # Convert ProductPattern objects to dictionaries
        patterns_data = []
        for pattern in self.product_patterns:
            patterns_data.append({
                'type': pattern.pattern_type,
                'values': pattern.values,
                'frequency': pattern.frequency,
                'examples': pattern.examples
            })
        
        output_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_categories": len(self.discovered_categories),
                "total_patterns": len(self.product_patterns),
            },
            "hierarchy": self.hierarchy,
            "categories": categories_data,
            "patterns": patterns_data,
        }
        
        # Ensure the directory exists
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Categories and patterns exported to {filepath}") 