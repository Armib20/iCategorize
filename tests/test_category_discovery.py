#!/usr/bin/env python3
"""
Test script for Category Discovery functionality.

This script tests the core category discovery process to ensure it works correctly
and returns proper data structures.
"""

import os
import sys
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_category_discovery():
    """Test the category discovery process with sample data."""
    print("🧪 Testing Category Discovery Process...")
    
    try:
        from icategorize.custom_classifier import CustomProductClassificationAgent
        print("✅ Successfully imported CustomProductClassificationAgent")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Sample products for testing
    test_products = [
        "Organic Hass Avocados, Box of 48, Mexico",
        "California Navel Oranges, 5lb bag",
        "Fresh Blueberries, 1 pint, Grade A",
        "Sweet Onions, 50lb sack, Washington",
        "Idaho Russet Potatoes, 10lb bag",
        "Honeycrisp Apples, Case, USA",
        "Organic Baby Spinach, 1lb clam shell",
        "Vine-ripened Tomatoes, 25lb box, Grade 1",
        "Fresh Atlantic Salmon Fillets, Norway",
        "Grass-Fed Ground Beef, 80/20, Texas",
        "Aged Cheddar Cheese, 24-month, Wisconsin",
        "Whole Grain Artisan Bread, Local",
        "Premium Olive Oil Extra Virgin, Italy",
        "Free-Range Chicken Breasts, USA",
        "Greek Yogurt Plain, Organic"
    ]
    
    print(f"📊 Testing with {len(test_products)} sample products")
    
    try:
        # Initialize agent
        print("🔧 Initializing CustomProductClassificationAgent...")
        agent = CustomProductClassificationAgent(
            enable_knowledge_graph=True, 
            model="gpt-4o"
        )
        print("✅ Agent initialized successfully")
        
        # Run category discovery
        print("🔍 Running category discovery...")
        results = agent.bootstrap_from_product_data(test_products, force_rebuild=True)
        
        # Analyze results
        print("\n📊 DISCOVERY RESULTS:")
        print("=" * 50)
        
        print(f"Categories Discovered: {results.get('categories_discovered', 0)}")
        print(f"Product Patterns Found: {results.get('patterns_found', 0)}")
        print(f"Products Analyzed: {results.get('products_analyzed', 0)}")
        
        # Check agent state
        print(f"\nAgent Categories: {len(agent.custom_categories) if agent.custom_categories else 0}")
        print(f"Agent Patterns: {len(agent.product_patterns) if agent.product_patterns else 0}")
        
        # Display discovered categories
        if agent.custom_categories:
            print(f"\n📂 DISCOVERED CATEGORIES ({len(agent.custom_categories)}):")
            print("-" * 50)
            for i, category in enumerate(agent.custom_categories, 1):
                print(f"{i}. {category.name}")
                print(f"   Description: {category.description}")
                print(f"   Size: {category.size} products")
                print(f"   Confidence: {category.confidence:.2%}")
                print(f"   Keywords: {', '.join(category.keywords[:5])}")
                print(f"   Sample Products: {category.sample_products[:3]}")
                print()
        
        # Display discovered patterns
        if hasattr(agent, 'discovery_engine') and agent.discovery_engine.product_patterns:
            patterns = agent.discovery_engine.product_patterns
            print(f"\n🔍 DISCOVERED PATTERNS ({len(patterns)}):")
            print("-" * 50)
            for i, pattern in enumerate(patterns, 1):
                print(f"{i}. {pattern.pattern_type}: {', '.join(pattern.values)} (freq: {pattern.frequency})")
        
        # Test classification
        print(f"\n🎯 TESTING CLASSIFICATION:")
        print("-" * 50)
        test_product = "Organic Fresh Avocados from Mexico, Premium Grade"
        print(f"Classifying: {test_product}")
        
        classification_result = agent.classify_product(test_product)
        print(f"Primary Category: {classification_result.primary_category}")
        print(f"Confidence: {classification_result.confidence:.2%}")
        print(f"Secondary Categories: {classification_result.secondary_categories}")
        
        # Test knowledge graph
        if agent.knowledge_graph:
            print(f"\n🕸️ KNOWLEDGE GRAPH:")
            print("-" * 50)
            graph_stats = agent.knowledge_graph.get_graph_stats()
            for key, value in graph_stats.items():
                print(f"{key}: {value}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_discovery_engine_directly():
    """Test the CategoryDiscoveryEngine directly."""
    print("\n🔬 Testing CategoryDiscoveryEngine directly...")
    
    try:
        from icategorize.custom_classifier.category_discovery import CategoryDiscoveryEngine
        
        # Sample products
        products = [
            "Organic Hass Avocados, Mexico",
            "Fresh Avocados, Peru",
            "California Navel Oranges",
            "Florida Orange Juice",
            "Atlantic Salmon Fillets",
            "Wild Salmon, Norway",
            "Cheddar Cheese, Aged",
            "Swiss Cheese, Premium"
        ]
        
        engine = CategoryDiscoveryEngine(model="gpt-4o")
        categories = engine.analyze_product_data(products, min_cluster_size=2, max_categories=10)
        
        print(f"Direct test - Categories discovered: {len(categories)}")
        for category in categories:
            print(f"  - {category.name} ({category.size} products)")
        
        return len(categories) > 0
        
    except Exception as e:
        print(f"❌ Direct engine test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Category Discovery Tests\n")
    
    # Test 1: Full agent test
    success1 = test_category_discovery()
    
    # Test 2: Direct engine test
    success2 = test_discovery_engine_directly()
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"Full Agent Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Direct Engine Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The category discovery system is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.") 