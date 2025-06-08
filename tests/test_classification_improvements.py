#!/usr/bin/env python3
"""
Test script to verify classification improvements.
Tests the specific misclassification issue from the user's screenshot.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core import ProductClassificationAgent

def test_classification_improvements():
    """Test the improved classification system."""
    
    # Initialize the agent
    agent = ProductClassificationAgent(model="gpt-4o", enable_learning=True)
    
    # Test cases from the user's screenshot
    test_products = [
        "Grated Cheese Tubs - Organic Local",
        "Bread and Butter Pickles", 
        "Whole Pickles Jar Packaged",
        "Orange Juice 64oz - Conventional",  # This was misclassified before
        "Oat Milk Unsweetened 6pk Value"
    ]
    
    expected_categories = [
        "Cheese, grated hard, e.g., Parmesan, Romano",
        "Pickles, all types",
        "Pickles, all types", 
        "Juices, nectars, fruit drinks",  # Should NOT be "Juices, all varieties"
        "Milk, milk-based drinks, e.g., instant breakfast, meal replacement, cocoa"
    ]
    
    print("🧪 Testing Classification Improvements")
    print("=" * 50)
    
    correct_predictions = 0
    total_predictions = len(test_products)
    
    for i, (product, expected) in enumerate(zip(test_products, expected_categories)):
        print(f"\n🔍 Test {i+1}: {product}")
        
        # Classify the product
        result = agent.classify_product(product, explain=True, method="hybrid")
        
        # Check if it matches expected
        is_correct = result.category == expected
        if is_correct:
            correct_predictions += 1
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        
        print(f"   Expected: {expected}")
        print(f"   Predicted: {result.category}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Status: {status}")
        
        if not is_correct:
            print(f"   ⚠️  This should be corrected by validation rules!")
    
    # Final accuracy
    accuracy = correct_predictions / total_predictions
    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    if accuracy >= 0.8:
        print("✅ Classification improvements are working well!")
    else:
        print("❌ Classification improvements need more work.")
    
    return accuracy

def test_critical_cases():
    """Test specific edge cases that were problematic before."""
    print("\n🎯 Testing Critical Edge Cases")
    print("=" * 35)
    
    critical_tests = [
        ("Orange Juice 64oz - Conventional", "Should be 'Juices, nectars, fruit drinks' NOT 'Juices, all varieties'"),
        ("Baby Apple Juice 4oz Gerber", "Should be 'Juices, all varieties' for infant products"),
        ("Grated Parmesan Cheese", "Should be 'Cheese, grated hard' category"),
    ]
    
    agent = ProductClassificationAgent(model="gpt-4o", enable_learning=True)
    
    for product, expectation in critical_tests:
        result = agent.classify_product(product, explain=True, method="hybrid")
        
        print(f"\n🔍 Product: {product}")
        print(f"   Classified as: {result.category}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Expectation: {expectation}")
        
        # Check for the specific orange juice issue
        if "Orange Juice 64oz" in product:
            if result.category == "Juices, all varieties":
                print(f"   ❌ STILL MISCLASSIFIED - This should NOT be an infant product!")
            elif result.category == "Juices, nectars, fruit drinks":
                print(f"   ✅ CORRECTLY CLASSIFIED - Fixed the infant product issue!")
            else:
                print(f"   ⚠️  UNEXPECTED CATEGORY - Neither infant nor general juice category")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    print("🚀 Starting Classification Improvement Tests\n")
    
    # Test critical edge cases
    test_critical_cases()
    
    # Test the full classification pipeline
    accuracy = test_classification_improvements()
    
    print(f"\n🎯 Focus: The AI should learn to distinguish infant vs general products!")
    print(f"   Improved prompts should help the AI make better decisions on its own.") 