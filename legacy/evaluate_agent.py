#!/usr/bin/env python3
"""
Evaluate the Product Classification AI Agent on test dataset.

Tests the agent against known ground truth classifications.
"""

import csv
import os
import sys
from typing import List, Tuple, Dict
from dataclasses import dataclass

from src.agent.core import ProductClassificationAgent


@dataclass
class EvaluationResult:
    """Result of evaluating one product."""
    product_name: str
    predicted_category: str
    ground_truth: str
    is_correct: bool
    confidence: float
    reasoning: str


def load_test_data(csv_path: str = "data/samples/test_products.csv") -> List[Tuple[str, str]]:
    """Load test products and ground truth from CSV."""
    test_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_name = row['product_name'].strip()
            ground_truth = row['ground_truth'].strip()
            test_data.append((product_name, ground_truth))
    
    return test_data


def evaluate_agent(agent: ProductClassificationAgent, test_data: List[Tuple[str, str]]) -> List[EvaluationResult]:
    """Evaluate agent performance on test data."""
    results = []
    
    print(f"🔄 Evaluating agent on {len(test_data)} products...")
    print("=" * 80)
    
    for i, (product_name, ground_truth) in enumerate(test_data, 1):
        print(f"[{i:2d}/{len(test_data)}] Testing: {product_name}")
        
        # Get agent classification
        classification_result = agent.classify_product(product_name, explain=True)
        
        # Check if correct
        is_correct = classification_result.category == ground_truth
        
        result = EvaluationResult(
            product_name=product_name,
            predicted_category=classification_result.category,
            ground_truth=ground_truth,
            is_correct=is_correct,
            confidence=classification_result.confidence,
            reasoning=classification_result.reasoning
        )
        
        results.append(result)
        
        # Show result
        status = "✅" if is_correct else "❌"
        print(f"     {status} Predicted: {classification_result.category}")
        if not is_correct:
            print(f"       Expected: {ground_truth}")
        print()
    
    return results


def print_detailed_results(results: List[EvaluationResult]) -> None:
    """Print detailed evaluation results."""
    correct = sum(1 for r in results if r.is_correct)
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Products: {total}")
    print(f"Correct Classifications: {correct}")
    print(f"Incorrect Classifications: {total - correct}")
    print(f"Accuracy: {accuracy:.1%}")
    print()
    
    # Show average confidence
    avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
    print(f"Average Confidence: {avg_confidence:.1%}")
    print()
    
    # Show misclassifications
    if any(not r.is_correct for r in results):
        print("❌ MISCLASSIFICATIONS:")
        print("-" * 60)
        
        for result in results:
            if not result.is_correct:
                print(f"Product: {result.product_name}")
                print(f"  Predicted: {result.predicted_category}")
                print(f"  Expected:  {result.ground_truth}")
                print(f"  Confidence: {result.confidence:.1%}")
                print(f"  Reasoning: {result.reasoning[:100]}...")
                print()
    
    # Show perfect classifications
    if any(r.is_correct for r in results):
        print("✅ CORRECT CLASSIFICATIONS:")
        print("-" * 60)
        
        for result in results:
            if result.is_correct:
                print(f"✓ {result.product_name:<30} → {result.predicted_category}")
        print()


def analyze_errors(results: List[EvaluationResult]) -> None:
    """Analyze error patterns."""
    errors = [r for r in results if not r.is_correct]
    
    if not errors:
        print("🎉 No errors to analyze!")
        return
    
    print("🔍 ERROR ANALYSIS:")
    print("-" * 40)
    
    # Group errors by predicted category
    predicted_categories = {}
    for error in errors:
        pred = error.predicted_category
        predicted_categories[pred] = predicted_categories.get(pred, 0) + 1
    
    print("Most common incorrect predictions:")
    for category, count in sorted(predicted_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} errors")
    print()
    
    # Low confidence errors
    low_confidence_errors = [e for e in errors if e.confidence < 0.7]
    if low_confidence_errors:
        print(f"Low confidence errors ({len(low_confidence_errors)}):")
        for error in low_confidence_errors:
            print(f"  {error.product_name} (confidence: {error.confidence:.1%})")
        print()


def enhanced_classify(product_name: str):
    # Rule-based overrides for edge cases
    if "chips" in product_name.lower() and any(fruit in product_name.lower() 
                                             for fruit in ["banana", "apple", "fruit"]):
        return "snack_chips_category"
    
    if "juice" in product_name.lower() and "100%" in product_name:
        return "pure_juice_category" 
    
    # Fall back to AI classification
    return ai_classify(product_name)


def main():
    """Main evaluation function."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("🤖 Product Classification AI Agent Evaluation")
    print("=" * 80)
    
    try:
        # Load test data
        test_data = load_test_data("data/samples/blueberries_eval.csv")
        print(f"📄 Loaded {len(test_data)} test products")
        print()
        
        # Initialize agent
        agent = ProductClassificationAgent(
            model="gpt-4o",
            enable_learning=False  # Disable learning for consistent evaluation
        )
        print("✅ Agent initialized")
        print()
        
        # Run evaluation
        results = evaluate_agent(agent, test_data)
        
        # Print detailed results
        print_detailed_results(results)
        
        # Analyze errors
        analyze_errors(results)
        
        # Compare with previous results
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        print("📈 COMPARISON WITH PREVIOUS APPROACHES:")
        print("-" * 50)
        print(f"Manual Alias Approach:     52.4% (11/21)")
        print(f"AI-Native Agent Approach:  {accuracy:.1%} ({correct}/{total})")
        
        if accuracy > 0.524:
            improvement = ((accuracy - 0.524) / 0.524) * 100
            print(f"Improvement: +{improvement:.1f}% 🎉")
        else:
            decline = ((0.524 - accuracy) / 0.524) * 100
            print(f"Performance: -{decline:.1f}% ⚠️")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 