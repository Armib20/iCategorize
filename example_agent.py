#!/usr/bin/env python3
"""
Example script demonstrating the Product Classification AI Agent.

This shows how to use the agent programmatically for various tasks.
"""

import os
from src.agent.core import ProductClassificationAgent


def main():
    """Demonstrate agent capabilities."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("🤖 Product Classification AI Agent Demo")
    print("=" * 50)
    
    # Initialize the agent
    agent = ProductClassificationAgent(
        model="gpt-4o",
        enable_learning=True
    )
    
    print("✅ Agent initialized successfully!")
    print()
    
    # Example 1: Single product classification
    print("📦 Example 1: Single Product Classification")
    print("-" * 45)
    
    result = agent.classify_product("Organic Honey 12oz", explain=True)
    
    print(f"Product: {result.product_name}")
    print(f"Category: {result.category}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Reasoning: {result.reasoning}")
    print()
    
    # Example 2: Batch classification
    print("📦 Example 2: Batch Classification")
    print("-" * 36)
    
    products = [
        "Whole Milk 1 Gallon",
        "Sourdough Bread Loaf",
        "Fresh Organic Apples",
        "Cheddar Cheese Block",
        "Greek Yogurt 32oz"
    ]
    
    results = agent.classify_batch(products, explain=False)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.product_name:<25} → {result.category}")
    print()
    
    # Example 3: Chat interface
    print("💬 Example 3: Chat Interface")
    print("-" * 30)
    
    # Simulate a conversation
    conversations = [
        "Classify Organic Coconut Oil",
        "Why did you choose that category?",
        "Actually, that should be in the Cooking Oils category",
        "Classify these: Pasta, Rice, Quinoa"
    ]
    
    for user_input in conversations:
        print(f"👤 User: {user_input}")
        response = agent.chat(user_input)
        print(f"🤖 Agent: {response.message}")
        
        if response.suggestions:
            print("💡 Suggestions:", ", ".join(response.suggestions[:2]))
        print()
    
    # Example 4: Export and statistics
    print("📊 Example 4: Statistics and Export")
    print("-" * 35)
    
    stats = agent.get_stats()
    print(f"Session ID: {stats['session_id'][:8]}...")
    print(f"Total Classifications: {stats['total_classifications']}")
    print(f"Categories Used: {stats['categories_used']}")
    print(f"Average Confidence: {stats['avg_confidence']:.1%}")
    print()
    
    # Export results
    exported_json = agent.export_results("json")
    print("📄 Exported results (JSON preview):")
    print(exported_json[:200] + "..." if len(exported_json) > 200 else exported_json)
    print()
    
    # Example 5: Advanced usage with tools
    print("🔧 Example 5: Advanced Tools")
    print("-" * 26)
    
    # Generate a report
    report = agent.tools.generate_classification_report(
        agent.memory.get_all_classifications(),
        agent.memory.corrections
    )
    
    print("📋 Classification Report Preview:")
    print(report[:300] + "..." if len(report) > 300 else report)
    print()
    
    print("✅ Demo completed successfully!")
    print()
    print("💡 Next steps:")
    print("- Try the CLI: python -m src.agent.cli")
    print("- Start web interface: python -m src.agent.web")
    print("- Read full docs: README-Agent.md")


if __name__ == "__main__":
    main() 