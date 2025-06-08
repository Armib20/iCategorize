#!/usr/bin/env python3
"""
Simple test script for the FDA Product Classification Agent
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ProductClassificationAgent

def test_agent():
    """Test the simplified agent functionality."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return False
    
    print("🤖 Testing FDA Product Classification Agent")
    print("=" * 50)
    
    # Initialize agent
    agent = ProductClassificationAgent(model="gpt-4o")
    print("✅ Agent initialized")
    
    # Test products
    test_products = [
        "Organic Honey 12oz",
        "Whole Milk 1 Gallon", 
        "Fresh Blueberries 6oz",
        "Sourdough Bread Loaf"
    ]
    
    print(f"\n🔄 Testing {len(test_products)} products...")
    print("-" * 50)
    
    for product in test_products:
        print(f"\n📦 Product: {product}")
        
        try:
            result = agent.classify_product(product, explain=True)
            print(f"   Category: {result.category}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Reasoning: {result.reasoning[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test chat interface
    print(f"\n💬 Testing chat interface...")
    print("-" * 50)
    
    chat_tests = [
        "Classify Organic Apple Juice",
        "What category is cheddar cheese?",
        "Hello, how are you?"
    ]
    
    for chat_input in chat_tests:
        print(f"\n👤 User: {chat_input}")
        try:
            response = agent.chat(chat_input)
            print(f"🤖 Assistant: {response.message[:150]}...")
            if response.results:
                print(f"   📊 Classifications: {len(response.results)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Show stats
    print(f"\n📊 Session Statistics:")
    print("-" * 50)
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Test completed!")
    return True

if __name__ == "__main__":
    test_agent() 