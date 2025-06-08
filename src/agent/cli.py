"""
Command-line interface for the Product Classification AI Agent.

Provides an interactive chat interface for classifying products.
"""

import argparse
import sys
from typing import Optional

from .core import ProductClassificationAgent


def interactive_mode(agent: ProductClassificationAgent) -> None:
    """Run the agent in interactive chat mode."""
    print("🤖 Product Classification AI Agent")
    print("=" * 50)
    print("I can help you classify products into FDA categories!")
    print("Type 'help' for assistance, 'stats' for session stats, or 'quit' to exit.")
    print()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for using the Product Classification Agent.")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'stats':
                print_stats(agent)
                continue
            
            elif user_input.lower() == 'clear':
                agent.memory.clear_session()
                print("🧹 Session cleared!")
                continue
            
            elif not user_input:
                continue
            
            # Get agent response
            response = agent.chat(user_input)
            
            print(f"🤖 Agent: {response.message}")
            
            if response.suggestions:
                print("\n💡 Suggestions:")
                for i, suggestion in enumerate(response.suggestions, 1):
                    print(f"   {i}. {suggestion}")
            
            if response.needs_clarification and response.clarification_question:
                print(f"\n❓ {response.clarification_question}")
            
            print()  # Add spacing
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'help' for assistance.")


def print_help() -> None:
    """Print help information."""
    help_text = """
🔗 Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 CLASSIFICATION
   • "Classify [product name]"     - Classify a single product
   • "Classify these: A, B, C"     - Classify multiple products
   • "What category is [product]?" - Alternative classification syntax

🧠 EXPLANATION & LEARNING  
   • "Why did you choose that?"    - Explain reasoning
   • "That's wrong, it should be X" - Provide feedback
   • "Explain [product]"           - Detailed explanation

📊 DATA MANAGEMENT
   • "Export my results"           - Export classifications
   • "Generate a report"           - Create classification report
   • stats                         - Show session statistics
   • clear                         - Clear current session

🔧 UTILITY
   • help                          - Show this help
   • quit/exit                     - Exit the agent

💡 EXAMPLES:
   • "Classify Organic Honey 12oz"
   • "Classify these: Milk, Bread, Apples, Cheese"
   • "Why did you put honey in that category?"
   • "Actually, that should be in the Sweeteners category"
"""
    print(help_text)


def print_stats(agent: ProductClassificationAgent) -> None:
    """Print session statistics."""
    stats = agent.get_stats()
    
    print("\n📊 Session Statistics")
    print("━" * 30)
    print(f"Session ID: {stats['session_id'][:8]}...")
    print(f"Classifications: {stats['total_classifications']}")
    print(f"Corrections: {stats['corrections_received']}")
    print(f"Categories Used: {stats['categories_used']}")
    print(f"Average Confidence: {stats['avg_confidence']:.1%}")
    
    if stats['total_classifications'] > 0:
        accuracy = 1 - (stats['corrections_received'] / stats['total_classifications'])
        print(f"Accuracy: {accuracy:.1%}")


def batch_mode(
    agent: ProductClassificationAgent, 
    products: list, 
    export_format: Optional[str] = None
) -> None:
    """Run the agent in batch processing mode."""
    print(f"🔄 Processing {len(products)} products...")
    
    results = agent.classify_batch(products, explain=False)
    
    print("\n📋 Results:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. {result.product_name:<30} → {result.category}")
    
    if export_format:
        exported = agent.export_results(export_format)
        filename = f"classifications_{agent.session_id[:8]}.{export_format}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(exported)
        
        print(f"\n💾 Results exported to: {filename}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Product Classification AI Agent",
        epilog="Examples:\n"
               "  python -m agent.cli                    # Interactive mode\n"
               "  python -m agent.cli --batch A,B,C      # Batch mode\n"
               "  python -m agent.cli --model gpt-4      # Use different model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model", 
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--batch",
        help="Comma-separated list of products to classify"
    )
    
    parser.add_argument(
        "--export",
        choices=["json", "csv"],
        help="Export format for batch results"
    )
    
    parser.add_argument(
        "--no-learning",
        action="store_true",
        help="Disable learning from user feedback"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    try:
        agent = ProductClassificationAgent(
            model=args.model,
            enable_learning=not args.no_learning
        )
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("Make sure you have OPENAI_API_KEY set in your environment.")
        sys.exit(1)
    
    # Run in appropriate mode
    if args.batch:
        products = [p.strip() for p in args.batch.split(',') if p.strip()]
        batch_mode(agent, products, args.export)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main() 