"""
End-to-end demo for the iTradenetwork Custom Classification Agent.

This script demonstrates:
1.  Initializing the custom classification agent.
2.  Bootstrapping the system with sample product data to discover categories.
3.  Classifying new products using the discovered custom categories.
4.  Displaying detailed, business-relevant classification results.
5.  Getting recommendations for improving the category system.
6.  Exporting discovered categories for business review.
"""
import os
import sys
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from icategorize.custom_classifier import CustomProductClassificationAgent

def get_sample_products():
    """Provides a list of sample product names for iTradenetwork."""
    return [
        "Organic Hass Avocados, Box of 48, Mexico",
        "California Navel Oranges, 5lb bag",
        "Fresh Blueberries, 1 pint, Grade A",
        "Sweet Onions, 50lb sack, Washington",
        "Idaho Russet Potatoes, 10lb bag",
        "Honeycrisp Apples, Case, USA",
        "Organic Baby Spinach, 1lb clam shell",
        "Vine-ripened Tomatoes, 25lb box, Grade 1",
        "Frozen Corn, Sweet Yellow, 20lb case",
        "Canned Diced Tomatoes, 6/#10 cans/case",
        "Imported Italian Pasta, Spaghetti, 20x1lb",
        "Extra Virgin Olive Oil, 4x1 Gallon, Spain",
        "Almonds, Raw, Unsalted, 25lb box, California",
        "Chicken Breast, Boneless, Skinless, Frozen, 40lb case",
        "Ground Beef 80/20, 10lb tubes, fresh",
        "Shredded Mozzarella Cheese, 4x5lb bags",
        "Sourdough Bread, Par-baked, 24 loaves",
        "Hass Avocados from Peru, 48 count",
        "Organic Fuji Apples, 40lb box",
        "Red Bell Peppers, 11lb case, Holland",
        "Jumbo Yellow Onions, 50 lbs",
        "Broccoli Crowns, 20lb case",
        "Seedless Watermelon, each, Mexico",
        "Artisan Lettuce Mix, 3lb bag",
        "Frozen French Fries, 6x5lb bags",
        "Canned Tuna in Oil, 24x6oz cans",
        "Basmati Rice, 50lb bag, India",
        "Whole Milk, 4x1 Gallon Case",
        "Greek Yogurt, Plain, 32oz tub",
        "Frozen Shrimp, Peeled & Deveined, 2lb bag, 21/25 ct",
        "Atlantic Salmon Fillets, Skin-on, 10lb case",
        "Organic Strawberries, 8x1lb flats",
        "Mexican Limes, 40lb case",
        "Organic Gala Apples from New Zealand",
        "Frozen Green Peas, 12x2.5lb bags",
        "Canned Chickpeas, Garbanzo Beans, case of 24",
        "Pure Maple Syrup, Grade A, 1 gallon",
        "Cashews, Roasted, Salted, 10lb tin",
        "Frozen Cod Loins, 10lb box"
    ]

def run_demo():
    """Main function to run the demonstration."""
    print("🚀 Starting iTradenetwork Custom Classification Demo")
    print("-" * 50)

    # 1. Initialize the agent
    print("1. Initializing CustomProductClassificationAgent...")
    agent = CustomProductClassificationAgent(
        model="gpt-4o",
        enable_knowledge_graph=True
    )
    print("✅ Agent initialized.")
    print("-" * 50)

    # 2. Bootstrap from product data
    print("2. Bootstrapping categories from sample product data...")
    products = get_sample_products()
    bootstrap_results = agent.bootstrap_from_product_data(products, force_rebuild=True)
    
    print("\n--- Bootstrap Summary ---")
    pprint.pprint(bootstrap_results)
    print("✅ Bootstrapping complete.")
    print("-" * 50)

    # 3. Classify new products
    print("3. Classifying sample products...")
    products_to_classify = [
        "Fresh Avocados from Mexico, 48ct",
        "Frozen Potato Wedges, Seasoned",
        "Canned Italian Plum Tomatoes",
        "Organic Almonds from California"
    ]

    for product in products_to_classify:
        print(f"\n--- Classifying: '{product}' ---")
        result = agent.classify_product(product)
        
        print(f"  - Primary Category: {result.primary_category} (Confidence: {result.confidence:.2f})")
        print(f"  - Secondary Categories: {result.secondary_categories}")
        print(f"  - Reasoning: {result.reasoning}")
        print(f"  - Patterns Matched: {result.patterns_matched}")
        print(f"  - Similar Products: {result.similar_products}")
        
        print("\n  --- Business Insights ---")
        pprint.pprint(result.business_insights)

    print("\n✅ Classification examples complete.")
    print("-" * 50)

    # 4. Get category system recommendations
    print("4. Getting recommendations for system improvement...")
    recommendations = agent.get_category_recommendations()
    
    print("\n--- System Improvement Recommendations ---")
    if recommendations:
        pprint.pprint(recommendations)
    else:
        print("No specific recommendations at this time.")
    print("✅ Recommendations generated.")
    print("-" * 50)

    # 5. Export categories for review
    print("5. Exporting discovered categories for business review...")
    export_path = agent.export_categories_for_review()
    print(f"✅ Categories exported to: {export_path}")
    print("-" * 50)

    print("🎉 Demo finished successfully!")

if __name__ == "__main__":
    run_demo() 