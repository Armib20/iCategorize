"""
Debug script to test category discovery with small dataset
"""

from icategorize.custom_classifier.category_discovery import CategoryDiscoveryEngine

# Test products
test_products = [
    "Fresh California Apples",
    "Organic Florida Oranges", 
    "Fresh Local Bananas",
    "Frozen Mixed Berries",
    "California Premium Strawberries",
    "Fresh Iceberg Lettuce",
    "Organic Baby Spinach",
    "Frozen Green Peas",  
    "Fresh Bell Peppers",
    "Organic Tomatoes",
    "Premium Chicken Breast",
    "Organic Ground Beef",
    "Fresh Atlantic Salmon",
    "Whole Milk Gallon",
    "Organic Greek Yogurt",
    "Sharp Cheddar Cheese",
    "Free Range Eggs",
    "Artisanal Sourdough Bread",
    "Organic Olive Oil",
    "Premium Sea Salt",
    "California Almonds",
    "Organic Honey",
    "Fresh Basil Leaves",
    "Dried Pasta",
    "Canned Tomato Sauce"
]

def debug_category_discovery():
    print("🔍 Starting category discovery debug test...")
    
    # Create discovery engine
    engine = CategoryDiscoveryEngine()
    
    # Test preprocessing
    print("\n1. Testing product preprocessing...")
    processed = engine._preprocess_products(test_products[:5])
    for p in processed[:2]:
        print(f"   Product: {p['original']}")
        print(f"   Clean: {p['clean']}")
        print(f"   Origin: {p['origin']}")
        print(f"   Processing: {p['processing']}")
        print(f"   Main ingredient: {p['main_ingredient']}")
        print()
    
    # Test pattern discovery
    print("2. Testing pattern discovery...")
    all_processed = engine._preprocess_products(test_products)
    patterns = engine._discover_patterns(all_processed)
    print(f"   Found {len(patterns)} patterns:")
    for pattern in patterns[:5]:
        print(f"   - {pattern.pattern_type}: {pattern.values} (frequency: {pattern.frequency})")
    
    # Test clustering
    print("\n3. Testing clustering...")
    try:
        clusters = engine._perform_semantic_clustering(all_processed, min_cluster_size=2)
        print(f"   Found {len(clusters)} clusters:")
        for i, cluster in enumerate(clusters):
            print(f"   Cluster {i+1}: {len(cluster)} products")
            for product in cluster[:3]:
                print(f"     - {product}")
            if len(cluster) > 3:
                print(f"     ... and {len(cluster)-3} more")
            print()
    except Exception as e:
        print(f"   Clustering failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test full analysis
    print("4. Testing full analysis...")
    try:
        categories = engine.analyze_product_data(test_products, min_cluster_size=2)
        print(f"   Discovered {len(categories)} categories:")
        for cat in categories:
            print(f"   - {cat.name}: {cat.size} products (confidence: {cat.confidence:.2f})")
            print(f"     Description: {cat.description}")
            print(f"     Keywords: {', '.join(cat.keywords[:3])}")
            print()
    except Exception as e:
        print(f"   Full analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_category_discovery() 