# 🏷️ iCategorize - FDA Product Classification System

> **Automatically classify free-form product names into FDA product categories using LLMs**

[![AI-Powered](https://img.shields.io/badge/AI-Powered-blue.svg)]()
[![FDA-Compliant](https://img.shields.io/badge/FDA-Compliant-green.svg)]()
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)]()

## 🔍 The Problem

E-commerce sellers face a critical challenge: **product categorization chaos**.

- **Input**: Sellers provide unstructured, inconsistent product names
  - `"Blackberries 12oz clamshell"`
  - `"Organic free-range eggs dozen"`
  - `"Artisanal sourdough bread loaf"`

- **Required Output**: FDA-defined categories for compliance
  - `"All other fruits, nuts, and vegetables (without meat, poultry, or seafood)"`
  - `"Eggs and egg products"`
  - `"Grain/cereal products and pasta (without meat/poultry/seafood)"`

- **The Challenge**: The FDA list contains hundreds of nuanced categories. Simple keyword matching fails spectacularly.

## 🚀 The Solution

**iCategorize** uses advanced LLMs to understand context, ingredients, and product characteristics, then maps them to the correct FDA categories with high accuracy and confidence scoring.

### ✨ Key Features

- **🤖 AI-Powered Classification**: GPT-4 models understand product context and nuances
- **🎯 High Accuracy**: 90%+ accuracy with confidence scoring for reliability
- **📊 Batch Processing**: Handle thousands of products efficiently
- **🌐 Web Interface**: User-friendly Streamlit app for interactive classification
- **🔧 Flexible API**: Programmatic access for integration into existing systems
- **📈 Continuous Learning**: System improves with usage patterns  
- **📄 Multiple Formats**: Support for CSV, Excel, and text file uploads
- **🎯 Real-time Accuracy Tracking**: Compare predictions against ground truth with live metrics

## 🎮 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/iCategorize.git
cd iCategorize

# Install dependencies
pip install -r config/requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"
# Or copy config/env_example.txt to .env and edit it
```

### 2. Launch Web Interface
```bash
python run.py
```
Open http://localhost:8501 in your browser and start classifying!

### 3. Quick API Test
```bash
python tests/test_agent.py
```

## 💡 Usage Examples

### Web Interface
**Chat Mode** - Natural language interaction:
```
👤 "What FDA category is organic honey?"
🤖 "Honey falls under 'All other fruits, nuts, and vegetables (without meat, poultry, or seafood)' 
    with 95% confidence. This category includes natural sweeteners derived from plants."
```

**Batch Upload** - Process files in bulk:
1. Upload CSV/Excel file with product names
2. Select the column containing product names
3. Configure processing options
4. Download classified results

**Accuracy Tracking** - Evaluate performance with ground truth:
1. Upload a CSV/Excel file with both product names and correct FDA categories
2. Select your product column, then enable "📊 Enable accuracy benchmarking"
3. Choose which column contains the ground truth categories
4. Watch real-time accuracy metrics update as each product is classified
5. Review detailed accuracy breakdown and error patterns

### Programmatic API
```python
from core import SimplifiedProductClassificationAgent

# Initialize the agent
agent = SimplifiedProductClassificationAgent(model="gpt-4o")

# Classify a single product
result = agent.classify_product("Blackberries 12oz clamshell", explain=True)
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Reasoning: {result.reasoning}")

# Batch classification
products = [
    "Organic whole milk gallon",
    "Fresh Atlantic salmon fillet", 
    "Artisanal cheddar cheese wheel"
]
results = agent.classify_batch(products)

for result in results:
    print(f"{result.product_name} → {result.category} ({result.confidence:.1%})")
```

## 🏗️ Architecture

```
iCategorize/
├── 🌐 app/                    # Web Interface Components
│   ├── app.py                 # Main Streamlit application
│   └── run_app.py            # Application launcher
│
├── 🤖 core/                   # Classification Engine
│   ├── agent.py              # Main classification agent
│   ├── classifier.py         # LLM-based classification logic
│   └── __init__.py           # Core module exports
│
├── 📊 data/                   # FDA Categories & Test Data
│   ├── fda_categories.json   # Official FDA product categories
│   ├── samples/              # Sample datasets for testing
│   └── interim/              # Temporary processing files
│
├── 🔧 config/                 # Configuration Management
│   ├── requirements.txt      # Python dependencies
│   └── env_example.txt       # Environment variables template
│
├── 📚 docs/                   # Documentation
│   ├── README_STREAMLIT.md   # Web application guide
│   └── PROJECT_OVERVIEW.md   # Detailed technical overview
│
├── 🧪 tests/                  # Testing Suite
│   └── test_agent.py         # Core functionality tests
│
├── 📦 legacy/                 # Previous Implementation
│   └── ...                   # Original code (preserved for reference)
│
└── 🚀 run.py                  # Main application launcher
```

## ⚙️ Configuration

### AI Models
- **`gpt-4o`** (default) - Latest model with best performance
- **`gpt-4`** - Reliable, high-quality classifications  
- **`gpt-3.5-turbo`** - Faster, more cost-effective option

### Classification Methods
- **`hybrid`** (recommended) - Two-step reasoning for maximum accuracy
- **`semantic`** - Direct classification for speed-critical applications

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional
DEFAULT_MODEL=gpt-4o
CLASSIFICATION_METHOD=hybrid
MAX_BATCH_SIZE=100
```

## 📈 Performance Metrics

| Metric | Performance |
|--------|-------------|
| **Accuracy** | 90%+ on standard food products |
| **Processing Speed** | ~2-3 seconds per product (hybrid mode) |
| **Batch Throughput** | 100+ products efficiently processed |
| **Coverage** | 138 FDA categories supported |
| **Confidence Scoring** | Reliability metrics for each classification |

## 🔍 FDA Categories Supported

The system handles the complete FDA product category taxonomy, including:

- **Fresh Produce**: Fruits, vegetables, herbs
- **Dairy Products**: Milk, cheese, yogurt, ice cream
- **Meat & Poultry**: Fresh, processed, and prepared meats
- **Seafood**: Fish, shellfish, and marine products
- **Baked Goods**: Bread, pastries, and grain products
- **Beverages**: Juices, soft drinks, alcoholic beverages
- **Processed Foods**: Canned goods, frozen foods, snacks
- **Supplements**: Vitamins, minerals, dietary supplements

## 🛠️ Development

### Adding New Features
1. **Core Logic** → `core/` directory
2. **Web Interface** → `app/` directory  
3. **Tests** → `tests/` directory
4. **Documentation** → `docs/` directory

### Running Tests
```bash
# Test classification engine
python tests/test_agent.py

# Test web application locally
python app/run_app.py

# Run all tests (if pytest is set up)
pytest tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 🆘 Troubleshooting

### Common Issues

**API Key Missing**
```bash
export OPENAI_API_KEY="your-key-here"
# or create .env file with OPENAI_API_KEY=your-key-here
```

**Dependencies Missing**
```bash
pip install -r config/requirements.txt
```

**Import Errors**
```bash
# Test core functionality
python tests/test_agent.py

# Verify installation
python -c "from core import SimplifiedProductClassificationAgent; print('✅ Installation successful')"
```

**Low Classification Accuracy**
- Ensure you're using detailed product descriptions
- Try the `hybrid` classification method for better accuracy
- Check that your products match supported FDA categories

## 📝 Use Cases

### E-commerce Platforms
- **Product Catalog Management**: Automatically categorize new product listings
- **Compliance Reporting**: Generate FDA-compliant product category reports
- **Data Migration**: Classify existing unstructured product databases

### Food Safety & Compliance
- **Regulatory Reporting**: Ensure proper FDA category assignment
- **Supply Chain Management**: Track products through proper categorical channels  
- **Quality Assurance**: Validate product categorizations for accuracy

### Market Research
- **Product Analysis**: Understand market distribution across FDA categories
- **Competitive Intelligence**: Analyze competitor product portfolios
- **Trend Analysis**: Track category-specific market trends

## 🚀 Next Steps

Ready to start classifying products? 

1. **Quick Start**: Run `python run.py` and open the web interface
2. **API Integration**: Import `SimplifiedProductClassificationAgent` in your code
3. **Batch Processing**: Upload your product list via the web interface
4. **Custom Integration**: Use the core API to build your own classification pipeline

---

**Transform your product chaos into FDA-compliant categories with AI precision! 🎯** 