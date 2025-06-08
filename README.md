# 🏷️ iCategorize - FDA Product Classification System

> **Clean, Organized, Production-Ready** 

An intelligent AI-powered system for classifying food products into FDA categories with a modern web interface and clean API.

## ✨ Features

- **🌐 Web Interface**: Interactive Streamlit app with chat and batch processing
- **🤖 AI-Powered**: Uses GPT models for accurate product classification  
- **📄 Batch Processing**: Upload CSV, Excel, or text files for bulk classification
- **📊 Export Results**: Download classifications as CSV or JSON
- **🎯 High Accuracy**: 90%+ accuracy with confidence scoring
- **🔧 Configurable**: Multiple AI models and classification methods

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r config/requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# Or copy config/env_example.txt to .env and edit it
```

### 2. Launch Web App
```bash
python run.py
```
Then open http://localhost:8501 in your browser.

### 3. Test Core API
```bash
python tests/test_agent.py
```

## 📁 Project Structure

```
iCategorize/
├── 🌐 app/                    # Web Interface
│   ├── app.py                 # Main Streamlit application
│   └── run_app.py            # App-specific launcher
│
├── 🤖 core/                   # Core Components  
│   ├── agent.py              # Main classification agent
│   ├── classifier.py         # LLM-based classification
│   └── __init__.py           # Module initialization
│
├── 📊 data/                   # Data & Categories
│   ├── fda_categories.json   # Official FDA categories
│   ├── samples/              # Test datasets
│   └── interim/              # Processing artifacts
│
├── 🔧 config/                 # Configuration
│   ├── requirements.txt      # Python dependencies
│   └── env_example.txt       # Environment variables template
│
├── 📚 docs/                   # Documentation
│   ├── README_STREAMLIT.md   # Web app guide
│   └── PROJECT_OVERVIEW.md   # Detailed project info
│
├── 🧪 tests/                  # Testing
│   └── test_agent.py         # Core functionality tests
│
├── 📦 legacy/                 # Legacy Code (kept for reference)
│   ├── src/                  # Original source structure
│   ├── notebooks/            # Jupyter notebooks
│   └── evaluate_agent.py     # Original evaluation script
│
└── 🚀 run.py                  # Main launcher script
```

## 🎮 Usage

### Web Interface
1. **Chat Mode**: Type natural language requests
   - "Classify Organic Honey 12oz"
   - "What category is whole milk?"

2. **Document Upload**: Process files in bulk
   - Upload CSV, Excel, or text files
   - Select product name column
   - Configure processing options
   - Download results

### Programmatic API
```python
from core import ProductClassificationAgent

# Initialize agent
agent = ProductClassificationAgent(model="gpt-4o")

# Single product
result = agent.classify_product("Organic Honey 12oz", explain=True)
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence:.1%}")

# Batch processing
products = ["Whole Milk", "Sourdough Bread", "Fresh Apples"]
results = agent.classify_batch(products)

# Chat interface
response = agent.chat("What category is cheddar cheese?")
print(response.message)
```

## 🔧 Configuration

### AI Models
- `gpt-4o` (default) - Latest, most capable
- `gpt-4` - High quality, reliable
- `gpt-3.5-turbo` - Fast, cost-effective

### Classification Methods
- `hybrid` (default) - Two-step reasoning for accuracy
- `semantic` - Direct classification for speed

## 📊 Performance

- **Accuracy**: 90%+ on standard food products
- **Speed**: ~2-3 seconds per product (hybrid method)
- **Batch Processing**: Handles 100+ products efficiently
- **Confidence Scoring**: Reliability metrics for each classification

## 🔍 What's New (Reorganized)

### ✅ Improvements
- **Clean Structure**: Logical organization by function
- **Easy Imports**: Simple, predictable import paths
- **Focused Components**: Only essential functionality
- **Better Documentation**: Clear guides and examples
- **Production Ready**: Proper configuration management

### 🗂️ Organization Benefits
- **app/**: All web interface code in one place
- **core/**: Clean API for classification functionality  
- **config/**: Centralized configuration management
- **docs/**: All documentation together
- **tests/**: Testing infrastructure
- **legacy/**: Original code preserved but out of the way

## 🆘 Troubleshooting

### Common Issues
```bash
# Missing API key
export OPENAI_API_KEY="your-key-here"

# Missing dependencies  
pip install -r config/requirements.txt

# Import errors
python tests/test_agent.py  # Test core functionality
```

### File Paths
- The reorganized structure uses relative imports
- All paths are relative to the project root
- FDA categories: `data/fda_categories.json`
- Main launcher: `python run.py`

## 🚀 Development

### Adding Features
1. Core functionality → `core/`
2. Web interface → `app/`
3. Tests → `tests/`
4. Documentation → `docs/`

### Running Tests
```bash
python tests/test_agent.py          # Test core agent
python app/run_app.py               # Test web app (local)
```

## 📝 Migration Notes

If you're upgrading from the old structure:
- Main launcher is now `python run.py` 
- Requirements moved to `config/requirements.txt`
- Core agent is `from core import ProductClassificationAgent`
- Legacy code preserved in `legacy/` folder

---

**Ready to classify? Run `python run.py` and start classifying products! 🎯** 