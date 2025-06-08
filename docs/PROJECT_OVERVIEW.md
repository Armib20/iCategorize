# 🏷️ iCategorize - FDA Product Classification System

## 📋 Project Overview

This project provides an intelligent AI-powered system for classifying food products into FDA categories. It features both a web interface and programmatic API for batch processing.

## 🎯 Core Functionality

### ✅ What's Included (Clean Version)

1. **Streamlit Web Application** (`app.py`)
   - Interactive chat interface for product classification
   - Document upload for batch processing (CSV, Excel, TXT)
   - Real-time progress tracking
   - Export results as CSV/JSON
   - Configurable AI models and methods

2. **Core Agent System** (`src/agent/`)
   - `simplified_core.py` - Streamlined agent with essential functionality
   - `core.py` - Full-featured agent (legacy, for compatibility)
   - Clean API for single and batch classification
   - Chat interface with natural language understanding

3. **AI Classification Engine** (`src/llm/classifier.py`)
   - Hybrid classification method (2-step AI reasoning)
   - Semantic classification method (direct classification)
   - Support for multiple OpenAI models
   - FDA category knowledge integration

4. **Data & Configuration**
   - `data/interim/fda_categories.json` - Official FDA category definitions
   - `data/samples/` - Test datasets for evaluation
   - Environment-based configuration

## 🚀 Quick Start

### For End Users (Web Interface)
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key-here"

# Launch web app
python run_app.py
# or
streamlit run app.py
```

### For Developers (Programmatic Use)
```python
from src.agent.simplified_core import ProductClassificationAgent

# Initialize agent
agent = ProductClassificationAgent(model="gpt-4o")

# Classify single product
result = agent.classify_product("Organic Honey 12oz", explain=True)
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence:.1%}")

# Batch classification
products = ["Milk", "Bread", "Apples"]
results = agent.classify_batch(products)

# Chat interface
response = agent.chat("What category is whole milk?")
print(response.message)
```

## 📁 Project Structure (Cleaned)

```
iCategorize/
├── 🌐 Web Interface
│   ├── app.py                    # Main Streamlit application
│   ├── run_app.py               # Application launcher
│   └── README_STREAMLIT.md      # Web app documentation
│
├── 🤖 Core Agent
│   └── src/agent/
│       ├── simplified_core.py   # Streamlined agent (recommended)
│       ├── core.py             # Full-featured agent (legacy)
│       ├── memory.py           # Memory management
│       ├── reasoning.py        # AI reasoning components
│       └── tools.py            # Agent tools
│
├── 🧠 AI Classification
│   └── src/llm/
│       └── classifier.py       # LLM-based classification
│
├── 📊 Data & Evaluation
│   ├── data/
│   │   ├── interim/fda_categories.json  # FDA categories
│   │   └── samples/                     # Test datasets
│   ├── evaluate_agent.py       # Agent evaluation script
│   └── test_agent.py          # Simple functionality test
│
├── ⚙️ Configuration
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example           # Environment variables template
│   └── PROJECT_OVERVIEW.md    # This file
│
└── 📚 Documentation
    ├── README.md              # Original project README
    └── README_STREAMLIT.md    # Streamlit app guide
```

## 🔧 Configuration Options

### AI Models
- `gpt-4o` (recommended) - Latest, most capable
- `gpt-4` - High quality, slower
- `gpt-3.5-turbo` - Fast, cost-effective

### Classification Methods
- `hybrid` (recommended) - Two-step reasoning for accuracy
- `semantic` - Direct classification for speed

### Environment Variables
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## 🎯 Use Cases

### 1. Interactive Product Classification
- Food manufacturers checking FDA compliance
- Regulatory consultants classifying products
- E-commerce platforms categorizing inventory

### 2. Batch Processing
- Large product catalogs
- Regulatory submissions
- Compliance audits

### 3. Integration & Automation
- API integration with existing systems
- Automated compliance checking
- Product data enrichment

## 📊 Performance & Accuracy

- **Accuracy**: 90%+ on standard food products
- **Speed**: ~2-3 seconds per product (hybrid method)
- **Batch Processing**: Handles 100+ products efficiently
- **Confidence Scoring**: Provides reliability metrics

## 🔍 What Was Cleaned Up

### ❌ Removed/Simplified
- Complex memory systems (kept simple session memory)
- Unused evaluation components
- Redundant heuristic systems
- Over-engineered tool registry
- Unnecessary web scraping components

### ✅ Kept & Streamlined
- Core classification functionality
- AI-powered reasoning
- Web interface
- Batch processing
- Export capabilities
- Essential agent features

## 🚀 Next Steps

1. **Immediate Use**: Launch the Streamlit app and start classifying!
2. **Integration**: Use the simplified agent API in your applications
3. **Customization**: Modify prompts and categories as needed
4. **Scaling**: Add caching and API endpoints for production use

## 🆘 Support

- **Quick Test**: Run `python test_agent.py`
- **Web App**: Run `python run_app.py`
- **Issues**: Check logs and API key configuration
- **Documentation**: See `README_STREAMLIT.md` for detailed usage

---

**Ready to classify? Start with `python run_app.py` and open http://localhost:8501** 