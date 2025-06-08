# 🏷️ FDA Product Classification Assistant

An intelligent AI-powered web application for classifying food products into FDA categories with high accuracy and detailed explanations.

## ✨ Features

- **💬 Interactive Chat Interface**: Natural language product classification
- **📄 Document Upload**: Batch processing of CSV, Excel, and text files
- **🎯 High Accuracy**: AI-powered classification with confidence scores
- **📊 Detailed Explanations**: Understand why products are classified
- **📈 Export Results**: Download classifications as CSV or JSON
- **🔄 Real-time Processing**: Live progress tracking for batch operations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- OpenAI API key

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd iCategorize
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Launch the application:**
   ```bash
   python run_app.py
   ```
   
   Or directly with streamlit:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 🎮 How to Use

### Chat Interface

Simply type natural language requests:
- "Classify Organic Honey 12oz"
- "What category is whole milk?"
- "Categorize this product: Fresh Blueberries"

### Document Upload

1. Upload a CSV, Excel, or text file with product names
2. Select the column containing product names (for CSV/Excel)
3. Configure classification options
4. Click "Classify Products" and wait for results
5. Download results as CSV or JSON

### Supported File Formats

- **CSV**: Must have a column with product names
- **Excel (.xlsx)**: Must have a column with product names  
- **Text (.txt)**: One product name per line

## 🔧 Configuration

- **AI Model**: Choose between GPT-4, GPT-4o, or GPT-3.5-turbo
- **Classification Method**: 
  - **Hybrid**: Two-step AI reasoning (recommended)
  - **Semantic**: Direct classification (faster)
- **Batch Processing**: Limit number of products to process
- **Explanations**: Toggle detailed reasoning on/off

## 📊 Features in Detail

### Chat Assistant
- Natural language understanding
- Context-aware responses
- Detailed classification explanations
- Alternative category suggestions

### Batch Processing  
- Progress tracking
- Error handling
- Configurable limits
- Multiple export formats

### Results Display
- Clean tabular format
- Confidence scores
- Reasoning summaries
- Timestamp tracking
- Export functionality

## 🎯 Accuracy

The system uses advanced AI models for classification with:
- **Hybrid approach**: 2-step reasoning for higher accuracy
- **Semantic understanding**: Goes beyond keyword matching
- **FDA category knowledge**: Trained on official FDA categories
- **Confidence scoring**: Know how certain the classification is

## 🔍 Troubleshooting

### Common Issues

**"OpenAI API key not set"**
- Make sure your API key is exported as an environment variable
- Check your `.env` file if using one

**"No products found in file"**
- Ensure your CSV/Excel has a column with product names
- Check that product names aren't empty

**"Error processing file"**
- Verify file format (CSV, XLSX, TXT)
- Check file encoding (should be UTF-8)

### Performance Tips

- Use batch processing for large datasets
- Disable detailed reasoning for faster processing
- Limit concurrent requests for large files

## 📁 Project Structure

```
iCategorize/
├── app.py                    # Main Streamlit application
├── run_app.py               # Application launcher
├── requirements.txt         # Python dependencies
├── src/
│   ├── agent/              # Core agent functionality
│   │   ├── simplified_core.py  # Streamlined agent
│   │   └── core.py         # Full-featured agent
│   └── llm/                # AI classification logic
│       └── classifier.py   # LLM-based classification
└── data/
    └── interim/
        └── fda_categories.json  # FDA category definitions
```

## 🚀 Next Steps

- Add more file format support
- Implement classification caching
- Add user feedback collection
- Create classification templates
- Add batch API endpoint

## 📝 License

This project is licensed under the MIT License.

---

**Need help?** Open an issue on GitHub or contact the development team. 