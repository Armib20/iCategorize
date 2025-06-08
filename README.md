# iCategorize AI Agent 🤖

An intelligent AI agent for product classification using FDA categories. This agent goes beyond simple classification to provide a conversational, learning, and extensible system.

## 🌟 Features

### Core Capabilities
- **Intelligent Classification**: Uses hybrid AI + semantic understanding
- **Conversational Interface**: Natural language chat for product classification
- **Learning System**: Learns from user feedback and corrections
- **Batch Processing**: Handle multiple products efficiently
- **Detailed Explanations**: Provides reasoning for classifications
- **Memory & Context**: Remembers conversation history and user preferences

### Interfaces
- **CLI**: Interactive command-line interface
- **Web API**: RESTful API with FastAPI
- **Web UI**: Built-in web interface for easy access

### Advanced Features
- **Export Capabilities**: JSON, CSV export of classifications
- **Performance Analytics**: Track accuracy and usage statistics
- **Tool Integration**: Extensible tool system for external APIs
- **Session Management**: Persistent conversations and learning

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements-agent.txt
```

### 2. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Start the Agent

#### CLI Mode (Interactive Chat)
```bash
python -m src.agent.cli
```

#### Web Interface
```bash
python -m src.agent.web
```
Then visit: http://localhost:8000

#### API Server
```bash
uvicorn src.agent.web:app --reload
```

## 💬 Using the Agent

### CLI Examples

```bash
# Interactive mode
$ python -m src.agent.cli

🤖 Product Classification AI Agent
==================================================
I can help you classify products into FDA categories!
Type 'help' for assistance, 'stats' for session stats, or 'quit' to exit.

👤 You: Classify Organic Honey 12oz

🤖 Agent: I classified 'Organic Honey 12oz' as:

**Category:** Sweeteners
**Confidence:** 92.0%

**Reasoning:** Honey is a natural sweetener derived from bees. It's commonly used as a sugar substitute and falls under the FDA's sweetener category for food products.

💡 Suggestions:
   1. Ask me to classify another product
   2. Tell me if this classification seems wrong
   3. Ask me to explain my reasoning in more detail
```

#### Batch Processing
```bash
python -m src.agent.cli --batch "Milk,Bread,Apples,Honey" --export csv
```

### Web API Examples

#### Classify Single Product
```bash
curl -X POST "http://localhost:8000/api/classify" \
-H "Content-Type: application/json" \
-d '{"product_name": "Organic Honey 12oz", "explain": true}'
```

#### Chat Interface
```bash
curl -X POST "http://localhost:8000/api/chat" \
-H "Content-Type: application/json" \
-d '{"message": "Classify these: Milk, Bread, Apples"}'
```

## 🎯 Agent Capabilities

### 1. Classification Commands
- `"Classify [product name]"` - Single product classification
- `"Classify these: A, B, C"` - Multiple product classification
- `"What category is [product]?"` - Alternative syntax

### 2. Learning & Feedback
- `"That's wrong, it should be [category]"` - Provide corrections
- `"Actually, [product] is [category]"` - Specific feedback
- Agent learns from corrections and improves over time

### 3. Explanation & Analysis
- `"Why did you choose that category?"` - Get reasoning
- `"Explain [product] classification"` - Detailed analysis
- `"Show me alternatives for [product]"` - See other options

### 4. Data Management
- `"Export my results"` - Export classifications
- `"Generate a report"` - Create comprehensive report
- `"Show my statistics"` - View session stats

## 🧠 AI Agent Architecture

### Core Components

```
src/agent/
├── core.py          # Main agent orchestrator
├── memory.py        # Conversation & learning memory
├── reasoning.py     # Intent analysis & explanation
├── tools.py         # External integrations & exports
├── cli.py           # Command-line interface
└── web.py           # Web API & interface
```

### Agent Flow

1. **Input Processing**: User input → Intent analysis
2. **Reasoning**: Understand context → Plan response
3. **Tool Usage**: Classification → External tools
4. **Memory Update**: Store results → Learn from feedback
5. **Response Generation**: Format response → Provide suggestions

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your-api-key

# Optional
OPENAI_MODEL=gpt-4o              # Default model
AGENT_MEMORY_SIZE=100            # Conversation history size
AGENT_ENABLE_LEARNING=true       # Enable learning from feedback
```

### Customization

#### Custom Models
```python
agent = ProductClassificationAgent(
    model="gpt-4-turbo",           # Use different model
    memory_size=200,               # Larger memory
    enable_learning=True           # Enable learning
)
```

#### Adding Tools
```python
# Extend the ToolRegistry
class CustomToolRegistry(ToolRegistry):
    def my_custom_tool(self, data):
        # Your custom integration
        return result

agent.tools = CustomToolRegistry()
```

## 📊 Performance & Analytics

### Session Statistics
- Total classifications performed
- Accuracy based on user corrections
- Most common categories used
- Average confidence scores
- User feedback patterns

### Learning Insights
- Correction patterns
- User preferences
- Category confusion matrix
- Performance improvements over time

## 🔗 Integrations

### Current Tools
- **Export**: JSON, CSV export of results
- **Validation**: Product name validation and cleaning
- **Batch Processing**: File upload and processing
- **Reporting**: Comprehensive classification reports

### Extensible Integration Points
- **E-commerce**: Shopify, WooCommerce integrations
- **Databases**: Product catalog integration
- **APIs**: Barcode lookup, product information
- **ERP Systems**: Inventory management integration

## 🛠️ Development

### Running Tests
```bash
python -m pytest src/agent/tests/
```

### Adding New Capabilities

1. **Create Tool**: Add to `tools.py`
2. **Update Reasoning**: Modify intent analysis in `reasoning.py`
3. **Extend Agent**: Add new methods to `core.py`
4. **Update Interface**: Add CLI commands or API endpoints

### Agent Architecture Principles

- **Modularity**: Each component has clear responsibilities
- **Extensibility**: Easy to add new tools and capabilities
- **Memory**: Persistent learning and context awareness
- **Reasoning**: AI-driven intent understanding and planning
- **Interfaces**: Multiple ways to interact (CLI, Web, API)

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements-agent.txt
CMD ["uvicorn", "src.agent.web:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Scaling Considerations
- Use Redis for session management
- Implement proper database for memory persistence
- Add authentication and rate limiting
- Monitor API usage and costs

## 📈 Future Enhancements

- **Multi-modal**: Support images and product descriptions
- **Custom Categories**: Allow users to define custom classification schemes
- **Advanced Learning**: Implement reinforcement learning from user feedback
- **Integration Hub**: Pre-built connectors for popular e-commerce platforms
- **Analytics Dashboard**: Real-time performance and usage analytics 