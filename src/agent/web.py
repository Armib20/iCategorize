"""
Web API interface for the Product Classification AI Agent.

Provides REST endpoints for classification and chat functionality.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import json

from .core import ProductClassificationAgent, AgentResponse


# Pydantic models for API
class ClassifyRequest(BaseModel):
    product_name: str
    explain: bool = True
    method: str = "hybrid"


class BatchClassifyRequest(BaseModel):
    product_names: List[str]
    explain: bool = False
    method: str = "hybrid"


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    product_name: str
    wrong_category: str
    correct_category: str
    reasoning: Optional[str] = ""


# Global agent instances (in production, use proper session management)
agents: Dict[str, ProductClassificationAgent] = {}


# FastAPI app
app = FastAPI(
    title="Product Classification AI Agent",
    description="AI-powered product classification system using FDA categories",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_or_create_agent(session_id: str = None) -> tuple[str, ProductClassificationAgent]:
    """Get existing agent or create new one."""
    if session_id and session_id in agents:
        return session_id, agents[session_id]
    
    # Create new agent
    new_session_id = session_id or str(uuid.uuid4())
    agents[new_session_id] = ProductClassificationAgent()
    return new_session_id, agents[new_session_id]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Product Classification AI Agent</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            .container {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            }
            .chat-container {
                max-height: 500px;
                overflow-y: auto;
                background: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background: #007bff;
                color: white;
                margin-left: 20%;
            }
            .agent-message {
                background: #e9ecef;
                margin-right: 20%;
            }
            input, textarea, button {
                width: 100%;
                padding: 10px;
                margin: 5px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            button {
                background: #007bff;
                color: white;
                border: none;
                cursor: pointer;
            }
            button:hover {
                background: #0056b3;
            }
            .examples {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 5px;
                padding: 15px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h1>🤖 Product Classification AI Agent</h1>
        
        <div class="container">
            <h2>Quick Classification</h2>
            <input type="text" id="productInput" placeholder="Enter product name (e.g., 'Organic Honey 12oz')" />
            <button onclick="classifyProduct()">Classify Product</button>
            <div id="classificationResult"></div>
        </div>

        <div class="container">
            <h2>Chat Interface</h2>
            <div class="chat-container" id="chatContainer"></div>
            <textarea id="chatInput" placeholder="Type your message here..." rows="3"></textarea>
            <button onclick="sendMessage()">Send Message</button>
        </div>

        <div class="examples">
            <h3>💡 Try these examples:</h3>
            <ul>
                <li>"Classify Organic Honey 12oz"</li>
                <li>"Classify these: Milk, Bread, Apples, Cheese"</li>
                <li>"Why did you choose that category?"</li>
                <li>"Export my results as CSV"</li>
            </ul>
        </div>

        <script>
            let sessionId = null;

            async function classifyProduct() {
                const productName = document.getElementById('productInput').value.trim();
                if (!productName) return;

                const resultDiv = document.getElementById('classificationResult');
                resultDiv.innerHTML = '<p>Classifying...</p>';

                try {
                    const response = await fetch('/api/classify', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            product_name: productName,
                            explain: true
                        })
                    });
                    
                    const result = await response.json();
                    
                    resultDiv.innerHTML = `
                        <h4>Result:</h4>
                        <p><strong>Category:</strong> ${result.category}</p>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Reasoning:</strong> ${result.reasoning}</p>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = '<p style="color: red;">Error classifying product</p>';
                }
            }

            async function sendMessage() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                if (!message) return;

                // Add user message to chat
                addMessage(message, 'user');
                input.value = '';

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId
                        })
                    });
                    
                    const result = await response.json();
                    sessionId = result.session_id;
                    
                    // Add agent response to chat
                    addMessage(result.response.message, 'agent');
                    
                } catch (error) {
                    addMessage('Error: Could not get response from agent', 'agent');
                }
            }

            function addMessage(text, sender) {
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.textContent = text;
                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
            }

            // Allow Enter to send message
            document.getElementById('chatInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // Allow Enter to classify product
            document.getElementById('productInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    classifyProduct();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/classify")
async def classify_product(request: ClassifyRequest):
    """Classify a single product."""
    try:
        _, agent = get_or_create_agent()
        result = agent.classify_product(
            request.product_name,
            explain=request.explain,
            method=request.method
        )
        
        return {
            "product_name": result.product_name,
            "category": result.category,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "alternatives": result.alternatives,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify/batch")
async def classify_batch(request: BatchClassifyRequest):
    """Classify multiple products."""
    try:
        _, agent = get_or_create_agent()
        results = agent.classify_batch(
            request.product_names,
            explain=request.explain,
            method=request.method
        )
        
        return {
            "total": len(results),
            "results": [
                {
                    "product_name": r.product_name,
                    "category": r.category,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning if request.explain else "",
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest):
    """Chat with the agent."""
    try:
        session_id, agent = get_or_create_agent(request.session_id)
        response = agent.chat(request.message)
        
        return {
            "session_id": session_id,
            "response": {
                "message": response.message,
                "results": [
                    {
                        "product_name": r.product_name,
                        "category": r.category,
                        "confidence": r.confidence
                    }
                    for r in response.results
                ],
                "suggestions": response.suggestions,
                "needs_clarification": response.needs_clarification,
                "clarification_question": response.clarification_question
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def provide_feedback(request: FeedbackRequest):
    """Provide feedback to improve the agent."""
    try:
        _, agent = get_or_create_agent()
        
        correction = {
            "product_name": request.product_name,
            "wrong_category": request.wrong_category,
            "correct_category": request.correct_category,
            "reasoning": request.reasoning
        }
        
        agent._learn_from_feedback(correction)
        
        return {"message": "Feedback received. Thank you for helping improve the agent!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/{session_id}")
async def get_session_stats(session_id: str):
    """Get statistics for a session."""
    if session_id not in agents:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = agents[session_id]
    return agent.get_stats()


@app.get("/api/export/{session_id}")
async def export_session_data(session_id: str, format: str = "json"):
    """Export session classification data."""
    if session_id not in agents:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = agents[session_id]
    
    try:
        exported_data = agent.export_results(format)
        
        if format == "csv":
            return JSONResponse(
                content=exported_data,
                headers={"Content-Type": "text/csv"}
            )
        else:
            return JSONResponse(content=json.loads(exported_data))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_batch_file(file: UploadFile = File(...)):
    """Upload a file for batch processing."""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        _, agent = get_or_create_agent()
        
        # Determine file format
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'txt'
        
        # Extract product names
        product_names = agent.tools.batch_process_file(content_str, file_extension)
        
        if not product_names:
            raise HTTPException(status_code=400, detail="No valid product names found in file")
        
        # Classify all products
        results = agent.classify_batch(product_names, explain=False)
        
        return {
            "message": f"Processed {len(results)} products from {file.filename}",
            "results": [
                {
                    "product_name": r.product_name,
                    "category": r.category,
                    "confidence": r.confidence
                }
                for r in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 