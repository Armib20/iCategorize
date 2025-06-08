import streamlit as st
import pandas as pd
import io
from typing import List, Dict, Any
import os
from datetime import datetime
import json

# Import agent functionality
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ProductClassificationAgent, ClassificationResult

# Configure page
st.set_page_config(
    page_title="FDA Product Classification Assistant",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = ProductClassificationAgent(
        model="gpt-4o",
        enable_learning=True
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []

def classify_products_from_data(products: List[str]) -> List[ClassificationResult]:
    """Classify a list of products and return results."""
    if not products:
        return []
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, product in enumerate(products):
        status_text.text(f"Classifying: {product}")
        result = st.session_state.agent.classify_product(product, explain=True)
        results.append(result)
        progress_bar.progress((i + 1) / len(products))
    
    status_text.text("Classification complete!")
    progress_bar.empty()
    return results

def display_classification_results(results: List[ClassificationResult]):
    """Display classification results in a neat table format."""
    if not results:
        return
    
    # Create DataFrame for display
    df_data = []
    for result in results:
        df_data.append({
            'Product Name': result.product_name,
            'FDA Category': result.category,
            'Confidence': f"{result.confidence:.1%}",
            'Reasoning': result.reasoning[:100] + "..." if len(result.reasoning) > 100 else result.reasoning,
            'Alternatives': ", ".join(result.alternatives[:3]) if result.alternatives else "None",
            'Timestamp': result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    df = pd.DataFrame(df_data)
    
    # Display results
    st.subheader("Classification Results")
    st.dataframe(df, use_container_width=True)
    
    # Export functionality
    col1, col2 = st.columns(2)
    
    with col1:
        # Download as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"fda_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download as JSON
        json_data = [
            {
                'product_name': r.product_name,
                'category': r.category,
                'confidence': r.confidence,
                'reasoning': r.reasoning,
                'alternatives': r.alternatives,
                'timestamp': r.timestamp.isoformat()
            }
            for r in results
        ]
        json_str = json.dumps(json_data, indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name=f"fda_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    # Header
    st.title("🏷️ FDA Product Classification Assistant")
    st.markdown("**Intelligent AI-powered product classification for FDA compliance**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "AI Model",
            ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        
        if model_choice != st.session_state.agent.model:
            st.session_state.agent = ProductClassificationAgent(
                model=model_choice,
                enable_learning=True
            )
        
        # Classification method
        classification_method = st.selectbox(
            "Classification Method",
            ["hybrid", "semantic"],
            index=0,
            help="Hybrid uses two-step AI reasoning, semantic uses direct classification"
        )
        
        st.divider()
        
        # Statistics
        st.header("📊 Session Stats")
        stats = st.session_state.agent.get_stats()
        st.metric("Products Classified", len(st.session_state.classification_results))
        st.metric("Chat Messages", len(st.session_state.chat_history))
        
        # Clear session
        if st.button("Clear Session", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.classification_results = []
            st.session_state.agent = ProductClassificationAgent(
                model=model_choice,
                enable_learning=True
            )
            st.rerun()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📄 Document Upload"])
    
    with tab1:
        st.header("Chat with the FDA Classification Assistant")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.chat_message("user").write(message['content'])
                else:
                    st.chat_message("assistant").write(message['content'])
                    
                    # Display classification results if any
                    if 'results' in message and message['results']:
                        with st.expander("View Classification Details"):
                            for result in message['results']:
                                st.json({
                                    'product': result.product_name,
                                    'category': result.category,
                                    'confidence': f"{result.confidence:.1%}",
                                    'reasoning': result.reasoning,
                                    'alternatives': result.alternatives
                                })
        
        # Chat input
        if prompt := st.chat_input("Ask me to classify products or ask questions..."):
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            
            # Get agent response
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat(prompt)
            
            # Add assistant response to history
            assistant_message = {
                'role': 'assistant', 
                'content': response.message,
                'results': response.results
            }
            st.session_state.chat_history.append(assistant_message)
            
            # Store results for session tracking
            st.session_state.classification_results.extend(response.results)
            
            st.rerun()
    
    with tab2:
        st.header("Upload Documents for Batch Classification")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'txt', 'xlsx'],
            help="Upload a CSV, TXT, or Excel file containing product names"
        )
        
        if uploaded_file is not None:
            # Process uploaded file
            try:
                products = []
                
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    st.write("**File Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Column selection
                    if len(df.columns) > 1:
                        product_column = st.selectbox(
                            "Select the column containing product names:",
                            df.columns.tolist()
                        )
                        products = df[product_column].dropna().tolist()
                    else:
                        products = df.iloc[:, 0].dropna().tolist()
                
                elif uploaded_file.name.endswith('.txt'):
                    content = str(uploaded_file.read(), "utf-8")
                    products = [line.strip() for line in content.split('\n') if line.strip()]
                    st.write(f"**Found {len(products)} products in text file**")
                    with st.expander("Preview products"):
                        st.write(products[:10])  # Show first 10
                
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                    st.write("**File Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Column selection
                    if len(df.columns) > 1:
                        product_column = st.selectbox(
                            "Select the column containing product names:",
                            df.columns.tolist()
                        )
                        products = df[product_column].dropna().tolist()
                    else:
                        products = df.iloc[:, 0].dropna().tolist()
                
                # Classification options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_products = st.number_input(
                        "Max products to process",
                        min_value=1,
                        max_value=len(products) if products else 100,
                        value=min(50, len(products) if products else 50),
                        help="Limit processing for large files"
                    )
                
                with col2:
                    include_reasoning = st.checkbox(
                        "Include detailed reasoning",
                        value=True,
                        help="Provides explanations but takes longer"
                    )
                
                with col3:
                    auto_download = st.checkbox(
                        "Auto-download results",
                        value=False,
                        help="Automatically download CSV after processing"
                    )
                
                # Process button
                if st.button("🚀 Classify Products", type="primary"):
                    if products:
                        limited_products = products[:max_products]
                        
                        # Run classification
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, product in enumerate(limited_products):
                            status_text.text(f"Classifying: {product}")
                            result = st.session_state.agent.classify_product(
                                product, 
                                explain=include_reasoning,
                                method=classification_method
                            )
                            results.append(result)
                            progress_bar.progress((i + 1) / len(limited_products))
                        
                        status_text.text("✅ Classification complete!")
                        progress_bar.empty()
                        
                        # Store results
                        st.session_state.classification_results.extend(results)
                        
                        # Display results
                        display_classification_results(results)
                        
                        # Auto-download if enabled
                        if auto_download:
                            df_data = []
                            for result in results:
                                df_data.append({
                                    'product_name': result.product_name,
                                    'fda_category': result.category,
                                    'confidence': result.confidence,
                                    'reasoning': result.reasoning if include_reasoning else "",
                                    'alternatives': ", ".join(result.alternatives) if result.alternatives else "",
                                    'timestamp': result.timestamp.isoformat()
                                })
                            
                            df_download = pd.DataFrame(df_data)
                            csv = df_download.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"fda_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="auto_download"
                            )
                    else:
                        st.error("No products found in the uploaded file.")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Manual input option
        st.divider()
        st.subheader("Manual Product Entry")
        
        manual_products = st.text_area(
            "Enter product names (one per line):",
            height=150,
            placeholder="Organic Honey 12oz\nWhole Milk 1 Gallon\nFresh Blueberries 6oz\n..."
        )
        
        if st.button("Classify Manual Entries") and manual_products:
            products = [line.strip() for line in manual_products.split('\n') if line.strip()]
            
            if products:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, product in enumerate(products):
                    status_text.text(f"Classifying: {product}")
                    result = st.session_state.agent.classify_product(
                        product, 
                        explain=True,
                        method=classification_method
                    )
                    results.append(result)
                    progress_bar.progress((i + 1) / len(products))
                
                status_text.text("✅ Classification complete!")
                progress_bar.empty()
                
                # Store and display results
                st.session_state.classification_results.extend(results)
                display_classification_results(results)

if __name__ == "__main__":
    # Check for API key - handle both local and Streamlit Cloud
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try Streamlit secrets for cloud deployment
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            os.environ["OPENAI_API_KEY"] = api_key
        except (KeyError, FileNotFoundError):
            st.error("🔑 OpenAI API key not found!")
            st.info("**For Streamlit Cloud:** Add your OPENAI_API_KEY to the app secrets.")
            st.info("**For local deployment:** Set the OPENAI_API_KEY environment variable.")
            st.code("export OPENAI_API_KEY='your-api-key-here'")
            st.stop()
    
    main() 