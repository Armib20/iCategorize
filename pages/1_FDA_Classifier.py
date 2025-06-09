import streamlit as st
import pandas as pd
import io
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import json

# Corrected import for the new structure
from icategorize.fda_classifier import SimplifiedProductClassificationAgent, classify_llm_hybrid
from icategorize.fda_classifier.agent import ClassificationResult # Agent still uses this

# Configure page
st.set_page_config(
    page_title="FDA Product Classifier",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'classification_agent' not in st.session_state:
    st.session_state.classification_agent = SimplifiedProductClassificationAgent(
        model="gpt-4o",
        enable_learning=True
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []

def classify_products_with_ground_truth(
    products: List[str], 
    ground_truth: Optional[List[str]] = None,
    **kwargs
) -> List[ClassificationResult]:
    """Classify products with optional ground truth comparison."""
    if not products:
        return []
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Set up accuracy tracking if ground truth is provided
    accuracy_container = None
    accuracy_metrics = None
    if ground_truth:
        accuracy_container = st.container()
        with accuracy_container:
            st.subheader("🎯 Real-time Accuracy Tracking")
            accuracy_metrics = {
                'total_processed': 0,
                'correct_predictions': 0,
                'running_accuracy': 0.0
            }
            
            # Create columns for metrics display
            col1, col2, col3 = st.columns(3)
            metric_processed = col1.empty()
            metric_accuracy = col2.empty()
            metric_correct = col3.empty()
            
            # Initialize metrics display
            metric_processed.metric("Processed", "0")
            metric_accuracy.metric("Accuracy", "0.0%")
            metric_correct.metric("Correct", "0/0")
    
    for i, product in enumerate(products):
        status_text.text(f"Classifying: {product}")
        
        # Get classification result
        result = st.session_state.classification_agent.classify_product(
            product, 
            explain=kwargs.get('include_reasoning', True),
            method=kwargs.get('classification_method', 'hybrid')
        )
        results.append(result)
        
        # Update accuracy tracking if ground truth is available
        if ground_truth and i < len(ground_truth):
            accuracy_metrics['total_processed'] = i + 1
            
            # Check if prediction is correct
            if result.category == ground_truth[i]:
                accuracy_metrics['correct_predictions'] += 1
            
            # Calculate running accuracy
            accuracy_metrics['running_accuracy'] = (
                accuracy_metrics['correct_predictions'] / accuracy_metrics['total_processed']
            )
            
            # Update metrics display
            with accuracy_container:
                metric_processed.metric(
                    "Processed", 
                    f"{accuracy_metrics['total_processed']}"
                )
                metric_accuracy.metric(
                    "Accuracy", 
                    f"{accuracy_metrics['running_accuracy']:.1%}",
                    delta=f"{accuracy_metrics['running_accuracy']:.1%}" if i == 0 else None
                )
                metric_correct.metric(
                    "Correct", 
                    f"{accuracy_metrics['correct_predictions']}/{accuracy_metrics['total_processed']}"
                )
        
        progress_bar.progress((i + 1) / len(products))
    
    status_text.text("✅ Classification complete!")
    progress_bar.empty()
    
    # Show final accuracy summary if ground truth was provided
    if ground_truth and accuracy_metrics:
        with accuracy_container:
            st.success(
                f"🎉 Final Accuracy: {accuracy_metrics['running_accuracy']:.1%} "
                f"({accuracy_metrics['correct_predictions']}/{accuracy_metrics['total_processed']})"
            )
            
            # Show detailed breakdown
            with st.expander("📊 Detailed Accuracy Breakdown"):
                # Create a dataframe showing correct/incorrect predictions
                breakdown_data = []
                for i, (product, result) in enumerate(zip(products, results)):
                    if i < len(ground_truth):
                        is_correct = result.category == ground_truth[i]
                        breakdown_data.append({
                            'Product': product,
                            'Predicted': result.category,
                            'Ground Truth': ground_truth[i],
                            'Correct': '✅' if is_correct else '❌',
                            'Confidence': f"{result.confidence:.1%}"
                        })
                
                if breakdown_data:
                    df_breakdown = pd.DataFrame(breakdown_data)
                    st.dataframe(df_breakdown, use_container_width=True)
                    
                    # Show error analysis
                    incorrect_results = [row for row in breakdown_data if row['Correct'] == '❌']
                    if incorrect_results:
                        st.write("**Common Misclassifications:**")
                        error_patterns = {}
                        for row in incorrect_results:
                            pattern = f"{row['Ground Truth']} → {row['Predicted']}"
                            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
                        
                        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"- {pattern}: {count} occurrences")
                            
                        # Suggest improvements
                        st.info("💡 **Tip:** These misclassifications can often be improved by using more specific product descriptions, or by switching classification methods (hybrid vs semantic).")
    
    return results

def classify_products_from_data(products: List[str]) -> List[ClassificationResult]:
    """Classify a list of products and return results."""
    if not products:
        return []
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, product in enumerate(products):
        status_text.text(f"Classifying: {product}")
        result = st.session_state.classification_agent.classify_product(product, explain=True)
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
        
        if model_choice != st.session_state.classification_agent.model:
            st.session_state.classification_agent = SimplifiedProductClassificationAgent(
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
        
        # Debug mode
        debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=False,
            help="Shows detailed classification decisions and AI reasoning process"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Classifications below this confidence will be flagged for review"
        )
        
        st.divider()
        
        # Statistics
        st.header("📊 Session Stats")
        stats = st.session_state.classification_agent.get_stats()
        st.metric("Products Classified", len(st.session_state.classification_results))
        st.metric("Chat Messages", len(st.session_state.chat_history))
        
        # Clear session
        if st.button("Reset Session", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.classification_results = []
            st.session_state.classification_agent = SimplifiedProductClassificationAgent(
                model=model_choice,
                enable_learning=True
            )
    
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
                response = st.session_state.classification_agent.chat(prompt)
            
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
                ground_truth = None  # Initialize ground_truth variable
                
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
                        
                        # Optional ground truth selection
                        enable_ground_truth = st.checkbox(
                            "📊 Enable accuracy benchmarking (optional)",
                            help="Select if you have a column with correct FDA categories for comparison"
                        )
                        
                        ground_truth = None
                        if enable_ground_truth:
                            other_columns = [col for col in df.columns.tolist() if col != product_column]
                            if other_columns:
                                ground_truth_column = st.selectbox(
                                    "Select the ground truth column:",
                                    other_columns,
                                    help="Column containing the correct FDA categories"
                                )
                                ground_truth = df[ground_truth_column].dropna().tolist()
                                st.success(f"🎯 Ground truth benchmarking enabled with column: **{ground_truth_column}**")
                                st.info("Real-time accuracy tracking will be shown during processing!")
                            else:
                                st.warning("No other columns available for ground truth selection.")
                    else:
                        products = df.iloc[:, 0].dropna().tolist()
                        ground_truth = None
                
                elif uploaded_file.name.endswith('.txt'):
                    content = str(uploaded_file.read(), "utf-8")
                    products = [line.strip() for line in content.split('\n') if line.strip()]
                    ground_truth = None  # Text files don't support ground truth
                    st.write(f"**Found {len(products)} products in text file**")
                    with st.expander("Preview products"):
                        st.write(products[:10])  # Show first 10
                    st.info("💡 For accuracy tracking, upload CSV/Excel with ground_truth column")
                
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
                        
                        # Optional ground truth selection
                        enable_ground_truth = st.checkbox(
                            "📊 Enable accuracy benchmarking (optional)",
                            help="Select if you have a column with correct FDA categories for comparison",
                            key="xlsx_ground_truth"
                        )
                        
                        ground_truth = None
                        if enable_ground_truth:
                            other_columns = [col for col in df.columns.tolist() if col != product_column]
                            if other_columns:
                                ground_truth_column = st.selectbox(
                                    "Select the ground truth column:",
                                    other_columns,
                                    help="Column containing the correct FDA categories",
                                    key="xlsx_ground_truth_col"
                                )
                                ground_truth = df[ground_truth_column].dropna().tolist()
                                st.success(f"🎯 Ground truth benchmarking enabled with column: **{ground_truth_column}**")
                                st.info("Real-time accuracy tracking will be shown during processing!")
                            else:
                                st.warning("No other columns available for ground truth selection.")
                    else:
                        products = df.iloc[:, 0].dropna().tolist()
                        ground_truth = None
                
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
                        
                        # Prepare ground truth if available and limit it to match products
                        limited_ground_truth = None
                        if ground_truth:
                            limited_ground_truth = ground_truth[:max_products]
                            # Ensure lengths match
                            min_length = min(len(limited_products), len(limited_ground_truth))
                            limited_products = limited_products[:min_length]
                            limited_ground_truth = limited_ground_truth[:min_length]
                        
                        # Run classification with ground truth support
                        results = classify_products_with_ground_truth(
                            limited_products,
                            ground_truth=limited_ground_truth,
                            include_reasoning=include_reasoning,
                            classification_method=classification_method
                        )
                        
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
                # Manual entries don't have ground truth
                results = classify_products_with_ground_truth(
                    products,
                    ground_truth=None,
                    include_reasoning=True,
                    classification_method=classification_method
                )
                
                # Store and display results
                st.session_state.classification_results.extend(results)
                display_classification_results(results)

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 Please set your OPENAI_API_KEY environment variable to use this application.")
        st.info("You can add it to a .env file in the project root or export it in your terminal.")
        st.stop()
    
    main() 