"""
Streamlit page for the iTradenetwork Custom Category Demo.

This page provides an interactive UI to:
- Discover business-relevant categories from product data.
- Build a knowledge graph for intelligent classification.
- Classify products using the custom-built system.
- View business insights and recommendations.
"""
import streamlit as st
import pandas as pd
import io
import json
import os
import pathlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from icategorize.custom_classifier import CustomProductClassificationAgent

def reset_all_system_knowledge():
    """Reset all system knowledge including categories, patterns, and cached data."""
    try:
        # Clear session state
        if 'custom_agent' in st.session_state:
            del st.session_state.custom_agent
        if 'bootstrap_done' in st.session_state:
            del st.session_state.bootstrap_done
        if 'bootstrap_results' in st.session_state:
            del st.session_state.bootstrap_results
        if 'uploaded_products' in st.session_state:
            del st.session_state.uploaded_products
        if 'process_steps' in st.session_state:
            del st.session_state.process_steps
        
        # List of files to remove
        files_to_remove = [
            "data/custom_categories.json",
            "data/knowledge_graph.pkl",
            "data/interim/categories_cache.json",
            "data/interim/patterns_cache.json"
        ]
        
        removed_files = []
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_files.append(file_path)
        
        return True, removed_files
        
    except Exception as e:
        return False, str(e)

def get_default_products():
    """Provides a default list of sample product names."""
    return (
        "Organic Hass Avocados, Box of 48, Mexico\n"
        "California Navel Oranges, 5lb bag\n"
        "Fresh Blueberries, 1 pint, Grade A\n"
        "Sweet Onions, 50lb sack, Washington\n"
        "Idaho Russet Potatoes, 10lb bag\n"
        "Honeycrisp Apples, Case, USA\n"
        "Organic Baby Spinach, 1lb clam shell\n"
        "Vine-ripened Tomatoes, 25lb box, Grade 1\n"
        "Frozen Corn, Sweet Yellow, 20lb case\n"
        "Canned Diced Tomatoes, 6/#10 cans/case\n"
        "Imported Italian Pasta, Spaghetti, 20x1lb\n"
        "Extra Virgin Olive Oil, 4x1 Gallon, Spain\n"
        "Almonds, Raw, Unsalted, 25lb box, California\n"
        "Chicken Breast, Boneless, Skinless, Frozen, 40lb case\n"
        "Ground Beef 80/20, 10lb tubes, fresh\n"
        "Shredded Mozzarella Cheese, 4x5lb bags\n"
        "Sourdough Bread, Par-baked, 24 loaves\n"
        "Hass Avocados from Peru, 48 count\n"
        "Organic Fuji Apples, 40lb box\n"
        "Red Bell Peppers, 11lb case, Holland\n"
        "Jumbo Yellow Onions, 50 lbs\n"
        "Broccoli Crowns, 20lb case\n"
        "Seedless Watermelon, each, Mexico\n"
        "Artisan Lettuce Mix, 3lb bag\n"
        "Frozen French Fries, 6x5lb bags"
    )

def process_uploaded_file(uploaded_file, column_name=None):
    """
    Process uploaded CSV or XLSX file and extract product names.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        column_name: Name of the column containing product names
        
    Returns:
        List of product names
    """
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")
            return []
        
        # Display dataframe preview
        st.subheader("📊 File Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Let user select the column containing product names
        if column_name is None:
            column_name = st.selectbox(
                "Select the column containing product names:",
                options=df.columns.tolist(),
                help="Choose the column that contains the product names or descriptions"
            )
        
        if column_name and column_name in df.columns:
            # Extract product names and remove empty values
            products = df[column_name].dropna().astype(str).tolist()
            products = [p.strip() for p in products if p.strip()]
            
            st.success(f"✅ Successfully extracted {len(products)} product names from column '{column_name}'")
            
            # Show sample of extracted products
            st.subheader("🔍 Sample Extracted Products")
            sample_size = min(10, len(products))
            for i, product in enumerate(products[:sample_size], 1):
                st.write(f"{i}. {product}")
            
            if len(products) > sample_size:
                st.info(f"... and {len(products) - sample_size} more products")
            
            return products
        else:
            st.error("Please select a valid column name.")
            return []
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []

def display_step_progress(step_name, status, details=None):
    """Display progress for each step of the categorization process."""
    if status == "running":
        st.info(f"🔄 {step_name} - In Progress...")
        if details:
            st.text(details)
    elif status == "completed":
        st.success(f"✅ {step_name} - Completed")
        if details:
            with st.expander(f"📊 {step_name} Details"):
                st.write(details)
    elif status == "error":
        st.error(f"❌ {step_name} - Failed")
        if details:
            st.text(details)

def visualize_patterns(patterns):
    """Create visualizations for discovered patterns."""
    if not patterns:
        return
    
    # Pattern frequency chart
    pattern_data = []
    for pattern in patterns:
        for value in pattern.values:
            pattern_data.append({
                'Pattern Type': pattern.pattern_type,
                'Value': value,
                'Frequency': pattern.frequency
            })
    
    if pattern_data:
        df_patterns = pd.DataFrame(pattern_data)
        
        # Group by pattern type and sum frequencies
        pattern_summary = df_patterns.groupby('Pattern Type')['Frequency'].sum().reset_index()
        
        fig = px.bar(
            pattern_summary, 
            x='Pattern Type', 
            y='Frequency',
            title="Discovered Pattern Types",
            color='Frequency',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed pattern breakdown
        for pattern_type in df_patterns['Pattern Type'].unique():
            type_data = df_patterns[df_patterns['Pattern Type'] == pattern_type]
            
            with st.expander(f"📋 {pattern_type.title()} Patterns"):
                fig2 = px.pie(
                    type_data, 
                    values='Frequency', 
                    names='Value',
                    title=f"{pattern_type.title()} Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)

def visualize_categories(categories):
    """Create comprehensive category visualizations."""
    if not categories:
        return
    
    # Category size distribution
    cat_data = [{
        'Category': cat.name,
        'Size': cat.size,
        'Confidence': cat.confidence,
        'Keywords': ', '.join(cat.keywords[:3])
    } for cat in categories]
    
    df_cats = pd.DataFrame(cat_data)
    
    # Size distribution chart
    fig1 = px.bar(
        df_cats.sort_values('Size', ascending=False), 
        x='Category', 
        y='Size',
        title="Category Sizes (Number of Products)",
        color='Size',
        color_continuous_scale='blues'
    )
    fig1.update_xaxes(tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Confidence vs Size scatter plot
    fig2 = px.scatter(
        df_cats,
        x='Size',
        y='Confidence',
        hover_data=['Category', 'Keywords'],
        title="Category Confidence vs Size",
        size='Size',
        color='Confidence',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig2, use_container_width=True)

def visualize_knowledge_graph(agent):
    """Visualize the knowledge graph if available."""
    if not agent or not hasattr(agent, 'knowledge_graph') or not agent.knowledge_graph:
        st.info("No knowledge graph available. Complete category discovery first.")
        return
    
    try:
        kg = agent.knowledge_graph
        
        # Get nodes and edges from the knowledge graph
        if hasattr(kg, 'graph') and kg.graph:
            G = kg.graph
            
            # Create network visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Extract node and edge data
            node_trace = []
            edge_trace = []
            
            # Add edges
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace.append(dict(x=[x0, x1, None], y=[y0, y1, None]))
            
            # Add nodes with proper labels
            for node in G.nodes():
                x, y = pos[node]
                # Get node data to extract proper name
                node_data = G.nodes[node]
                
                # Determine display name based on node type and data
                if 'name' in node_data:
                    display_name = node_data['name']
                elif 'original_name' in node_data:
                    display_name = node_data['original_name']
                elif node.startswith('product_'):
                    # For product nodes, try to get the actual product name from the knowledge graph
                    if hasattr(kg, 'nodes') and node in kg.nodes:
                        kg_node = kg.nodes[node]
                        display_name = kg_node.name if hasattr(kg_node, 'name') else str(node)
                    else:
                        display_name = str(node)
                elif node.startswith('category_'):
                    # Clean up category names
                    display_name = node.replace('category_', '').replace('_', ' ').title()
                elif node.startswith('pattern_'):
                    # Clean up pattern names
                    display_name = node.replace('pattern_', '').replace('_', ' ').title()
                else:
                    display_name = str(node)
                
                # Truncate long names for better display
                if len(display_name) > 25:
                    display_name = display_name[:22] + "..."
                
                node_trace.append(dict(x=x, y=y, text=display_name))
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add edges
            for edge in edge_trace:
                fig.add_trace(go.Scatter(
                    x=edge['x'], y=edge['y'],
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                ))
            
            # Add nodes
            node_x = [trace['x'] for trace in node_trace]
            node_y = [trace['y'] for trace in node_trace]
            node_text = [trace['text'] for trace in node_trace]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=20,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                )
            ))
            
            fig.update_layout(
                title="Product Knowledge Graph",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Each node represents a product or category concept",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#888", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Graph statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", G.number_of_nodes())
            with col2:
                st.metric("Total Edges", G.number_of_edges())
            with col3:
                if G.number_of_nodes() > 0:
                    density = nx.density(G)
                    st.metric("Graph Density", f"{density:.3f}")
        
        else:
            st.info("Knowledge graph structure not available for visualization.")
            
    except Exception as e:
        st.error(f"Error visualizing knowledge graph: {str(e)}")
        st.info("This may be normal if the knowledge graph hasn't been fully built yet.")

# Page Configuration
st.set_page_config(
    page_title="iTradenetwork Custom Categories",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'custom_agent' not in st.session_state:
    st.session_state.custom_agent = None
if 'bootstrap_done' not in st.session_state:
    st.session_state.bootstrap_done = False
if 'bootstrap_results' not in st.session_state:
    st.session_state.bootstrap_results = None
if 'uploaded_products' not in st.session_state:
    st.session_state.uploaded_products = []
if 'process_steps' not in st.session_state:
    st.session_state.process_steps = {}

# --- UI ---
st.title("📈 iTradenetwork: Business-Intelligent Categories")
st.markdown("""
This demo showcases an advanced system that automatically discovers business-relevant categories 
from your product listings. Instead of relying on generic FDA codes, it builds a custom
category structure and knowledge graph tailored to how your business and customers think.
""")

# --- Step 1: Data Input and Bootstrapping ---
st.header("Step 1: Discover Categories from Product Data")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Provide Product Listings")
    
    # Add input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Text Entry", "Upload CSV/XLSX File"],
        help="Select how you want to provide your product data"
    )
    
    products_to_process = []
    
    if input_method == "Manual Text Entry":
        st.markdown("Enter a list of product names, one per line. The more examples you provide, the better the category discovery will be.")
        
        product_data = st.text_area(
            "Product Names",
            height=300,
            value=get_default_products(),
            label_visibility="collapsed"
        )
        
        products_to_process = [p.strip() for p in product_data.split('\n') if p.strip()]
        
    else:  # File Upload
        st.markdown("Upload a CSV or XLSX file containing your product names. The system will help you select the correct column.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or XLSX file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing product names in one of the columns"
        )
        
        if uploaded_file is not None:
            # Process the uploaded file
            extracted_products = process_uploaded_file(uploaded_file)
            if extracted_products:
                st.session_state.uploaded_products = extracted_products
                products_to_process = extracted_products
        elif st.session_state.uploaded_products:
            # Use previously uploaded products
            products_to_process = st.session_state.uploaded_products
            st.info(f"Using previously uploaded file with {len(products_to_process)} products")
    
    # Show current product count
    if products_to_process:
        st.metric("Products Ready for Analysis", len(products_to_process))
        
        # Show product preview
        with st.expander("📋 Product Preview"):
            for i, product in enumerate(products_to_process[:10], 1):
                st.write(f"{i}. {product}")
            if len(products_to_process) > 10:
                st.info(f"... and {len(products_to_process) - 10} more products")
    
    col_rebuild, col_reset = st.columns([2, 1])
    
    with col_rebuild:
        force_rebuild = st.checkbox("Force Re-Discovery", help="If checked, will ignore cached categories and rebuild from scratch.")
    
    with col_reset:
        if st.button("🗑️ Quick Reset", help="Reset all system knowledge", type="secondary"):
            success, result = reset_all_system_knowledge()
            if success:
                st.success("✅ System reset!")
                st.rerun()
            else:
                st.error(f"❌ Reset failed: {result}")

    # Update the button logic to use products_to_process
    if st.button("🚀 Discover Categories & Build KG", type="primary", use_container_width=True):
        if len(products_to_process) < 5:  # Reduced minimum for better UX
            st.error("Please provide at least 5 products for meaningful category discovery.")
        else:
            # Clear previous steps
            st.session_state.process_steps = {}
            
            # Create placeholders for step tracking
            step_placeholder = st.empty()
            
            with step_placeholder.container():
                st.markdown("### 🔄 Category Discovery Process")
                
                # Step 1: Initialize Agent
                step1_placeholder = st.empty()
                with step1_placeholder:
                    display_step_progress("Agent Initialization", "running")
                
                try:
                    agent = CustomProductClassificationAgent(enable_knowledge_graph=True, model="gpt-4o")
                    st.session_state.custom_agent = agent
                    step1_placeholder.empty()
                    display_step_progress("Agent Initialization", "completed", "Agent created successfully")
                    
                    # Step 2: Bootstrap Process
                    step2_placeholder = st.empty()
                    with step2_placeholder:
                        display_step_progress("Product Analysis & Category Discovery", "running", 
                                           f"Analyzing {len(products_to_process)} products...")
                    
                    results = agent.bootstrap_from_product_data(products_to_process, force_rebuild=force_rebuild)
                    
                    step2_placeholder.empty()
                    display_step_progress("Product Analysis & Category Discovery", "completed",
                                       f"Discovered {results.get('categories_discovered', 0)} categories")
                    
                    # Store results
                    st.session_state.bootstrap_results = results
                    st.session_state.bootstrap_done = True
                    
                    st.success("🎉 Category discovery and knowledge graph build complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during category discovery: {str(e)}")
                    st.exception(e)

with col2:
    st.subheader("Discovery Summary")
    if st.session_state.bootstrap_done and st.session_state.bootstrap_results:
        results = st.session_state.bootstrap_results
        
        # Main metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Categories Discovered", results.get('categories_discovered', 0))
        with col_b:
            st.metric("Product Patterns Found", results.get('patterns_found', 0))
        with col_c:
            st.metric("Products Analyzed", results.get('products_analyzed', 0))

        # Debug information
        st.markdown("**Debug Information:**")
        with st.expander("🔍 Raw Results Data"):
            st.json(results)

        st.markdown("**Top 5 Discovered Categories:**")
        if results.get('top_categories'):
            df_top_cat = pd.DataFrame(results['top_categories'])
            st.dataframe(df_top_cat, use_container_width=True)
        else:
            st.info("No top categories to display.")
            
        st.markdown("**Discovered Category Hierarchy:**")
        if results.get('category_hierarchy'):
            # Create a simple text representation of the hierarchy for better display
            hierarchy_text = ""
            for main_type, details in results['category_hierarchy'].items():
                hierarchy_text += f"**{main_type}**\n"
                if 'subcategories' in details:
                    for subcat in details['subcategories']:
                        hierarchy_text += f"  - {subcat['name']} ({subcat['size']} products)\n"
                hierarchy_text += "\n"
            
            if hierarchy_text:
                st.markdown(hierarchy_text)
            else:
                st.info("No hierarchy to display.")
        else:
             st.info("No hierarchy generated.")
        
    else:
        st.info("Run the discovery process to see a summary here.")

# --- Detailed Process Visualization ---
if st.session_state.bootstrap_done and st.session_state.custom_agent:
    st.header("📊 Detailed Process Analysis")
    
    agent = st.session_state.custom_agent
    
    # Visualize patterns if available
    if hasattr(agent, 'discovery_engine') and agent.discovery_engine.product_patterns:
        st.subheader("🔍 Discovered Patterns")
        visualize_patterns(agent.discovery_engine.product_patterns)
    
    # Visualize categories if available  
    if agent.custom_categories:
        st.subheader("📈 Category Analysis")
        visualize_categories(agent.custom_categories)

# --- Knowledge Graph Visualization ---
if st.session_state.bootstrap_done and st.session_state.custom_agent:
    st.header("🕸️ Knowledge Graph Visualization")
    visualize_knowledge_graph(st.session_state.custom_agent)

# --- System Reset Section ---
st.header("🔄 System Management")

# Reset functionality UI
st.subheader("🗑️ Reset System Knowledge")
st.markdown("""
**Warning**: This will permanently delete all discovered categories, patterns, and cached data. 
You will need to run category discovery again from scratch.
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**What will be reset:**")
    st.markdown("• All discovered categories and patterns")
    st.markdown("• Knowledge graph data")
    st.markdown("• Cached classification results")
    st.markdown("• Session state and temporary data")

with col2:
    # Add confirmation checkbox
    confirm_reset = st.checkbox("I understand this action cannot be undone", key="confirm_reset")
    
    if st.button("🔄 Reset All System Knowledge", 
                type="secondary", 
                disabled=not confirm_reset,
                use_container_width=True):
        
        with st.spinner("Resetting system knowledge..."):
            success, result = reset_all_system_knowledge()
            
            if success:
                st.success("✅ System knowledge has been reset successfully!")
                if result:  # removed_files list
                    with st.expander("📋 Files Removed"):
                        for file in result:
                            st.write(f"• {file}")
                else:
                    st.info("No cached files were found to remove.")
                
                # Force a rerun to update the UI
                st.rerun()
            else:
                st.error(f"❌ Error resetting system: {result}")

# Current system status
if st.session_state.bootstrap_done:
    st.subheader("📊 Current System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        categories_count = len(st.session_state.custom_agent.custom_categories) if st.session_state.custom_agent and st.session_state.custom_agent.custom_categories else 0
        st.metric("Categories Loaded", categories_count)
    
    with status_col2:
        patterns_count = len(st.session_state.custom_agent.product_patterns) if st.session_state.custom_agent and st.session_state.custom_agent.product_patterns else 0
        st.metric("Patterns Loaded", patterns_count)
    
    with status_col3:
        kg_status = "Active" if st.session_state.custom_agent and st.session_state.custom_agent.knowledge_graph else "Inactive"
        st.metric("Knowledge Graph", kg_status)
    
    # Show cached files status
    st.subheader("💾 Cached Data Files")
    
    cache_files = [
        ("Categories", "data/custom_categories.json"),
        ("Knowledge Graph", "data/knowledge_graph.pkl"),
        ("Patterns Cache", "data/interim/patterns_cache.json"),
        ("Categories Cache", "data/interim/categories_cache.json")
    ]
    
    cache_df = []
    for name, path in cache_files:
        exists = os.path.exists(path)
        size = ""
        if exists:
            try:
                size_bytes = os.path.getsize(path)
                if size_bytes < 1024:
                    size = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size = f"{size_bytes / 1024:.1f} KB"
                else:
                    size = f"{size_bytes / (1024 * 1024):.1f} MB"
            except:
                size = "Unknown"
        
        cache_df.append({
            "File Type": name,
            "Path": path,
            "Status": "✅ Exists" if exists else "❌ Not Found",
            "Size": size
        })
    
    st.dataframe(pd.DataFrame(cache_df), use_container_width=True)

else:
    st.info("No system knowledge currently loaded. Run category discovery to populate the system.")

# --- Step 2: Interactive Classification ---
st.header("Step 2: Classify Products in Real-Time")

if not st.session_state.bootstrap_done:
    st.warning("You must discover categories in Step 1 before you can classify products.")
else:
    product_to_classify = st.text_input(
        "Enter a product name to classify", 
        "Fresh Organic Avocados from Mexico, 48ct box"
    )

    if st.button("Classify Product", use_container_width=True):
        if product_to_classify and st.session_state.custom_agent:
            with st.spinner("Classifying product..."):
                result = st.session_state.custom_agent.classify_product(product_to_classify)
                
                st.subheader(f"Classification for: `{result.product_name}`")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Primary Category", result.primary_category)
                    st.metric("Confidence Score", f"{result.confidence:.2%}")
                    
                    st.markdown("**Secondary Categories:**")
                    if result.secondary_categories:
                        for i, cat in enumerate(result.secondary_categories, 1):
                            st.write(f"{i}. {cat}")
                    else:
                        st.write("None")
                    
                    st.markdown("**Reasoning:**")
                    st.info(result.reasoning)

                with res_col2:
                    st.markdown("**Business Insights:**")
                    if result.business_insights:
                        for key, value in result.business_insights.items():
                            st.write(f"**{key}:** {value}")
                    else:
                        st.write("No specific insights available")

                    st.markdown("**Patterns Matched:**")
                    if result.patterns_matched:
                        for pattern in result.patterns_matched:
                            st.write(f"• {pattern}")
                    else:
                        st.write("None")
                    
                    st.markdown("**Similar Products in Knowledge Graph:**")
                    if result.similar_products:
                        for product in result.similar_products:
                            st.write(f"• {product}")
                    else:
                        st.write("None")
        else:
            st.error("Please enter a product name.")

# --- Step 3: System Insights and Recommendations ---
st.header("Step 3: Category System Health & Recommendations")
if st.session_state.bootstrap_done and st.session_state.custom_agent:
    agent = st.session_state.custom_agent
    
    st.subheader("Discovered Categories Overview")
    if agent.custom_categories:
        # Enhanced category display
        for i, cat in enumerate(sorted(agent.custom_categories, key=lambda x: x.size, reverse=True), 1):
            with st.expander(f"📂 {i}. {cat.name} ({cat.size} products)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {cat.description}")
                    st.write(f"**Confidence:** {cat.confidence:.2%}")
                    st.write(f"**Keywords:** {', '.join(cat.keywords)}")
                
                with col2:
                    st.write("**Sample Products:**")
                    for j, product in enumerate(cat.sample_products[:5], 1):
                        st.write(f"{j}. {product}")
                    if len(cat.sample_products) > 5:
                        st.write(f"... and {len(cat.sample_products) - 5} more")
        
        # Summary table
        cat_df = pd.DataFrame([
            {
                "Category Name": cat.name,
                "Description": cat.description,
                "Size": cat.size,
                "Confidence": f"{cat.confidence:.2f}",
                "Keywords": ", ".join(cat.keywords[:5]),
                "Sample Products": ", ".join(cat.sample_products[:2])
            }
            for cat in sorted(agent.custom_categories, key=lambda x: x.size, reverse=True)
        ])
        
        st.subheader("📋 Categories Summary Table")
        st.dataframe(cat_df, use_container_width=True)
        
        # Add download functionality for discovered categories
        csv_data = cat_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Categories as CSV",
            data=csv_data,
            file_name=f"discovered_categories_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("No categories discovered. Try running the discovery process again.")
    
    st.subheader("System Improvement Suggestions")
    try:
        recommendations = agent.get_category_recommendations()
        if recommendations:
            st.table(recommendations)
        else:
            st.success("The category system looks balanced. No specific recommendations at this time.")
    except Exception as e:
        st.info(f"Recommendations not available: {str(e)}")

else:
    st.warning("Run the discovery process in Step 1 to see system insights.")

# --- Step 4: Bulk Classification (New Feature) ---
if st.session_state.bootstrap_done and st.session_state.custom_agent:
    st.header("Step 4: Bulk Product Classification")
    st.markdown("Upload a file with products to classify them all at once.")
    
    bulk_file = st.file_uploader(
        "Choose a CSV or XLSX file for bulk classification",
        type=['csv', 'xlsx', 'xls'],
        key="bulk_classification_file",
        help="Upload a file containing products you want to classify"
    )
    
    if bulk_file is not None:
        bulk_products = process_uploaded_file(bulk_file)
        
        if bulk_products and st.button("🔄 Classify All Products", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_list = []
            
            for i, product in enumerate(bulk_products):
                status_text.text(f"Classifying product {i+1} of {len(bulk_products)}: {product[:50]}...")
                
                try:
                    result = st.session_state.custom_agent.classify_product(product)
                    results_list.append({
                        'Product Name': result.product_name,
                        'Primary Category': result.primary_category,
                        'Confidence': f"{result.confidence:.2%}",
                        'Secondary Categories': ', '.join(result.secondary_categories),
                        'Reasoning': result.reasoning
                    })
                except Exception as e:
                    st.warning(f"Failed to classify '{product}': {str(e)}")
                    results_list.append({
                        'Product Name': product,
                        'Primary Category': 'Error',
                        'Confidence': '0%',
                        'Secondary Categories': '',
                        'Reasoning': f'Classification failed: {str(e)}'
                    })
                
                progress_bar.progress((i + 1) / len(bulk_products))
            
            status_text.text("Classification complete!")
            
            if results_list:
                results_df = pd.DataFrame(results_list)
                st.subheader("🎯 Bulk Classification Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Classification Results",
                    data=csv_results,
                    file_name=f"bulk_classification_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                ) 