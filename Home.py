"""
# iCategorize - Main Welcome Page
"""
import streamlit as st

st.set_page_config(
    page_title="Welcome to iCategorize",
    page_icon="👋",
    layout="wide"
)

st.title("👋 Welcome to iCategorize!")
st.sidebar.success("Select a classifier demo above.")

st.markdown(
    """
    iCategorize is a powerful tool for automatically classifying product names.
    This application demonstrates two different classification approaches:

    ### 1. FDA Product Code Classifier
    - Classifies products against the official FDA product category codes.
    - Useful for regulatory compliance and standardized categorization.
    - **👈 Select "FDA Classifier" from the sidebar to get started!**

    ### 2. iTradenetwork Custom Category Explorer
    - A business-intelligent system that discovers custom categories from your data.
    - Creates a knowledge graph to provide nuanced, market-relevant classifications.
    - **NEW:** Bulk upload CSV/XLSX files to process thousands of products at once.
    - **NEW:** Step-by-step process visualization with interactive charts and graphs.
    - **NEW:** Knowledge graph visualization with network diagrams.
    - **NEW:** Complete system reset functionality to start fresh.
    - Includes bulk classification and downloadable results.
    - Ideal for businesses like iTradenetwork that need a bespoke category system.
    - **👈 Select "Custom Category Explorer" from the sidebar to try it out!**

    ---

    **To begin, choose a demonstration from the navigation bar on the left.**
"""
) 