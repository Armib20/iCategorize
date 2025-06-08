#!/usr/bin/env python3
"""
Launch the FDA Product Classification Streamlit App

This is the main launcher for the reorganized iCategorize project.
"""

import os
import sys
import subprocess


def main():
    """Launch the Streamlit application."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        print("📝 Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   Or add it to a .env file in the project root")
        print()
        print("💡 You can also copy config/env_example.txt to .env and edit it")
        return 1
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Error: Streamlit is not installed")
        print("📦 Please install the requirements:")
        print("   pip install -r config/requirements.txt")
        return 1
    
    print("🏷️ iCategorize - FDA Product Classification Assistant")
    print("=" * 60)
    print("🚀 Starting the Streamlit web application...")
    print("🌐 The app will open in your browser at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    # Get the app.py path
    app_path = os.path.join(os.path.dirname(__file__), "app", "app.py")
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, 
            "-m", "streamlit", 
            "run", 
            app_path,
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 