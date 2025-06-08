#!/usr/bin/env python3
"""
Deployment Debug Script for iCategorize

This script helps diagnose deployment issues by checking:
1. File system structure
2. FDA categories file accessibility
3. Import functionality
4. Path resolution

Run this script in your deployment environment to debug issues.
"""

import json
import pathlib
import sys
import traceback
from typing import Dict, List

def check_file_system():
    """Check the file system structure."""
    print("🔍 File System Check")
    print("=" * 50)
    
    current_file = pathlib.Path(__file__).resolve()
    print(f"Script location: {current_file}")
    print(f"Working directory: {pathlib.Path.cwd()}")
    print(f"Python path: {sys.path}")
    
    # Check if we're in a deployment environment
    deployment_indicators = [
        "/mount/src/",
        "/app/",
        "/home/appuser/",
        "streamlit"
    ]
    
    is_deployment = any(indicator in str(current_file) for indicator in deployment_indicators)
    print(f"Deployment environment detected: {is_deployment}")
    
    print()

def check_fda_categories_file():
    """Check FDA categories file accessibility."""
    print("📁 FDA Categories File Check")
    print("=" * 50)
    
    # The two main paths to check
    possible_paths = [
        # Development: relative to repo root
        pathlib.Path("data/fda_categories.json"),
        # Streamlit Cloud: absolute deployment path  
        pathlib.Path("/mount/src/icategorize/data/fda_categories.json"),
    ]
    
    environment_names = ["Development (local)", "Streamlit Cloud (deployment)"]
    
    found_paths = []
    for i, (path, env_name) in enumerate(zip(possible_paths, environment_names)):
        exists = path.exists()
        print(f"{i+1}. {env_name}")
        print(f"   Path: {path}")
        print(f"   {'✅ Found' if exists else '❌ Not found'}")
        
        if exists:
            found_paths.append(path)
            try:
                with path.open("r") as f:
                    data = json.load(f)
                print(f"   📊 Contains {len(data)} categories")
                
                # Show first few categories
                categories = list(data.keys())[:3]
                print(f"   📝 Sample categories: {categories}")
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        print()
    
    if found_paths:
        print(f"✅ Found {len(found_paths)} accessible FDA categories file(s)")
        return found_paths[0]  # Return the first found path
    else:
        print("❌ No FDA categories file found in either location")
        print("\n💡 Expected locations:")
        print("   - Development: data/fda_categories.json (from repo root)")
        print("   - Streamlit Cloud: /mount/src/icategorize/data/fda_categories.json")
        return None

def check_imports():
    """Check if core modules can be imported."""
    print("📦 Import Check")
    print("=" * 50)
    
    imports_to_test = [
        ("core", "Core module"),
        ("core.classifier", "Classifier module"),
        ("core.agent", "Agent module"),
        ("core.classifier._load_fda_categories", "FDA categories loader"),
        ("core.agent.SimplifiedProductClassificationAgent", "Agent class"),
    ]
    
    successful_imports = 0
    
    for import_path, description in imports_to_test:
        try:
            if "." in import_path and not import_path.endswith("_load_fda_categories"):
                # Handle class imports
                module_path, class_name = import_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
            else:
                # Handle module imports
                __import__(import_path)
            
            print(f"✅ {description}: OK")
            successful_imports += 1
        except Exception as e:
            print(f"❌ {description}: {e}")
            print(f"   Traceback: {traceback.format_exc().splitlines()[-1]}")
    
    print(f"\n📊 Import Summary: {successful_imports}/{len(imports_to_test)} successful")
    print()

def test_fda_categories_loading():
    """Test FDA categories loading functionality."""
    print("🧪 FDA Categories Loading Test")
    print("=" * 50)
    
    try:
        from core.classifier import _load_fda_categories
        categories = _load_fda_categories()
        
        print(f"✅ Successfully loaded {len(categories)} categories")
        
        # Show some sample categories
        sample_categories = list(categories.keys())[:5]
        print(f"📝 Sample categories: {sample_categories}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load FDA categories: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_agent_initialization():
    """Test agent initialization."""
    print("🤖 Agent Initialization Test")
    print("=" * 50)
    
    try:
        from core.agent import SimplifiedProductClassificationAgent
        agent = SimplifiedProductClassificationAgent()
        
        print(f"✅ Agent initialized successfully")
        print(f"📊 Agent has {len(agent.fda_categories)} categories loaded")
        print(f"🔧 Agent model: {agent.model}")
        print(f"🆔 Session ID: {agent.session_id}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all diagnostic checks."""
    print("🚀 iCategorize Deployment Diagnostic")
    print("=" * 60)
    print()
    
    # Run all checks
    check_file_system()
    fda_file_path = check_fda_categories_file()
    check_imports()
    fda_loading_ok = test_fda_categories_loading()
    agent_init_ok = test_agent_initialization()
    
    # Summary
    print("📋 Diagnostic Summary")
    print("=" * 50)
    print(f"FDA categories file found: {'✅' if fda_file_path else '❌'}")
    print(f"FDA categories loading: {'✅' if fda_loading_ok else '❌'}")
    print(f"Agent initialization: {'✅' if agent_init_ok else '❌'}")
    
    if fda_file_path and fda_loading_ok and agent_init_ok:
        print("\n🎉 All checks passed! Your deployment should work correctly.")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        
        if not fda_file_path:
            print("\n💡 Suggestion: Ensure data/fda_categories.json is included in your deployment.")
        
        if not fda_loading_ok or not agent_init_ok:
            print("\n💡 Suggestion: Check import paths and file permissions.")

if __name__ == "__main__":
    main() 