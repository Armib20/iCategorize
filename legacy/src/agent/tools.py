"""
Tools and external integrations for the AI agent.

Provides capabilities like data export, API integrations, 
and external system connections.
"""

from __future__ import annotations

import csv
import json
import io
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict


class ToolRegistry:
    """
    Registry of tools and external capabilities available to the agent.
    """
    
    def __init__(self):
        self.available_tools = {
            "export_csv": self.export_classifications_csv,
            "export_json": self.export_classifications_json,
            "validate_product": self.validate_product_name,
            "batch_process": self.batch_process_file,
            "generate_report": self.generate_classification_report
        }
    
    def export_classifications(
        self, 
        classifications: List, 
        format: str = "json"
    ) -> str:
        """Export classifications in various formats."""
        if format.lower() == "csv":
            return self.export_classifications_csv(classifications)
        elif format.lower() == "json":
            return self.export_classifications_json(classifications)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def export_classifications_csv(self, classifications: List) -> str:
        """Export classifications as CSV string."""
        if not classifications:
            return "product_name,category,confidence,timestamp\n"
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "product_name", 
            "category", 
            "confidence", 
            "reasoning",
            "timestamp"
        ])
        
        # Write data
        for result in classifications:
            writer.writerow([
                result.product_name,
                result.category,
                f"{result.confidence:.2f}",
                result.reasoning[:100] + "..." if len(result.reasoning) > 100 else result.reasoning,
                result.timestamp.isoformat()
            ])
        
        return output.getvalue()
    
    def export_classifications_json(self, classifications: List) -> str:
        """Export classifications as JSON string."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_classifications": len(classifications),
            "classifications": [asdict(result) for result in classifications]
        }
        return json.dumps(data, default=str, indent=2)
    
    def validate_product_name(self, product_name: str) -> Dict[str, Any]:
        """Validate and clean a product name."""
        # Basic validation and cleaning
        cleaned = product_name.strip()
        
        validation = {
            "original": product_name,
            "cleaned": cleaned,
            "is_valid": len(cleaned) > 0,
            "issues": [],
            "suggestions": []
        }
        
        if len(cleaned) == 0:
            validation["issues"].append("Empty product name")
            validation["suggestions"].append("Provide a product name")
        
        elif len(cleaned) < 3:
            validation["issues"].append("Product name too short")
            validation["suggestions"].append("Provide a more descriptive name")
        
        elif len(cleaned) > 200:
            validation["issues"].append("Product name too long")
            validation["suggestions"].append("Shorten the product name")
        
        # Check for common issues
        if cleaned.lower().startswith("product"):
            validation["suggestions"].append("Remove generic prefixes like 'Product'")
        
        if any(char in cleaned for char in ['<', '>', '{', '}', '[', ']']):
            validation["issues"].append("Contains special characters")
            validation["suggestions"].append("Remove HTML tags or special characters")
        
        return validation
    
    def batch_process_file(self, file_content: str, format: str = "csv") -> List[str]:
        """Process a batch file and extract product names."""
        product_names = []
        
        if format.lower() == "csv":
            # Parse CSV content
            lines = file_content.strip().split('\n')
            reader = csv.reader(lines)
            
            # Skip header if present
            first_row = next(reader, None)
            if first_row and any(header in first_row[0].lower() for header in ['product', 'name', 'item']):
                pass  # Skip header
            else:
                product_names.append(first_row[0] if first_row else "")
            
            # Process remaining rows
            for row in reader:
                if row and len(row) > 0:
                    product_names.append(row[0].strip())
        
        elif format.lower() == "txt":
            # Simple text file, one product per line
            product_names = [line.strip() for line in file_content.split('\n') if line.strip()]
        
        elif format.lower() == "json":
            # JSON format
            try:
                data = json.loads(file_content)
                if isinstance(data, list):
                    product_names = [str(item) for item in data]
                elif isinstance(data, dict) and 'products' in data:
                    product_names = [str(item) for item in data['products']]
            except json.JSONDecodeError:
                pass
        
        return [name for name in product_names if len(name.strip()) > 0]
    
    def generate_classification_report(
        self, 
        classifications: List,
        corrections: List = None
    ) -> str:
        """Generate a comprehensive classification report."""
        if not classifications:
            return "No classifications to report."
        
        # Calculate statistics
        total = len(classifications)
        categories = {}
        confidences = []
        
        for result in classifications:
            categories[result.category] = categories.get(result.category, 0) + 1
            confidences.append(result.confidence)
        
        avg_confidence = sum(confidences) / len(confidences)
        
        # Generate report
        report = f"""# Product Classification Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Classifications:** {total}
- **Unique Categories:** {len(categories)}
- **Average Confidence:** {avg_confidence:.1%}

## Category Breakdown
"""
        
        # Sort categories by frequency
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories:
            percentage = (count / total) * 100
            report += f"- **{category}:** {count} products ({percentage:.1f}%)\n"
        
        # Confidence distribution
        high_conf = sum(1 for c in confidences if c >= 0.8)
        med_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.6)
        
        report += f"""
## Confidence Distribution
- **High Confidence (≥80%):** {high_conf} ({high_conf/total*100:.1f}%)
- **Medium Confidence (60-79%):** {med_conf} ({med_conf/total*100:.1f}%)
- **Low Confidence (<60%):** {low_conf} ({low_conf/total*100:.1f}%)
"""
        
        # Add corrections info if available
        if corrections:
            report += f"""
## User Corrections
- **Total Corrections:** {len(corrections)}
- **Correction Rate:** {len(corrections)/total*100:.1f}%
"""
        
        return report
    
    def integrate_api(self, api_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Template for external API integrations.
        
        This could be extended to integrate with:
        - E-commerce platforms (Shopify, WooCommerce)
        - Product databases (UPC, barcode APIs)
        - Inventory management systems
        - ERP systems
        """
        integrations = {
            "shopify": self._shopify_integration,
            "woocommerce": self._woocommerce_integration,
            "barcode_lookup": self._barcode_lookup_integration
        }
        
        if api_name in integrations:
            return integrations[api_name](data)
        else:
            return {
                "success": False,
                "error": f"Integration '{api_name}' not implemented",
                "available_integrations": list(integrations.keys())
            }
    
    def _shopify_integration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for Shopify integration."""
        return {
            "success": False,
            "message": "Shopify integration not implemented",
            "data": data
        }
    
    def _woocommerce_integration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for WooCommerce integration."""
        return {
            "success": False,
            "message": "WooCommerce integration not implemented", 
            "data": data
        }
    
    def _barcode_lookup_integration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for barcode lookup integration."""
        return {
            "success": False,
            "message": "Barcode lookup integration not implemented",
            "data": data
        }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.available_tools.keys())
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name with arguments."""
        if tool_name in self.available_tools:
            return self.available_tools[tool_name](**kwargs)
        else:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self.available_tools.keys())}") 