# 📊 Bulk Upload Guide - Custom Category Explorer

## Overview

The Custom Category Explorer now supports bulk file uploads for both **category discovery** and **bulk classification**. This allows you to process hundreds or thousands of products at once using CSV or XLSX files.

## 🚀 New Features

### 1. Bulk Category Discovery
- Upload CSV/XLSX files to discover categories from your entire product catalog
- Automatically extract product names from any column
- Generate custom business-relevant categories from bulk data
- Download discovered categories as CSV for further analysis

### 2. Bulk Product Classification  
- Upload files to classify all products at once using discovered categories
- Real-time progress tracking with status updates
- Download classification results with confidence scores and reasoning
- Handle large datasets efficiently with error handling

## 📁 File Format Requirements

### Supported File Types
- **CSV files** (`.csv`)
- **Microsoft Excel files** (`.xlsx`, `.xls`)

### File Structure
Your file should contain at least one column with product names. Other columns are optional but can provide additional context:

```csv
Product Name,Description,Origin
Organic Avocados Large,Fresh Hass avocados from certified organic farms,Mexico
Fresh Atlantic Salmon Fillets,Wild-caught salmon fillets skin-on,Norway
Whole Grain Artisan Bread,Hand-crafted sourdough bread made with organic flour,Local
Premium Olive Oil Extra Virgin,Cold-pressed olive oil from first harvest,Italy
```

## 🔧 Step-by-Step Usage

### Step 1: Category Discovery from Bulk Data

1. **Choose Input Method**
   - Select "Upload CSV/XLSX File" option in Step 1
   
2. **Upload Your File**
   - Click "Choose a CSV or XLSX file"
   - Select your product data file
   
3. **Column Selection**
   - The system will preview your data
   - Select the column containing product names from the dropdown
   
4. **Review Extracted Data**
   - Verify the extracted product names look correct
   - Check the count of products ready for analysis
   
5. **Discover Categories**
   - Click "🚀 Discover Categories & Build KG"
   - Wait for the AI to analyze your data and discover patterns
   - Review the discovered categories and hierarchy

### Step 2: Bulk Classification

1. **Prerequisites**
   - Complete Step 1 (category discovery) first
   - Ensure categories have been successfully discovered
   
2. **Upload Classification File**
   - In Step 4, upload a CSV/XLSX file containing products to classify
   - This can be the same file or a different one
   
3. **Start Bulk Classification**
   - Select the product column
   - Click "🔄 Classify All Products"
   - Monitor the real-time progress bar
   
4. **Download Results**
   - Review classification results in the table
   - Click "📥 Download Classification Results" to save as CSV

## 📊 Sample Data

A sample CSV file is provided at `data/sample_products.csv` with the following structure:

```csv
Product Name,Description,Origin
Organic Avocados Large,Fresh Hass avocados from certified organic farms,Mexico
Fresh Atlantic Salmon Fillets,Wild-caught salmon fillets skin-on,Norway
...
```

You can use this file to test the bulk upload functionality.

## 🎯 Best Practices

### File Preparation
- **Clean Data**: Remove empty rows and ensure product names are descriptive
- **Consistent Format**: Use consistent naming conventions across products
- **Sufficient Volume**: Provide at least 20-50 products for meaningful category discovery
- **Descriptive Names**: Include relevant details (origin, processing, size) in product names

### Column Selection
- Choose the most descriptive column containing full product names
- Combine multiple columns if needed before uploading
- Avoid columns with only codes or abbreviated names

### Performance Optimization
- **Large Files**: For files with 1000+ products, consider splitting into smaller batches
- **Processing Time**: Allow 2-3 seconds per product for classification
- **Memory Usage**: Close other applications when processing very large files

## ⚠️ Limitations and Troubleshooting

### File Size Limits
- **Recommended**: Up to 1000 products per file for optimal performance
- **Maximum**: System can handle larger files but processing time increases significantly

### Common Issues

**"Empty vocabulary after processing"**
- Your product names may be too generic or short
- Try using more descriptive product names
- Ensure minimum 5-10 products for discovery

**"Failed to classify product"**
- Individual product classification failures won't stop the batch
- Check the warning messages for specific products that failed
- Review failed products manually if needed

**File Upload Errors**
- Ensure file is in CSV or XLSX format
- Check that the file isn't corrupted or password-protected
- Refresh the page and try uploading again

### Performance Tips
- Use "Force Re-Discovery" sparingly as it rebuilds the entire category system
- Cache is preserved between sessions for faster subsequent runs
- Download results immediately after processing to avoid loss

## 📈 Output Formats

### Category Discovery Results
- **Categories CSV**: Contains discovered category names, descriptions, sizes, and keywords
- **Hierarchy Display**: Visual representation of category relationships

### Classification Results  
- **Product Name**: Original product name from your file
- **Primary Category**: Best matching discovered category
- **Confidence**: Percentage confidence in the classification
- **Secondary Categories**: Alternative category matches
- **Reasoning**: AI explanation for the classification decision

## 🔄 Workflow Integration

This bulk upload functionality integrates seamlessly with existing workflows:

1. **Data Import** → Upload your product catalog
2. **Category Discovery** → Let AI discover business-relevant categories  
3. **Classification** → Classify all products using discovered categories
4. **Export Results** → Download structured data for your systems
5. **Continuous Learning** → Re-run discovery as your catalog grows

## 🆘 Support

If you encounter issues with bulk uploads:

1. Check the file format and structure
2. Verify column selection is correct
3. Review error messages for specific guidance
4. Try with a smaller subset of data first
5. Ensure your OpenAI API key is properly configured

The system is designed to be robust and provide helpful error messages to guide you through any issues. 