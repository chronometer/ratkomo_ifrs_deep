"""
Script to run the document analyzer on a sample document
"""
import asyncio
from pathlib import Path
from src.document_analyzer import DocumentAnalyzer

async def main():
    # Initialize the analyzer
    analyzer = DocumentAnalyzer()
    
    # Path to the test document
    doc_path = Path(__file__).parent / "pihlajalinna-oyj-vuosiraportti-2023 (1).pdf"
    
    print(f"Starting analysis of {doc_path}")
    print("This may take a few minutes depending on the document size...")
    
    # Run the analysis
    output_path = await analyzer.analyze_document(doc_path)
    
    print(f"\nAnalysis complete! Check the results at:")
    print(f"Report: {output_path}")
    print(f"Structured data: {output_path.parent / 'analysis_data.json'}")

if __name__ == "__main__":
    asyncio.run(main())
