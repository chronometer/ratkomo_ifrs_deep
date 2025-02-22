"""Script to run the document analyzer with pydantic-ai integration
"""
import asyncio
import argparse
import logging
from pathlib import Path
from src.document_analyzer import DocumentAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main(document_path: str):
    """Run the document analyzer on the specified document"""
    try:
        # Convert to Path and validate
        document_path = Path(document_path)
        if not document_path.exists():
            logging.error(f"Document not found at {document_path}")
            return
        
        # Initialize analyzer with pydantic-ai agent
        logging.info(f"Initializing document analyzer for {document_path}")
        analyzer = DocumentAnalyzer()
        
        # Run analysis
        logging.info("Starting document analysis...")
        output_path = await analyzer.analyze_document(document_path)
        
        logging.info("Analysis complete!")
        logging.info(f"Report saved to: {output_path}")
        
        # Display the report
        with open(output_path, encoding='utf-8') as f:
            print("\nReport preview:")
            print("-" * 80)
            print(f.read())
            print("-" * 80)
            
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a document for IFRS compliance using pydantic-ai"
    )
    parser.add_argument(
        "document_path", 
        help="Path to the document to analyze (supports PDF and text files)"
    )
    args = parser.parse_args()
    
    asyncio.run(main(args.document_path))
