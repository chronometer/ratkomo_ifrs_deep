"""
Run IFRS document analysis with optimized API usage
"""
import asyncio
from pathlib import Path
import logging
from document_analyzer_optimized import OptimizedAnalyzer

async def main():
    # Initialize analyzer
    analyzer = OptimizedAnalyzer()
    
    # Analyze document
    document_path = Path('data/annual_report_2023.txt')  # Update path as needed
    
    logging.info("Starting optimized IFRS analysis...")
    report = await analyzer.analyze_document(document_path)
    
    # Save report
    output_path = Path('output/analysis_report.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logging.info(f"Analysis complete. Report saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
