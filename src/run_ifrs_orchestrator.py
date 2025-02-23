"""
Script to run IFRS document analysis
"""
import logging
from pathlib import Path
from src.utils.pdf_utils import pdf_to_text
from src.orchestrator.ifrs_orchestrator import IFRSOrchestrator

logging.basicConfig(level=logging.INFO)

async def main():
    # Convert PDF to text
    pdf_path = Path("/Users/jargothia/Projects/Ratkomo/Agent_chain_doc_analysis/pihlajalinna-oyj-vuosiraportti-2023 (1).pdf")
    text_path = pdf_to_text(pdf_path)
    
    # Initialize orchestrator
    orchestrator = IFRSOrchestrator()
    
    # Run analysis
    report = await orchestrator.analyze_and_save_report(text_path)
    logging.info(f"Analysis complete. Report saved to: {report}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
