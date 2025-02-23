"""
Run IFRS analysis using Chain of Agents with OpenRouter and pydantic-ai
"""
import asyncio
import logging
import sys
import os
from pathlib import Path
from src.orchestrator.ifrs_orchestrator import IFRSOrchestrator

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('ifrs_analysis.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

async def main():
    try:
        # Validate environment
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.error("OPENROUTER_API_KEY environment variable is required")
            sys.exit(1)
            
        # Initialize orchestrator with Gemini model
        orchestrator = IFRSOrchestrator(
            num_workers=1,  # Single worker for stability
            model_name="google/gemini-2.0-flash-001",  # Use Gemini model
            min_segment_size=12000,  # Large segments to reduce total count
            max_segment_size=24000,  # Increased max size
            batch_size=1  # Process one at a time for reliability
        )
    
        # Analyze document
        document_path = Path('pihlajalinna-oyj-vuosiraportti-2023 (1).txt')
        
        logger.info("Starting IFRS analysis using Chain of Agents...")
        output_path = await orchestrator.analyze_and_save_report(document_path)
        
        logger.info(f"Analysis complete. Report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
