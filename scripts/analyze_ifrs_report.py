"""
IFRS Report Analysis Script
Combines PDF processing with IFRS compliance analysis using LLM-powered agents
"""
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime
import signal
import os
import psutil
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.orchestration.langchain_orchestrator import LangchainOrchestrator
from scripts.process_report import ReportProcessor

class IFRSReportAnalyzer:
    def __init__(
        self,
        pdf_path: str,
        output_dir: str = "output",
        num_workers: int = 3,
        segment_size: int = 1000
    ):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.segment_size = segment_size
        
        # Initialize processor and orchestrator
        self.processor = ReportProcessor(str(self.pdf_path), output_dir)
        self.orchestrator = LangchainOrchestrator(
            num_workers=num_workers,
            segment_size=segment_size
        )
        
    async def process_and_analyze(self) -> Dict[str, Any]:
        """Process PDF and run IFRS analysis"""
        # Step 1: Process the PDF
        print("Processing PDF document...")
        document_content = self.processor.process()
        
        if document_content is None:
            raise ValueError("Failed to process PDF document. Check if the file exists and is accessible.")
        
        # Step 2: Prepare analysis parameters
        standards = ["IFRS 15", "IFRS 16", "IAS 1"]  # Add more standards as needed
        fiscal_period = {
            "start": "2023-01-01",
            "end": "2023-12-31"
        }
        company_info = {
            "name": "Pihlajalinna Oyj",
            "industry": "Healthcare",
            "country": "Finland"
        }
        
        # Calculate total segments
        content_length = len(document_content["content"])
        total_segments = content_length // self.segment_size + (1 if content_length % self.segment_size > 0 else 0)
        
        # Step 3: Run the analysis
        print("Running IFRS analysis...")
        results = await self.orchestrator.process_document(
            document=document_content,
            task_type="ifrs_compliance",
            standards=standards,
            fiscal_period=fiscal_period,
            company_info=company_info,
            start_time=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            total_segments=total_segments
        )
        
        # Step 4: Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON output
        json_file = self.output_dir / f"ifrs_analysis_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Generate and save markdown output
        markdown_content = [
            "# IFRS Analysis Report\n\n",
            "## Executive Summary\n\n",
            f"{results.get('results', 'No summary available.')}\n\n",
            "## Technical Analysis\n\n",
            f"{results.get('technical_analysis', 'No technical analysis available.')}\n\n",
            "## Business Impact Analysis\n\n",
            f"{results.get('business_analysis', 'No business analysis available.')}\n\n",
            "## Detailed Findings\n\n"
        ]
        
        # Add detailed findings
        if 'detailed_findings' in results:
            for segment in results['detailed_findings']:
                markdown_content.append(f"### Segment {segment['segment_id']}\n\n")
                for iteration in segment.get('iterations', []):
                    markdown_content.append(f"#### Iteration {iteration['iteration'] + 1}\n\n")
                    for finding in iteration.get('findings', []):
                        markdown_content.append(f"- **{finding['worker']}**: {finding['analysis']}\n\n")
        
        markdown_content = ''.join(markdown_content)
        
        # Save markdown output
        markdown_file = self.output_dir / f"ifrs_analysis_{timestamp}.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"Analysis complete. Results saved to:")
        print(f"- JSON: {json_file}")
        print(f"- Markdown: {markdown_file}")
        
        return results

def cleanup_processes():
    """Clean up any running analysis processes"""
    current_process = psutil.Process(os.getpid())
    
    # Get all processes with the same parent
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if this is a Python process running our script
            if proc.info['cmdline'] and 'analyze_ifrs_report.py' in ' '.join(proc.info['cmdline']):
                if proc.pid != current_process.pid:  # Don't kill ourselves
                    print(f"Cleaning up process {proc.pid}")
                    proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\nReceived interrupt signal. Cleaning up...")
    cleanup_processes()
    sys.exit(0)

async def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if len(sys.argv) < 2:
        print("Please provide the path to the PDF file as an argument")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    try:
        # Initialize analyzer
        analyzer = IFRSReportAnalyzer(
            pdf_path=pdf_path,
            output_dir="output",
            num_workers=3,
            segment_size=1000
        )
        
        # Run analysis
        results = await analyzer.process_and_analyze()
        
        # Print summary
        print("\nAnalysis Summary:")
        print("-" * 50)
        
        # Show preview of markdown output
        if results.get('results'):
            print("Executive Summary:")
            print(results['results'][:500] + "..." if len(results['results']) > 500 else results['results'])
            print("\nSee the markdown file for the complete analysis.")
            print("\nKey Findings:")
            for finding in results["findings"][:5]:  # Show top 5 findings
                print(f"- {finding}")
        
        if "recommendations" in results:
            print("\nRecommendations:")
            for rec in results["recommendations"][:3]:  # Show top 3 recommendations
                print(f"- {rec}")
                
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        cleanup_processes()
        sys.exit(1)
    finally:
        cleanup_processes()

if __name__ == "__main__":
    asyncio.run(main())
