"""
Test IFRS analysis results generation
"""
import unittest
import asyncio
from pathlib import Path
import sys
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.analyze_ifrs_report import IFRSReportAnalyzer

class TestAgentAnalysis(unittest.TestCase):
    async def test_analysis_results(self):
        """Test that agents generate proper analysis results"""
        # Initialize analyzer with a real PDF
        analyzer = IFRSReportAnalyzer(
            pdf_path="pihlajalinna-oyj-vuosiraportti-2023 (1).pdf",
            output_dir="test_output",
            num_workers=3,
            segment_size=1000
        )
        
        # Run analysis
        results = await analyzer.process_and_analyze()
        
        # Check results structure
        self.assertTrue(results["success"])
        self.assertIn("results", results)
        self.assertIn("technical_analysis", results)
        self.assertIn("business_analysis", results)
        self.assertIn("detailed_findings", results)
        
        # Check that we have actual content
        self.assertIsInstance(results["results"], str)
        self.assertGreater(len(results["results"]), 0)
        
        # Check detailed findings
        findings = results["detailed_findings"]
        self.assertIsInstance(findings, list)
        self.assertGreater(len(findings), 0)
        
        # Print sample of results for inspection
        print("\nSample of analysis results:")
        print("===========================")
        print(f"Executive Summary (first 200 chars):\n{results['results'][:200]}...")
        print("\nNumber of detailed findings:", len(findings))
        if findings:
            print("\nFirst finding sample:")
            print(json.dumps(findings[0], indent=2))

def run_async_test():
    """Run async test"""
    test = TestAgentAnalysis()
    test.setUp()
    asyncio.run(test.test_analysis_results())
    test.tearDown()

if __name__ == "__main__":
    run_async_test()
