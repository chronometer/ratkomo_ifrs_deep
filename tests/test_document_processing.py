"""
Test document processing and segmentation
"""
import unittest
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.process_report import ReportProcessor
from src.orchestration.langchain_orchestrator import LangchainOrchestrator

class TestDocumentProcessing(unittest.TestCase):
    def setUp(self):
        # Create a simple test PDF file
        self.test_output_dir = Path("test_output")
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Use a sample PDF from the test_data directory
        self.test_pdf = project_root / "tests" / "test_data" / "sample.pdf"
        
    def test_document_processing(self):
        """Test that document processing returns the expected format"""
        processor = ReportProcessor(str(self.test_pdf), str(self.test_output_dir))
        result = processor.process()
        
        # Check basic structure
        self.assertIsInstance(result, dict)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        
        # Check metadata fields
        metadata = result["metadata"]
        self.assertIn("title", metadata)
        self.assertIn("pages", metadata)
        self.assertIsInstance(metadata["pages"], int)
        
        # Check content
        self.assertIsInstance(result["content"], str)
        self.assertGreater(len(result["content"]), 0)
        
    def test_document_segmentation(self):
        """Test that document segmentation works correctly"""
        # Initialize orchestrator with small segment size for testing
        orchestrator = LangchainOrchestrator(segment_size=100)
        
        # Create a test document
        test_document = {
            "content": "This is a test document " * 10,  # Create content > 100 chars
            "metadata": {
                "title": "Test Doc",
                "pages": 1
            }
        }
        
        # Test segmentation
        segments = orchestrator._segment_document(test_document)
        
        # Verify segments
        self.assertGreater(len(segments), 1)  # Should have multiple segments
        
        for segment in segments:
            # Check segment structure
            self.assertIn("id", segment)
            self.assertIn("content", segment)
            self.assertIn("start_pos", segment)
            self.assertIn("end_pos", segment)
            
            # Check segment size
            self.assertLessEqual(len(segment["content"]), orchestrator.segment_size)
            
            # Check position consistency
            self.assertEqual(
                len(segment["content"]), 
                segment["end_pos"] - segment["start_pos"]
            )

if __name__ == "__main__":
    unittest.main()
