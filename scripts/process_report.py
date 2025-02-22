"""
Universal report processor for extracting content and structure from PDF reports
"""
import PyPDF2
import re
from pathlib import Path
import fitz  # PyMuPDF
import io
import os
from PIL import Image
import numpy as np

class ReportProcessor:
    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_images(self, pdf_document):
        """Extract images from PDF and save them"""
        image_paths = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Try to determine image type
                image_ext = base_image.get("ext", "png")
                
                # Save image
                image_path = f"images/page_{page_num + 1}_image_{img_index + 1}.{image_ext}"
                image_file = self.output_dir / image_path
                
                with open(image_file, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_paths.append(image_path)
                
        return image_paths

    def clean_text(self, text: str) -> str:
        """Clean and format extracted text"""
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(?<=[a-z])-\s*\n(?=[a-z])', '', text)  # Join hyphenated words
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)  # Join broken words
        
        # Detect and format section headers based on common patterns:
        # 1. All caps text blocks
        # 2. Numbered sections
        # 3. Text followed by page numbers
        # 4. Larger or bold text (detected by PDF metadata)
        
        # Format major section headers (all caps with optional numbers)
        text = re.sub(r'\n([A-Z][A-Z\s]{5,}[A-Z])\s*\n', r'\n## \1\n', text)
        
        # Format subsection headers (numbered sections)
        text = re.sub(r'\n(\d+\.(?:\d+\.)*\s+[^\n]+)\s*\n', r'\n### \1\n', text)
        
        return text.strip()

    def detect_sections(self, text: str) -> list:
        """Detect document sections based on structural patterns"""
        sections = []
        
        # Find all potential section headers (marked with ## from clean_text)
        section_matches = list(re.finditer(r'\n##\s+(.*?)\n', text))
        
        if section_matches:
            for i, match in enumerate(section_matches):
                section_title = match.group(1).strip()
                start = match.end()
                end = section_matches[i + 1].start() if i < len(section_matches) - 1 else len(text)
                
                section_content = text[start:end].strip()
                sections.append({
                    'title': section_title,
                    'content': section_content,
                    'level': 1  # Main section
                })
                
                # Look for subsections within this section
                subsection_matches = re.finditer(r'\n###\s+(.*?)\n(.*?)(?=\n###|\Z)', 
                                               section_content, 
                                               re.DOTALL)
                
                for submatch in subsection_matches:
                    sections.append({
                        'title': submatch.group(1).strip(),
                        'content': submatch.group(2).strip(),
                        'level': 2  # Subsection
                    })
        
        return sections

    def process(self):
        """Process PDF report and extract content"""
        try:
            # Open PDF with both PyPDF2 and PyMuPDF
            pdf_reader = PyPDF2.PdfReader(self.pdf_path)
            pdf_document = fitz.open(self.pdf_path)
            
            # Extract text
            content = []
            for page in pdf_reader.pages:
                content.append(page.extract_text())
            
            # Clean and format text
            text = '\n'.join(content)
            text = self.clean_text(text)
            
            # Close PDF document
            pdf_document.close()
            
            # Return the document content
            return {
                "content": text,
                "metadata": {
                    "title": pdf_reader.metadata.get('/Title', 'Unknown'),
                    "author": pdf_reader.metadata.get('/Author', 'Unknown'),
                    "subject": pdf_reader.metadata.get('/Subject', 'Unknown'),
                    "creator": pdf_reader.metadata.get('/Creator', 'Unknown'),
                    "producer": pdf_reader.metadata.get('/Producer', 'Unknown'),
                    "pages": len(pdf_reader.pages)
                }
            }
            
            # Return the processed text
            return text
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return None

if __name__ == "__main__":
    processor = ReportProcessor(
        "pihlajalinna-oyj-vuosiraportti-2023 (1).pdf",
        "output"
    )
    processor.process()
