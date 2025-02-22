"""
Enhanced PDF to Markdown converter with image extraction and better formatting
"""
import PyPDF2
import re
from pathlib import Path
import fitz  # PyMuPDF
import io
import os
from PIL import Image
import numpy as np

class PDFConverter:
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
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)  # Join hyphenated words
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)  # Join broken words
        
        # Format section headers
        text = re.sub(r'\n(\d+\..*)\n', r'\n## \1\n', text)
        text = re.sub(r'\n(\d+\.\d+\..*)\n', r'\n### \1\n', text)
        
        # Format references
        text = re.sub(r'\[(\d+)\]', r'[[\\1]](#ref-\\1)', text)
        
        return text.strip()

    def extract_sections(self, text: str) -> dict:
        """Extract major sections of the paper"""
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'experiments': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'references': ''
        }
        
        # Find section boundaries
        section_matches = list(re.finditer(r'\n##\s+(.*?)\n', text))
        
        for i, match in enumerate(section_matches):
            section_name = match.group(1).lower()
            start = match.end()
            end = section_matches[i + 1].start() if i < len(section_matches) - 1 else len(text)
            
            # Map section names to our structure
            if 'abstract' in section_name:
                sections['abstract'] = text[start:end].strip()
            elif 'introduction' in section_name or section_name.startswith('1'):
                sections['introduction'] = text[start:end].strip()
            elif 'method' in section_name or 'approach' in section_name:
                sections['methodology'] = text[start:end].strip()
            elif 'experiment' in section_name or 'evaluation' in section_name:
                sections['experiments'] = text[start:end].strip()
            elif 'result' in section_name or 'finding' in section_name:
                sections['results'] = text[start:end].strip()
            elif 'discussion' in section_name:
                sections['discussion'] = text[start:end].strip()
            elif 'conclusion' in section_name:
                sections['conclusion'] = text[start:end].strip()
            elif 'reference' in section_name:
                sections['references'] = text[start:end].strip()
        
        return sections

    def convert(self):
        """Convert PDF to enhanced markdown"""
        # Open PDF with both PyPDF2 and PyMuPDF
        pdf_reader = PyPDF2.PdfReader(self.pdf_path)
        pdf_document = fitz.open(self.pdf_path)
        
        # Extract images
        image_paths = self.extract_images(pdf_document)
        
        # Extract text
        content = []
        for page in pdf_reader.pages:
            content.append(page.extract_text())
        
        # Clean and format text
        text = '\n'.join(content)
        text = self.clean_text(text)
        
        # Extract sections
        sections = self.extract_sections(text)
        
        # Generate markdown
        markdown = f"""# {pdf_reader.metadata.get('/Title', 'Chain of Agents Paper')}

## Authors
{pdf_reader.metadata.get('/Author', '')}

## Abstract
{sections['abstract']}

## Introduction
{sections['introduction']}

## Methodology
{sections['methodology']}

## Experiments
{sections['experiments']}

## Results
{sections['results']}

## Discussion
{sections['discussion']}

## Conclusion
{sections['conclusion']}

## References
{sections['references']}

## Figures
"""
        
        # Add images
        for i, image_path in enumerate(image_paths, 1):
            markdown += f"\n### Figure {i}\n![Figure {i}]({image_path})\n"
        
        # Write markdown file
        output_file = self.output_dir / "paper_full.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        # Close PDF document
        pdf_document.close()

if __name__ == "__main__":
    converter = PDFConverter(
        "docs/19782_Chain_of_Agents_Large_La.pdf",
        "docs"
    )
    converter.convert()
