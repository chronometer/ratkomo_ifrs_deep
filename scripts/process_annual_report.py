"""
Enhanced PDF to Markdown converter optimized for Finnish annual reports
"""
import PyPDF2
import re
from pathlib import Path
import fitz  # PyMuPDF
import io
import os
from PIL import Image
import numpy as np

class AnnualReportConverter:
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
        # Remove multiple newlines and spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(?<=[a-zäöå])-\s*\n(?=[a-zäöå])', '', text)  # Join hyphenated words
        text = re.sub(r'([a-zäöå])\n([a-zäöå])', r'\1 \2', text)  # Join broken words
        
        # Format section headers (common in Finnish annual reports)
        section_patterns = [
            (r'\n(?:HALLITUKSEN\s+)?TOIMINTAKERTOMUS\s*\n', r'\n## Toimintakertomus\n'),
            (r'\n(?:KONSERNIN\s+)?TILINPÄÄTÖS\s*\n', r'\n## Tilinpäätös\n'),
            (r'\nVASTUULLISUUS(?:RAPORTTI)?\s*\n', r'\n## Vastuullisuus\n'),
            (r'\nHALLINNOINTI(?:RAPORTTI)?\s*\n', r'\n## Hallinnointi\n'),
            (r'\nTIETOJA\s+OSAKKEENOMISTAJILLE\s*\n', r'\n## Tietoja osakkeenomistajille\n'),
            (r'\nTILINTARKASTUSKERTOMUS\s*\n', r'\n## Tilintarkastuskertomus\n'),
        ]
        
        for pattern, replacement in section_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Format subsection headers
        text = re.sub(r'\n([A-ZÄÖÅ][A-ZÄÖÅ\s]{5,})\n', r'\n### \1\n', text)
        
        return text.strip()

    def extract_sections(self, text: str) -> dict:
        """Extract major sections of the annual report"""
        sections = {
            'toimintakertomus': '',
            'tilinpaatos': '',
            'vastuullisuus': '',
            'hallinnointi': '',
            'tietoja_osakkeenomistajille': '',
            'tilintarkastuskertomus': ''
        }
        
        # Find section boundaries
        section_matches = list(re.finditer(r'\n##\s+(.*?)\n', text))
        
        for i, match in enumerate(section_matches):
            section_name = match.group(1).lower()
            start = match.end()
            end = section_matches[i + 1].start() if i < len(section_matches) - 1 else len(text)
            
            # Map section names
            if 'toimintakertomus' in section_name:
                sections['toimintakertomus'] = text[start:end].strip()
            elif 'tilinpäätös' in section_name:
                sections['tilinpaatos'] = text[start:end].strip()
            elif 'vastuullisuus' in section_name:
                sections['vastuullisuus'] = text[start:end].strip()
            elif 'hallinnointi' in section_name:
                sections['hallinnointi'] = text[start:end].strip()
            elif 'osakkeenomistaj' in section_name:
                sections['tietoja_osakkeenomistajille'] = text[start:end].strip()
            elif 'tilintarkastus' in section_name:
                sections['tilintarkastuskertomus'] = text[start:end].strip()
        
        return sections

    def convert(self):
        """Convert annual report PDF to enhanced markdown"""
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
        markdown = f"""# Pihlajalinna Oyj Vuosiraportti 2023

## Toimintakertomus
{sections['toimintakertomus']}

## Tilinpäätös
{sections['tilinpaatos']}

## Vastuullisuus
{sections['vastuullisuus']}

## Hallinnointi
{sections['hallinnointi']}

## Tietoja osakkeenomistajille
{sections['tietoja_osakkeenomistajille']}

## Tilintarkastuskertomus
{sections['tilintarkastuskertomus']}

## Kuvat
"""
        
        # Add images with page references
        for i, image_path in enumerate(image_paths, 1):
            page_num = int(re.search(r'page_(\d+)_', image_path).group(1))
            markdown += f"\n### Kuva {i} (sivu {page_num})\n![Kuva {i}]({image_path})\n"
        
        # Write markdown file
        output_file = self.output_dir / "vuosiraportti.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        # Close PDF document
        pdf_document.close()

if __name__ == "__main__":
    converter = AnnualReportConverter(
        "pihlajalinna-oyj-vuosiraportti-2023 (1).pdf",
        "output"
    )
    converter.convert()
