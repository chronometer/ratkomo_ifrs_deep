"""
Utilities for handling PDF files
"""
import logging
from pathlib import Path
from typing import Optional
import PyPDF2

def pdf_to_text(pdf_path: Path, output_path: Optional[Path] = None) -> Path:
    """Convert PDF to text file"""
    if not output_path:
        output_path = pdf_path.with_suffix('.txt')
    
    logging.info(f"Converting PDF {pdf_path} to text {output_path}")
    
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            
            # Write text to file
            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write('\n\n'.join(text))
            
            logging.info(f"Successfully converted PDF to text: {output_path}")
            return output_path
    
    except Exception as e:
        logging.error(f"Error converting PDF to text: {e}")
        raise
