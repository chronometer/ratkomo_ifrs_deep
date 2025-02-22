"""
Text processing utilities for document analysis
"""
from typing import List, Tuple
import re

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove unicode control characters
    text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
    return text.strip()

def extract_sections(text: str) -> List[Tuple[str, str]]:
    """Extract sections and their titles from text"""
    sections = []
    current_title = ""
    current_content = []
    
    lines = text.split('\n')
    for line in lines:
        # Heuristic for section titles:
        # - All caps
        # - Short (less than 100 chars)
        # - Doesn't end with common sentence endings
        clean_line = line.strip()
        if (clean_line.isupper() and 
            len(clean_line) < 100 and 
            not clean_line.endswith(('.', ':', ';', '?', '!'))):
            if current_title and current_content:
                sections.append((
                    current_title, 
                    '\n'.join(current_content).strip()
                ))
            current_title = clean_line
            current_content = []
        else:
            current_content.append(clean_line)
    
    # Add the last section
    if current_title and current_content:
        sections.append((
            current_title, 
            '\n'.join(current_content).strip()
        ))
    
    return sections

def estimate_page_numbers(text: str, chars_per_page: int = 3000) -> List[int]:
    """Estimate page numbers for text segments"""
    pages = []
    total_chars = 0
    
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        total_chars += len(para)
        pages.append(total_chars // chars_per_page)
    
    return pages

def extract_financial_metrics(text: str) -> List[Tuple[str, str]]:
    """Extract financial metrics and their values"""
    # Common financial metric patterns
    patterns = [
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|M|B|€|$|£)',  # Currency amounts
        r'(\d+(?:\.\d+)?)\s*(%)',  # Percentages
        r'((?:€|$|£)\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # Currency with symbol
    ]
    
    metrics = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end]
            metrics.append((match.group(), context))
    
    return metrics
