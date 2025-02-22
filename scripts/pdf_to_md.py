import PyPDF2
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def detect_table(text):
    """Detect if text contains a table structure"""
    # Look for patterns that indicate a table:
    # - Multiple numbers in a row
    # - Aligned columns with spaces or tabs
    # - Common financial table headers
    number_pattern = r'\d+([,.]\d+)?\s+\d+([,.]\d+)?'
    column_pattern = r'\s{2,}|\t'
    financial_headers = ['milj. €', 'tuhatta euroa', 'MEUR', '1000 EUR', 'Liitetieto']
    
    if re.search(number_pattern, text) and re.search(column_pattern, text):
        return True
    if any(header in text for header in financial_headers):
        return True
    return False

def format_table(text):
    """Format detected table into markdown table"""
    lines = text.split('\n')
    formatted_lines = []
    
    # Add markdown table formatting
    for i, line in enumerate(lines):
        # Clean up excessive spaces while preserving column alignment
        cells = [cell.strip() for cell in re.split(r'\s{2,}|\t', line) if cell.strip()]
        if cells:
            formatted_lines.append('| ' + ' | '.join(cells) + ' |')
            # Add separator line after header
            if i == 0:
                formatted_lines.append('|' + '|'.join('-' * len(cell) for cell in cells) + '|')
    
    return '\n'.join(formatted_lines)

def clean_text(text):
    # Fix common Finnish character encoding issues
    replacements = {
        'ä': 'ä', 'ö': 'ö', 'å': 'å',
        'Ä': 'Ä', 'Ö': 'Ö', 'Å': 'Å',
        '€': '€', '—': '-'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Preserve table structures
    if detect_table(text):
        return format_table(text)
    
    # Clean up other formatting
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove extra newlines
    
    return text.strip()

def detect_section(text):
    """Detect if text is a section header"""
    section_patterns = [
        r'^\d+\s+[A-ZÄÖÅ]',  # Numbered sections
        r'^(?:Liitetieto|Note)\s+\d+',  # Notes to financial statements
        r'^(?:TASE|TULOSLASKELMA|RAHAVIRTALASKELMA)',  # Financial statements
        r'^(?:Hallituksen toimintakertomus|Tilinpäätös)'  # Main sections
    ]
    
    return any(re.match(pattern, text.strip()) for pattern in section_patterns)

def pdf_to_markdown(pdf_path, output_path):
    logging.info(f"Converting {pdf_path} to {output_path}...")
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        content = []
        current_section = None
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            
            # Split text into lines for processing
            lines = text.split('\n')
            processed_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect and format section headers
                if detect_section(line):
                    current_section = line
                    processed_lines.append(f"\n## {line}\n")
                # Detect and format tables
                elif detect_table(line):
                    processed_lines.append(clean_text(line))
                else:
                    processed_lines.append(line)
            
            # Add page marker for reference
            content.append(f"\n<!-- Page {page_num} -->\n")
            content.extend(processed_lines)
        
        # Join all content
        markdown_content = '\n'.join(content)
        
        # Write to markdown file
        with open(output_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
        
        logging.info("Done!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python pdf_to_md.py <input_pdf> <output_md>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not pdf_path.exists():
        print(f"Error: PDF file {pdf_path} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {pdf_path} to {output_path}...")
    pdf_to_markdown(pdf_path, output_path)
    print("Done!")
