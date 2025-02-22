"""
PDF extractor using PyMuPDF (fitz)
"""
import fitz
import os
from pathlib import Path
import re

def extract_pdf(pdf_path: str, output_dir: str):
    """Extract text and images from PDF"""
    # Create output directory
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    # Extract text and images
    full_text = []
    image_refs = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text
        text = page.get_text()
        full_text.append(text)
        
        # Extract images
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Save image
            image_ext = base_image.get("ext", "png")
            image_path = f"images/page_{page_num + 1}_image_{img_index + 1}.{image_ext}"
            image_file = output_dir / image_path
            
            with open(image_file, "wb") as img_file:
                img_file.write(image_bytes)
            
            image_refs.append((page_num + 1, image_path))
    
    # Clean and format text
    text = "\n".join(full_text)
    
    # Basic cleaning
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove multiple newlines
    text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)  # Join hyphenated words
    
    # Format markdown
    markdown = f"""# Chain of Agents: Large Language Models Collaborating on Long-Context Tasks

{text}

## Figures

"""
    
    # Add image references
    for page_num, image_path in image_refs:
        markdown += f"\n### Image from page {page_num}\n![Page {page_num}]({image_path})\n"
    
    # Write markdown file
    output_file = output_dir / "paper_complete.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    # Close document
    doc.close()

if __name__ == "__main__":
    extract_pdf(
        "docs/19782_Chain_of_Agents_Large_La.pdf",
        "docs"
    )
