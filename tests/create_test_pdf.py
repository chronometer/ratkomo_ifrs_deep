"""
Create a test PDF file for document processing tests
"""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pathlib import Path

def create_test_pdf():
    # Get the test data directory
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Create PDF
    pdf_path = test_data_dir / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    
    # Add some test content
    c.setFont("Helvetica", 12)
    
    # Add title
    c.drawString(72, 750, "IFRS Test Document")
    
    # Add some paragraphs
    paragraphs = [
        "1. Revenue Recognition (IFRS 15)",
        "The company recognizes revenue when control of goods or services is transferred to the customer.",
        "",
        "2. Leases (IFRS 16)",
        "All significant leases are recognized on the balance sheet as right-of-use assets.",
        "",
        "3. Financial Statements (IAS 1)",
        "The financial statements are prepared in accordance with International Financial Reporting Standards."
    ]
    
    y_position = 700
    for para in paragraphs:
        c.drawString(72, y_position, para)
        y_position -= 20
    
    # Save the PDF
    c.save()
    
    return pdf_path

if __name__ == "__main__":
    create_test_pdf()
