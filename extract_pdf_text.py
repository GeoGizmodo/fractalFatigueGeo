import PyPDF2
import os
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading {pdf_path}: {str(e)}"

def extract_all_pdfs():
    """Extract text from all PDF files in current directory."""
    current_dir = Path(".")
    pdf_files = list(current_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in current directory")
        return
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        print("=" * 50)
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        
        # Save to text file
        output_file = pdf_file.with_suffix('.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Text extracted and saved to: {output_file}")
        print(f"Preview (first 200 chars):\n{text[:200]}...")

if __name__ == "__main__":
    extract_all_pdfs()