from config import PDF_FILES
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return " ".join(page.extract_text() for page in reader.pages)

# Example usage:
if __name__ == "__main__":
    documents = {name: extract_text_from_pdf(path) for name, path in PDF_FILES.items()}
    for name, content in documents.items():
        print(f"Extracted content from {name}: {content[:100]}...")
