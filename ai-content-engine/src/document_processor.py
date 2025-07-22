import os
from typing import List
from pypdf import PdfReader
from langchain.docstore.document import Document

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str, chunk_size: int = 1000) -> List[Document]:
        """
        Extract text from PDF, splitting into manageable chunks
        
        Args:
            pdf_path (str): Path to PDF file
            chunk_size (int): Size of text chunks
        
        Returns:
            List of Document objects
        """
        reader = PdfReader(pdf_path)
        documents = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            # Split text into chunks
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path,
                        "page": page_num
                    }
                )
                documents.append(doc)
        
        return documents

    @classmethod
    def process_documents(cls, document_directory: str) -> List[Document]:
        """
        Process all PDF documents in a directory
        
        Args:
            document_directory (str): Path to directory containing PDFs
        
        Returns:
            List of processed Document objects
        """
        all_documents = []
        
        for filename in os.listdir(document_directory):
            if filename.endswith('.pdf'):
                filepath = os.path.join(document_directory, filename)
                documents = cls.extract_text_from_pdf(filepath)
                all_documents.extend(documents)
        
        return all_documents