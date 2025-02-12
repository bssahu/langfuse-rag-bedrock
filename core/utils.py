import os
from typing import List
import logging
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def process_text_file(self, file_path: str) -> List[Document]:
        """Process a text file and return chunks."""
        logger.info(f"Processing text file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            splits = self.text_splitter.split_text(text)
            return [
                Document(
                    page_content=split,
                    metadata={"source": file_path}
                ) for split in splits
            ]
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []

    def process_pdf(self, file_path: str) -> List[Document]:
        """Process a single PDF file and return chunks."""
        try:
            logger.info(f"Processing PDF file: {file_path}")
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            splits = []
            for doc in documents:
                doc_splits = self.text_splitter.split_text(doc.page_content)
                splits.extend([
                    Document(
                        page_content=split,
                        metadata={"source": file_path, "page": doc.metadata.get("page", 0)}
                    ) for split in doc_splits
                ])
            
            return splits
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            return []

    def process_image_file(self, file_path: str) -> List[Document]:
        """Process an image file using OCR and return chunks."""
        logger.info(f"Processing image file: {file_path}")
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            splits = self.text_splitter.split_text(text)
            return [
                Document(
                    page_content=split,
                    metadata={"source": file_path}
                ) for split in splits
            ]
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {str(e)}")
            return []

    def process_documents_directory(self, directory_path: str) -> List[Document]:
        """Process all PDF files in the specified directory."""
        all_documents = []
        
        try:
            # Process only PDF files
            pdf_files = [
                os.path.join(directory_path, f) 
                for f in os.listdir(directory_path) 
                if f.lower().endswith('.pdf')
            ]
            
            for pdf_file in pdf_files:
                chunks = self.process_pdf(pdf_file)
                all_documents.extend(chunks)
            
            logger.info(f"Processed {len(all_documents)} documents in total")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return [] 