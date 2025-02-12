from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
import shutil
from datetime import datetime
from core.services import RAGService
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
rag_service = RAGService()

class ChatRequest(BaseModel):
    message: str
    chat_history: List = []

class IndexRequest(BaseModel):
    directory_path: str = "documents"

@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    Chat endpoint that processes messages and returns responses
    """
    logger.info(f"Received chat request: {request.message}")
    response = await rag_service.chat(request.message, request.chat_history)
    
    if response["status"] == "error":
        raise HTTPException(status_code=500, detail=response["message"])
    
    return response

@router.post("/index")
async def index_endpoint(request: IndexRequest) -> Dict[str, Any]:
    """
    Index endpoint that processes documents in the specified directory
    """
    logger.info(f"Received index request for directory: {request.directory_path}")
    response = rag_service.index_documents(request.directory_path)
    
    if response["status"] == "error":
        raise HTTPException(status_code=500, detail=response["message"])
    
    return response

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Upload documents to the documents directory and index them.
    """
    try:
        # Create documents directory if it doesn't exist
        os.makedirs("documents", exist_ok=True)
        
        # Save uploaded files with timestamp to avoid name conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        for file in files:
            # Only allow PDF files
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF file"
                )
            
            # Create unique filename with timestamp
            filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join("documents", filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_files.append(file_path)
            
        # Index the documents
        result = rag_service.index_documents("documents")
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Error indexing documents: {result['message']}"
            )
        
        return {
            "status": "success",
            "message": f"Successfully uploaded and indexed {len(saved_files)} documents",
            "files": saved_files
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading documents: {str(e)}"
        ) 