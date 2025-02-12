import uvicorn
from fastapi import FastAPI
from api.endpoints import router
from core.config import settings
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="A RAG-based chatbot using LangChain, Qdrant, and Claude",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting RAG Chatbot API")
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    ) 