import logging
from typing import List, Dict, Any
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import QdrantClient
from qdrant_client.http import models
import boto3
from core.config import settings
from core.utils import DocumentProcessor

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # Create a boto3 session with credentials
        self.session = boto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )

        # Create Bedrock runtime client
        self.bedrock_runtime = self.session.client(
            service_name='bedrock-runtime',
            region_name=settings.AWS_REGION
        )

        # Create Bedrock client
        self.bedrock = self.session.client(
            service_name='bedrock',
            region_name=settings.AWS_REGION
        )

        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_runtime,
            model_id="amazon.titan-embed-text-v1"
        )
        
        self.llm = Bedrock(
            client=self.bedrock_runtime,
            model_id=settings.BEDROCK_MODEL_ID,
            model_kwargs={"temperature": 0.7, "max_tokens_to_sample": 500}
        )
        
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        
        # Initialize Qdrant collection if it doesn't exist
        self._init_qdrant_collection()
        
        # Create vector store without search_kwargs
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embeddings=self.embeddings
        )
        
        # Configure retriever with search parameters
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": settings.SEARCH_TOP_K,
                "score_threshold": settings.SEARCH_SCORE_THRESHOLD,
            }
        )
        
        # Create conversation chain with configured retriever
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=True,
            chain_type="stuff"
        )
        
        self.document_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def _init_qdrant_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if settings.QDRANT_COLLECTION_NAME not in collection_names:
                logger.info(f"Creating collection: {settings.QDRANT_COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=1536,  # Titan embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection {settings.QDRANT_COLLECTION_NAME} created successfully")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}")
            raise

    def index_documents(self, directory_path: str) -> Dict[str, Any]:
        """Index documents from the specified directory."""
        try:
            documents = self.document_processor.process_documents_directory(directory_path)
            self.vector_store.add_documents(documents)
            return {"status": "success", "message": f"Indexed {len(documents)} documents"}
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def chat(self, message: str, chat_history: List = None) -> Dict[str, Any]:
        """Process a chat message and return the response."""
        if chat_history is None:
            chat_history = []
            
        try:
            response = await self.conversation_chain.ainvoke({
                "question": message,
                "chat_history": chat_history
            })
            
            # Include similarity scores in the response
            sources_with_scores = [
                {
                    "source": doc.metadata["source"],
                    "page": doc.metadata.get("page", 1),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in response["source_documents"]
            ]
            
            return {
                "status": "success",
                "response": response["answer"],
                "sources": sources_with_scores
            }
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            return {"status": "error", "message": str(e)} 