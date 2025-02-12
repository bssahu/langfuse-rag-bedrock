# RAG Chatbot with AWS Bedrock and Qdrant with Langfuse for observability

A Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, AWS Bedrock, and Qdrant vector database. The system processes PDF documents and uses them to provide context-aware responses using Claude.

## Features

- PDF document processing and indexing
- Semantic search using Qdrant vector database
- Context-aware responses using Claude via AWS Bedrock
- Document chunking with configurable size and overlap
- REST API endpoints for chat and document management
- Docker containerization for easy deployment
- Configurable search parameters (top-k, similarity threshold)

## Prerequisites

- Docker and Docker Compose
- AWS Account with Bedrock access
- AWS Access Key and Secret Key

## Quick Start

1. Clone the repository:

```bash
git clone [https://github.com/bssahu/langfuse-rag-bedrock](https://github.com/bssahu/langfuse-rag-bedrock.git
cd rag-chatbot
```

2. Copy the example environment file and update with your credentials:

```bash
cp .env.example .env
```

3. Update the following variables in `.env`:
```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=your-region
```

4. Start the services:
```bash
docker-compose up --build
```

## API Endpoints

### 1. Chat Endpoint
```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{
    "message": "Your question here",
    "chat_history": []
}'
```

### 2. Document Indexing
```bash
curl -X POST "http://localhost:8000/index" \
-H "Content-Type: application/json" \
-d '{
    "directory_path": "documents"
}'
```

### 3. Document Upload
```bash
curl -X POST "http://localhost:8000/upload" \
-F "files=@/path/to/your/document.pdf"
```

## Configuration

The system can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| BEDROCK_MODEL_ID | AWS Bedrock model identifier | anthropic.claude-v2 |
| CHUNK_SIZE | Document chunk size | 1000 |
| CHUNK_OVERLAP | Overlap between chunks | 200 |
| SEARCH_TOP_K | Number of similar documents to retrieve | 3 |
| SEARCH_SCORE_THRESHOLD | Minimum similarity score | 0.7 |

## Project Structure

```
.
├── api/
│   └── endpoints.py    # API route definitions
├── core/
│   ├── config.py      # Configuration management
│   ├── services.py    # Core RAG service
│   └── utils.py       # Document processing utilities
├── documents/         # Document storage directory
├── .env              # Environment variables
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Architecture

1. **Document Processing**:
   - PDF documents are processed using PyMuPDF
   - Documents are split into chunks with configurable size and overlap
   - Each chunk is embedded using AWS Bedrock's Titan embedding model

2. **Vector Storage**:
   - Document embeddings are stored in Qdrant
   - Semantic search is performed using cosine similarity
   - Configurable search parameters for retrieval

3. **Chat Processing**:
   - User queries are processed using Claude
   - Relevant context is retrieved from Qdrant
   - Responses include source documents and metadata

## Error Handling

The system includes comprehensive error handling for:
- Document processing failures
- AWS Bedrock API errors
- Vector store operations
- Invalid file types
- Storage space issues

## Logging

Detailed logging is implemented throughout the application:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License


This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.


## Support

For support, please open an issue in the GitHub repository.
