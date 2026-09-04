# Automated RAG Pipeline

A Retrieval-Augmented Generation (RAG) project that ingests PDF documents and web pages, stores embeddings in ChromaDB, retrieves relevant context using semantic search, and generates answers using Gemini.

## Features

- Automated PDF ingestion
- Web URL ingestion
- Fast PDF text extraction
- Recursive document chunking
- Hugging Face embeddings
- ChromaDB vector storage
- MMR-based semantic retrieval
- Gemini-powered answer generation
- Retrieval evaluation
- Retrieved source metadata

## Project Structure

```text
RAG/
├── .gitignore
├── .env
├── requirements.txt
├── README.md
├── data/
│   └── documents/
│       └── .gitkeep
├── ChromaDB/
└── src/
    ├── initiator.py
    ├── auto_ingestion_pipeline.py
    ├── auto_retrieval_pipeline.py
    └── auto_rag_evaluator.py
```

## Pipeline Flow

```text
PDF / URL
    ↓
Document Ingestion
    ↓
Text Extraction
    ↓
Chunking
    ↓
Hugging Face Embeddings
    ↓
ChromaDB
    ↓
MMR Semantic Retrieval
    ↓
Relevant Context
    ↓
Gemini
    ↓
Generated Answer
    ↓
Retrieval Evaluation
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_gemini_api_key
```

Do not commit `.env` to Git.

## Adding Documents

Place PDF files inside:

```text
data/documents/
```

Update `FILES_TO_INGEST` in `src/initiator.py`.

Web pages can be added to the `URLS_TO_INGEST` list.

## Running the Project

From the project root:

```bash
python src/initiator.py
```

The application performs:

1. Source validation
2. PDF and URL ingestion
3. Document chunking
4. Embedding generation
5. ChromaDB storage
6. Semantic retrieval
7. Gemini answer generation
8. Retrieval evaluation

## Retrieval

The retrieval pipeline uses Maximal Marginal Relevance (MMR) to retrieve relevant and diverse document chunks.

The retrieved context is passed to Gemini to generate an answer using the available document context.

## Evaluation

The project includes a basic retrieval evaluation pipeline that:

- Runs predefined queries
- Retrieves relevant chunks
- Checks for expected text snippets
- Calculates a simple query-document similarity score

The evaluation summary displays total tests, successful retrievals, accuracy, and average similarity.

## Configuration

### Chunking

```python
chunk_size=1000
chunk_overlap=150
```

### Retrieval

```python
search_type="mmr"
k=5
fetch_k=25
lambda_mult=0.5
```

### LLM

```python
model="gemini-2.5-flash"
temperature=0
```

## Technologies Used

- Python
- LangChain
- ChromaDB
- Hugging Face
- Sentence Transformers
- Google Gemini
- Unstructured
- BeautifulSoup

## Notes

- `ChromaDB/` is generated locally and should not be committed.
- API keys should remain in `.env`.
- Delete the existing `ChromaDB/` directory when rebuilding the vector store after major chunking or embedding changes.

## Future Improvements

- Additional file formats
- Better metadata and citations
- Answer quality evaluation
- Conversation memory
- Interactive chat interface
- Hybrid search
- Reranking
- Docker support
