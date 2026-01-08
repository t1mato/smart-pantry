# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart Pantry is a RAG application for recipe discovery from PDF cookbooks. It uses local HuggingFace embeddings for vector search and Google Gemini for recipe adaptation.

## Architecture

### Two-Phase System
1. **Ingestion** (`ingest.py`): PDF → text chunks → local embeddings → ChromaDB
2. **Query** (`app.py`): user query → vector search → context → Gemini → formatted recipe

### Key Components

**Vector Database (ChromaDB)**
- Local SQLite persistence at `./chroma_db`
- Stores 384-dim embeddings from HuggingFace `all-MiniLM-L6-v2`
- Chunks: 2000 chars with 200-char overlap

**Embeddings Strategy**
- **Critical:** Uses local HuggingFace embeddings, NOT Google Embedding API
- Reason: Google's free tier has strict rate limits (hit 429 errors immediately)
- Model: `all-MiniLM-L6-v2` - fast, lightweight (~120MB), sufficient for recipe matching
- Both `ingest.py` and `app.py` MUST use identical embedding model

**LLM Integration**
- Google Gemini 2.5 Flash via `langchain-google-genai`
- Model: `gemini-flash-latest` (automatically uses newest available)
- Used only for recipe generation, NOT embeddings
- API key: `GOOGLE_API_KEY` in `.env`

**Document Processing**
- PDFs loaded via `PyPDFLoader` from `data/` directory
- `RecursiveCharacterTextSplitter` with recipe-specific separators: `["\n\n", "Title:", "Ingredients:"]`
- Preserves recipe structure (title, ingredients, instructions)

## Development Commands

### Environment Setup
```bash
# Use Python 3.10+ (3.9 has compatibility issues)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Ingestion
```bash
# Process default PDF
python ingest.py

# Process specific PDF
python ingest.py data/cookbook.pdf
```

**Output:** Creates/updates `./chroma_db/` with embedded recipe chunks

### Run Application
```bash
streamlit run app.py
```

**URL:** http://localhost:8501

### Reset Database
```bash
rm -rf chroma_db/
python ingest.py
```

## Configuration

### Critical Settings

**ingest.py:**
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` - must match app.py
- `CHUNK_SIZE = 2000`
- `CHUNK_OVERLAP = 200`
- `CHUNK_SEPARATORS = ["\n\n", "Title:", "Ingredients:"]`

**app.py:**
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` - must match ingest.py
- `GEMINI_MODEL = "gemini-flash-latest"`
- `NUM_RESULTS = 5` - number of chunks retrieved

### Environment Variables
- `GOOGLE_API_KEY` - required for Gemini, stored in `.env` (gitignored)
- NOT required for ingestion (uses local embeddings)

## Important Implementation Details

### Embedding Model Consistency
**Critical:** Both scripts must use identical embedding model. Mismatched embeddings result in nonsensical similarity scores.

```python
# Both ingest.py and app.py use:
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Chunk Configuration
- `chunk_size=2000`: Balances keeping recipes together vs staying within model limits
- `chunk_overlap=200`: Prevents splitting mid-recipe
- Separators preserve recipe structure for better retrieval

### Error Handling
- `ingest.py`: Validates PDF existence, progress bars via `tqdm`, graceful failures
- `app.py`: Checks database existence, validates API key, handles search failures

### Model Name Format
- LangChain Google GenAI 4.x uses model names without `models/` prefix
- Use `gemini-flash-latest` not `models/gemini-1.5-flash`
- Available models queryable via Google GenAI API

## Common Issues & Solutions

### Python Version
**Problem:** Python 3.9 has `importlib.metadata` and package compatibility issues
**Solution:** Use Python 3.10 or higher

### Sentence-Transformers Error
**Problem:** `Cannot copy out of meta tensor` with v5.2.0
**Solution:** Pinned to v3.0.1 in requirements.txt

### Gemini 404 Errors
**Problem:** `models/gemini-*` not found
**Solution:** Use `gemini-flash-latest` (newer API versions)

### Import Errors
**Problem:** `ModuleNotFoundError: langchain.prompts`
**Solution:** Use `langchain_core.prompts` (LangChain 0.1.0+)

### Rate Limits
**Problem:** Google Embedding API 429 errors
**Solution:** Using local HuggingFace embeddings (already implemented)

## Design Rationale

### Why Local Embeddings?
- Google Embedding API has strict free tier limits (0 quota in our testing)
- Embeddings are bulk operations (expensive at scale with APIs)
- One-time compute cost, then unlimited local searches
- Trade-off: 384-dim vs Google's 768-dim, but sufficient for recipe matching

### Why Gemini for Generation?
- LLM queries are low-volume (per user search)
- Free tier covers ~1500 requests/day
- Excellent instruction following for recipe adaptation
- Latest model (2.5 Flash) is fast and capable

### Code Organization
- Configuration constants at top of files
- Helper functions with comprehensive docstrings
- Main execution in `main()` function
- Extensive inline comments for complex logic

## Dependencies

**Key packages (see requirements.txt for versions):**
- `langchain` + integrations (core framework)
- `chromadb` + `langchain-chroma` (vector store)
- `sentence-transformers==3.0.1` (pinned for stability)
- `langchain-huggingface` (local embeddings)
- `langchain-google-genai` (Gemini integration)
- `streamlit` (web UI)
- `pypdf` (PDF processing)
- `tqdm` (progress bars)

**Important:** `sentence-transformers` must be 3.0.1 to avoid PyTorch compatibility issues

## File Purposes

- `app.py`: Streamlit UI, vector search, Gemini integration, recipe formatting
- `ingest.py`: PDF validation, text extraction, chunking, embedding generation, ChromaDB storage
- `requirements.txt`: Pinned dependencies
- `.env`: API credentials (create manually, gitignored)
- `data/`: PDF cookbook storage (gitignored)
- `chroma_db/`: Vector database (gitignored)

## Testing

**Verify installation:**
```python
python -c "from langchain_huggingface import HuggingFaceEmbeddings; print('OK')"
python -c "from langchain_google_genai import ChatGoogleGenerativeAI; print('OK')"
```

**Test ingestion:**
```bash
python ingest.py
# Should complete in ~2 seconds for 90-page PDF
```

**Test embeddings:**
```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
test = embeddings.embed_query("test")
print(f"Embedding dimension: {len(test)}")  # Should be 384
```

**Test vector store:**
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
results = vectorstore.similarity_search("chicken recipe", k=2)
print(f"Found {len(results)} results")
```

## Performance Characteristics

- **Ingestion:** ~2s for 90-page PDF
- **Query latency:** 2-5s (vector search + Gemini generation)
- **Storage:** ~3MB per cookbook
- **Costs:** $0 embeddings, ~$0.0001 per query
- **Free tier:** 1500 Gemini requests/day

## Future Considerations

- Batch PDF ingestion (currently one at a time)
- GPU acceleration for embeddings (change `device: "cpu"` to `"cuda"`)
- Multiple cookbook support in single ingestion run
- Recipe deduplication across cookbooks
- Caching frequently requested recipes
