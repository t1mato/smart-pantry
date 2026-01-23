# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart Pantry is an advanced RAG application for recipe discovery from PDF cookbooks. Features **hybrid search** (BM25 + Semantic), **cross-encoder reranking**, and **RAGAS evaluation framework**.

## Architecture

### Three-Phase System
1. **Ingestion** (`ingest.py`): PDF → text chunks → local embeddings → ChromaDB
2. **Hybrid Retrieval** (`app.py`): BM25 + Semantic → RRF fusion → cross-encoder reranking (optional)
3. **Generation** (`app.py`): Context → Gemini → formatted recipe
4. **Evaluation** (`evaluation.py`): RAGAS metrics via Groq API

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

**Hybrid Search (app.py)**
- **BM25 Retriever**: Keyword-based search using `rank-bm25` library
- **Semantic Retriever**: ChromaDB vector similarity search
- **RRF Fusion**: Reciprocal Rank Fusion combines both result sets
- Equal weighting (50/50) between BM25 and semantic

**Cross-Encoder Reranking (app.py)**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Retrieves top 10 candidates from RRF fusion
- Reranks with cross-encoder scores
- Returns top 5 for LLM generation
- **Improvement**: +6.1% answer relevancy (RAGAS evaluated)

**RAGAS Evaluation (evaluation.py)**
- Framework: RAGAS v0.3 (legacy API for LangChain compatibility)
- LLM: Groq `llama-3.1-8b-instant` (fast, free tier)
- Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
- Evaluates 4 methods: Semantic Only, BM25 Only, Hybrid (RRF), Hybrid + Reranking
- **Critical Fix**: `AnswerRelevancy(strictness=1)` for Groq n=1 compatibility

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

### Run RAGAS Evaluation
```bash
# Requires GROQ_API_KEY in .env
python evaluation.py
```

**Output**:
- `evaluation_results.csv` - Raw metric scores
- `evaluation_results_report.txt` - Auto-generated summary
- Duration: ~35 minutes for 3 test cases × 4 methods

**Note:** Use legacy `ragas.metrics` API (not `ragas.metrics.collections`) for LangChain/Groq compatibility.

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
- `NUM_RESULTS = 5` - final number of results returned
- `RERANK_TOP_K = 10` - candidates for cross-encoder reranking
- `CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`

**evaluation.py:**
- `TEST_CASES = 3` - recipe queries for evaluation
- `GROQ_MODEL = "llama-3.1-8b-instant"` - fast, free tier
- `AnswerRelevancy(strictness=1)` - CRITICAL: Groq only supports n=1
- Use `ragas.metrics` (legacy), NOT `ragas.metrics.collections`

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

### RAGAS AnswerRelevancy with Groq
**Problem:** `BadRequestError: 'n' : number must be at most 1`
**Solution:** Use `AnswerRelevancy(strictness=1)` - Groq only supports n=1, but AnswerRelevancy generates N questions by default

### RAGAS Collections Incompatibility
**Problem:** `ValueError: Collections metrics only support modern InstructorLLM. Found: ChatGroq`
**Solution:** Use legacy `ragas.metrics` API (not `ragas.metrics.collections`) - collections doesn't support LangChain wrappers

### RAGAS Timeout Errors
**Problem:** `TimeoutError()` during evaluation
**Solution:** Increase `RunConfig(timeout=300)` and reduce `max_workers=2` to avoid rate limits

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
- `sentence-transformers==3.0.1` (pinned for stability, includes cross-encoder)
- `langchain-huggingface` (local embeddings)
- `langchain-google-genai` (Gemini integration)
- `langchain-groq` (Groq API for evaluation)
- `rank-bm25` (BM25 keyword search)
- `ragas` (RAG evaluation framework)
- `datasets` (RAGAS dependency)
- `streamlit` (web UI)
- `pypdf` (PDF processing)

**Important:**
- `sentence-transformers` must be 3.0.1 to avoid PyTorch compatibility issues
- Use `ragas.metrics` (legacy API), not `ragas.metrics.collections` (requires InstructorLLM)

## File Purposes

- `app.py`: Streamlit UI, hybrid search (BM25 + Semantic + RRF), cross-encoder reranking, Gemini integration
- `ingest.py`: PDF validation, text extraction, chunking, embedding generation, ChromaDB storage
- `evaluation.py`: RAGAS evaluation framework, tests 4 retrieval methods with 4 metrics
- `requirements.txt`: Pinned dependencies
- `EVALUATION_RESULTS.md`: RAGAS evaluation findings and recommendations
- `.env`: API credentials (GOOGLE_API_KEY, GROQ_API_KEY - create manually, gitignored)
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

### Speed
- **Ingestion:** ~2s for 90-page PDF
- **Query latency:** 2-5s (search + generation), +200ms with reranking
- **Storage:** ~3MB per cookbook
- **Evaluation:** ~35 minutes for 3 test cases × 4 methods

### Quality (RAGAS Evaluation Results)
| Method | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|--------|-------------------|----------------|--------------|------------------|
| **Hybrid + Reranking** | 66.8% | **100%** | 88.4% | **59.5%** ⭐ |
| Hybrid (RRF) | 73.3% | **100%** | 88.6% | 53.4% |
| Semantic Only | 79.6% | 93.3% | 86.1% | 58.7% |
| BM25 Only | 60.2% | **100%** | 75.0% | 42.0% |

**Key Findings:**
- Cross-encoder reranking achieves **best answer relevancy** (59.5%, +6.1% vs basic hybrid)
- Hybrid methods achieve **perfect recall** (100% - users won't miss recipes)
- Trade-off: Lower precision (66.8%) but better final answer quality
- **Recommendation**: Use Hybrid + Reranking for production

See `EVALUATION_RESULTS.md` for detailed analysis.

### Cost
- **Embeddings:** $0 (local HuggingFace)
- **Vector DB:** $0 (local ChromaDB)
- **Generation:** ~$0.0001/query (Gemini free tier: 1500 req/day)
- **Evaluation:** $0 (Groq free tier)
- **Total:** **Free** for typical usage

## Future Considerations

- Batch PDF ingestion (currently one at a time)
- GPU acceleration for embeddings (change `device: "cpu"` to `"cuda"`)
- Multiple cookbook support in single ingestion run
- Recipe deduplication across cookbooks
- Caching frequently requested recipes
