# Smart Pantry & Diet Guardian

Advanced RAG-based recipe search with **hybrid retrieval** (BM25 + Semantic) and **cross-encoder reranking** for optimal recipe discovery from PDF cookbooks.

## Overview

Three-phase RAG system with rigorous RAGAS evaluation:

1. **Ingestion:** PDFs → text extraction → chunking → local embeddings → ChromaDB
2. **Hybrid Retrieval:** BM25 (keyword) + Semantic (embeddings) → RRF fusion → cross-encoder reranking
3. **Generation:** Context → Gemini LLM → formatted recipe with dietary adaptation

### Architecture

- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (local, 384-dim)
- **Hybrid Search:** BM25 + Semantic with Reciprocal Rank Fusion (RRF)
- **Reranking:** Cross-encoder `ms-marco-MiniLM-L-6-v2` (optional, +6.1% answer quality)
- **Vector Store:** ChromaDB (local SQLite persistence)
- **LLM:** Google Gemini 2.5 Flash
- **Evaluation:** RAGAS framework (Groq llama-3.1-8b for metrics)
- **Frontend:** Streamlit

### Key Features

- **Hybrid Search:** Combines keyword matching (BM25) with semantic understanding
- **Cross-Encoder Reranking:** Improves answer relevancy from 53.4% → 59.5%
- **RAGAS Evaluation:** 4 metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy)
- **Perfect Recall:** 100% - users won't miss relevant recipes
- **Debug Mode:** View raw retrieval results without LLM generation

## Requirements

- Python 3.10+ (3.9 has compatibility issues with `sentence-transformers`)
- Google AI API key (https://ai.google.dev/) - for recipe generation
- Groq API key (https://console.groq.com/) - optional, for RAGAS evaluation only

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "GOOGLE_API_KEY=<your-key>" > .env
echo "GROQ_API_KEY=<your-key>" >> .env  # Optional: for evaluation only

# Ingest PDFs
python ingest.py                      # Default: data/good-and-cheap-by-leanne-brown.pdf
python ingest.py data/cookbook.pdf    # Custom PDF

# Run application
streamlit run app.py
```

## Usage

### Running the App
```
1. Navigate to http://localhost:8501
2. Enter ingredients (e.g., "chicken, rice, bell peppers")
3. Optionally specify dietary restrictions (e.g., "gluten-free, vegetarian")
4. Toggle "Enable Cross-Encoder Reranking" (recommended: ON)
5. Toggle "Debug Mode" to view retrieval results without LLM generation
6. Submit query
7. Receive adapted recipe with source citation
```

### Running Evaluation
```bash
# Requires Groq API key in .env
python evaluation.py

# Generates:
# - evaluation_results.csv (raw scores)
# - evaluation_results_report.txt (summary)
```

## Project Structure

```
smart-pantry/
├── app.py                      # Streamlit UI + hybrid search + reranking
├── ingest.py                   # PDF processing + embedding generation
├── evaluation.py               # RAGAS evaluation framework
├── requirements.txt            # Python dependencies
├── .env                        # API credentials (gitignored)
├── data/                       # Source PDFs (gitignored)
├── chroma_db/                  # Vector database (gitignored)
├── CLAUDE.md                   # Development documentation
├── EVALUATION_RESULTS.md       # RAGAS evaluation findings
└── README.md                   # This file
```

## Configuration

### ingest.py
- `CHUNK_SIZE`: 2000
- `CHUNK_OVERLAP`: 200
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SEPARATORS`: `["\n\n", "Title:", "Ingredients:"]`

### app.py
- `GEMINI_MODEL`: "gemini-flash-latest"
- `NUM_RESULTS`: 5 (final results returned)
- `RERANK_TOP_K`: 10 (candidates for reranking)
- `CROSS_ENCODER_MODEL`: "cross-encoder/ms-marco-MiniLM-L-6-v2"
- `EMBEDDING_MODEL`: Must match ingest.py

### evaluation.py
- `TEST_CASES`: 3 recipe queries (expand to 20+ for production)
- `GROQ_MODEL`: "llama-3.1-8b-instant"
- Evaluates 4 methods: Semantic Only, BM25 Only, Hybrid (RRF), Hybrid + Reranking

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Vector database not found` | Run `python ingest.py` |
| `GOOGLE_API_KEY not found` | Create `.env` file |
| `404 models/gemini-* not found` | Use `gemini-flash-latest` |
| `Cannot copy out of meta tensor` | `pip install sentence-transformers==3.0.1` |
| Python 3.9 import errors | Upgrade to Python 3.10+ |

## Performance

### Speed
- **Ingestion**: ~2s for 90-page PDF
- **Query latency**: 2-5s (search + generation), +200ms with reranking
- **Storage**: ~3MB per cookbook

### Quality (RAGAS Evaluation)
- **Answer Relevancy**: 59.5% (with reranking, +6.1% improvement)
- **Context Recall**: 100% (perfect - won't miss recipes)
- **Faithfulness**: 88.4% (high answer grounding)
- **Context Precision**: 66.8% (acceptable noise/signal ratio)

See `EVALUATION_RESULTS.md` for detailed analysis.

### Cost
- **Embeddings**: $0 (local)
- **Vector DB**: $0 (local)
- **Generation**: ~$0.0001/query (Gemini free tier: 1500 req/day)
- **Total**: **Free** for typical usage

## License

MIT
