# Smart Pantry & Diet Guardian

RAG-based recipe search application that queries PDF cookbooks by ingredients and adapts recipes to dietary restrictions using Google Gemini.

## Overview

This application implements a two-phase retrieval-augmented generation system for recipe discovery:

1. **Ingestion:** PDFs → text extraction → chunking → local embeddings → ChromaDB
2. **Query:** user input → semantic search → context retrieval → LLM adaptation → formatted recipe

### Architecture

- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (local, 384-dim) - avoids API rate limits
- **Vector Store:** ChromaDB (local SQLite persistence)
- **LLM:** Google Gemini 2.5 Flash via `langchain-google-genai`
- **Frontend:** Streamlit
- **Document Processing:** PyPDF + RecursiveCharacterTextSplitter

### Key Design Decisions

- Local embeddings chosen after hitting Google Embedding API rate limits (429 errors)
- Gemini used only for generation (low volume, stays in free tier)
- Recipe-specific text splitting with separators: `["\n\n", "Title:", "Ingredients:"]`
- Chunk size: 2000 chars, overlap: 200 chars

## Requirements

- Python 3.10+ (3.9 has `importlib.metadata` and `sentence-transformers` compatibility issues)
- Google AI API key (https://ai.google.dev/)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "GOOGLE_API_KEY=<your-key>" > .env

# Ingest PDFs
python ingest.py                      # Default: data/good-and-cheap-by-leanne-brown.pdf
python ingest.py data/cookbook.pdf    # Custom PDF

# Run application
streamlit run app.py
```

## Usage

```
1. Navigate to http://localhost:8501
2. Enter ingredients (e.g., "chicken, rice, bell peppers")
3. Optionally specify dietary restrictions (e.g., "gluten-free, vegetarian")
4. Submit query
5. Receive adapted recipe with source citation
```

## Project Structure

```
smart-pantry/
├── app.py              # Streamlit UI + query logic
├── ingest.py           # PDF processing + embedding generation
├── requirements.txt    # Python dependencies
├── .env               # API credentials (gitignored)
├── data/              # Source PDFs (gitignored)
├── chroma_db/         # Vector database (gitignored)
└── CLAUDE.md          # Development documentation
```

## Configuration

### ingest.py
- `CHUNK_SIZE`: 2000
- `CHUNK_OVERLAP`: 200
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SEPARATORS`: `["\n\n", "Title:", "Ingredients:"]`

### app.py
- `GEMINI_MODEL`: "gemini-flash-latest"
- `NUM_RESULTS`: 5 (vector search k value)
- `EMBEDDING_MODEL`: Must match ingest.py

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Vector database not found` | Run `python ingest.py` |
| `GOOGLE_API_KEY not found` | Create `.env` file |
| `404 models/gemini-* not found` | Use `gemini-flash-latest` |
| `Cannot copy out of meta tensor` | `pip install sentence-transformers==3.0.1` |
| Python 3.9 import errors | Upgrade to Python 3.10+ |

## Performance

- Ingestion: ~2s for 90-page PDF
- Query latency: 2-5s (vector search + LLM generation)
- Storage: ~3MB per cookbook in ChromaDB
- Cost: $0 embeddings, ~$0.0001 per query (Gemini free tier: 1500 req/day)

## License

MIT
