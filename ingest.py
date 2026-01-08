"""
Smart Pantry - Document Ingestion Script

This script processes PDF cookbooks and stores them in a local vector database (ChromaDB).
It uses local embeddings (HuggingFace) to avoid API rate limits and costs.

Usage:
    python ingest.py [pdf_path]

    If no path is provided, defaults to: data/good-and-cheap-by-leanne-brown.pdf

Examples:
    python ingest.py data/my-cookbook.pdf
    python ingest.py  # Uses default PDF
"""

import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables (for future API keys if needed)
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default PDF path - can be overridden via command line argument
DEFAULT_PDF_PATH = "data/good-and-cheap-by-leanne-brown.pdf"

# Vector database storage location
CHROMA_DB_PATH = "./chroma_db"

# Text splitting configuration
# - chunk_size: Maximum characters per chunk (larger = more context, but less precise matching)
# - chunk_overlap: Characters shared between chunks (prevents splitting mid-recipe)
# - separators: Split preferentially on these patterns to preserve recipe structure
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "Title:", "Ingredients:"]

# Embedding model configuration
# all-MiniLM-L6-v2: Fast, lightweight, runs locally, no API costs
# - 384 dimensions (vs Google's 768, but sufficient for recipe matching)
# - ~120MB model size (downloads on first run)
# - Good balance of speed and quality for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Batch size for processing chunks (lower = less memory, more progress updates)
BATCH_SIZE = 50


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_pdf_path(pdf_path):
    """
    Validates that the PDF file exists and is readable.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        Path: Validated Path object

    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If file is not a PDF
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    if path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")

    return path


def load_pdf_documents(pdf_path):
    """
    Loads a PDF file and extracts text content page by page.

    Args:
        pdf_path (Path): Path to the PDF file

    Returns:
        list: List of Document objects (one per page)

    Raises:
        Exception: If PDF loading fails
    """
    print(f"\n📖 Loading PDF: {pdf_path}")

    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        print(f"✓ Loaded {len(docs)} pages")
        return docs

    except Exception as e:
        raise Exception(f"Failed to load PDF: {e}")


def split_documents(docs):
    """
    Splits documents into smaller chunks optimized for recipe retrieval.

    The splitter:
    1. Tries to split on double newlines first (paragraph boundaries)
    2. Then tries recipe-specific markers (Title:, Ingredients:)
    3. Falls back to character count if needed
    4. Overlaps chunks to avoid cutting recipes in half

    Args:
        docs (list): List of Document objects

    Returns:
        list: List of smaller Document chunks
    """
    print(f"\n✂️  Splitting documents into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} characters")
    print(f"   Overlap: {CHUNK_OVERLAP} characters")
    print(f"   Separators: {CHUNK_SEPARATORS}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS
    )

    splits = text_splitter.split_documents(docs)
    print(f"✓ Created {len(splits)} chunks")

    return splits


def initialize_embeddings():
    """
    Initializes the local embedding model.

    This uses HuggingFace's sentence-transformers library to run embeddings
    locally. The model will be downloaded to ~/.cache/huggingface on first run.

    Benefits:
    - No API costs
    - No rate limits
    - Data privacy (nothing sent to external servers)
    - Offline capability

    Returns:
        HuggingFaceEmbeddings: Embedding model instance
    """
    print(f"\n🤖 Initializing local embedding model: {EMBEDDING_MODEL}")
    print("   (First run will download ~120MB model)")

    # Set model_kwargs to use CPU (change to {"device": "cuda"} if GPU available)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # Normalize for better similarity search
    )

    print("✓ Embedding model ready")
    return embeddings


def store_in_vectordb(splits, embeddings):
    """
    Stores document chunks in ChromaDB with embeddings.

    This creates vector embeddings for each chunk and stores them in a local
    SQLite database. The embeddings enable semantic search (finding recipes by
    meaning, not just keyword matching).

    Args:
        splits (list): Document chunks to embed and store
        embeddings (HuggingFaceEmbeddings): Embedding model

    Returns:
        Chroma: Vector store instance
    """
    print(f"\n💾 Storing chunks in vector database: {CHROMA_DB_PATH}")
    print(f"   Processing {len(splits)} chunks in batches of {BATCH_SIZE}...")

    try:
        # Process all documents at once with progress bar
        # ChromaDB handles batching internally, but we show progress
        with tqdm(total=len(splits), desc="Embedding & storing", unit="chunk") as pbar:
            # Create vector store (this embeds all documents)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH
            )
            pbar.update(len(splits))

        print(f"✓ Successfully stored {len(splits)} chunks")
        return vectorstore

    except Exception as e:
        raise Exception(f"Failed to store documents in vector database: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.

    Workflow:
    1. Parse command line arguments
    2. Validate PDF exists
    3. Load and parse PDF
    4. Split into chunks
    5. Initialize embedding model
    6. Store in vector database
    """
    print("=" * 70)
    print("🍳 Smart Pantry - Document Ingestion")
    print("=" * 70)

    # Step 1: Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Using PDF from command line: {pdf_path}")
    else:
        pdf_path = DEFAULT_PDF_PATH
        print(f"Using default PDF: {pdf_path}")

    try:
        # Step 2: Validate PDF exists and is readable
        pdf_path = validate_pdf_path(pdf_path)

        # Step 3: Load PDF documents
        docs = load_pdf_documents(pdf_path)

        # Step 4: Split documents into chunks
        splits = split_documents(docs)

        # Step 5: Initialize local embedding model
        embeddings = initialize_embeddings()

        # Step 6: Store in ChromaDB
        vectorstore = store_in_vectordb(splits, embeddings)

        # Success!
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"📊 Statistics:")
        print(f"   - Pages processed: {len(docs)}")
        print(f"   - Chunks created: {len(splits)}")
        print(f"   - Database location: {CHROMA_DB_PATH}")
        print(f"\n💡 Next step: Run 'streamlit run app.py' to query your recipes!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(f"💡 Make sure your PDF is in the correct location.")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"💡 Check the error message above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
