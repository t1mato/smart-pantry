"""
Smart Pantry - RAGAS Evaluation Framework

Uses RAGAS (RAG Assessment) framework to evaluate retrieval quality.

Metrics evaluated:
- Context Precision: Are retrieved documents relevant to the query?
- Context Recall: Did we retrieve all relevant information?
- Faithfulness: Is the generated answer grounded in retrieved context?
- Answer Relevance: Does the answer actually address the query?

Compares:
- Semantic Search only
- BM25 Search only
- Hybrid (BM25 + Semantic + RRF)
- Hybrid + Cross-Encoder Reranking (future)
"""

import os
import sys
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv

# RAGAS imports
from ragas import evaluate
from ragas.metrics.collections import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

load_dotenv()

# Import from app.py
from app import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    reciprocal_rank_fusion,
    initialize_vectorstore,
    initialize_hybrid_retriever,
    initialize_llm,
    format_context,
    generate_recipe
)


# ============================================================================
# TEST DATASET
# ============================================================================

@dataclass
class TestCase:
    """A test case for RAG evaluation."""
    question: str  # User query
    ground_truth: str  # Expected answer content (for context_recall)
    relevant_keywords: List[str]  # Keywords that should appear in results


# Reduced test cases to stay within API limits
# Gemini free tier: 20 requests/day
# We need: 4 test cases × 3 methods × 2 (generation + RAGAS) = ~24 calls
TEST_CASES = [
    TestCase(
        question="I have chicken and rice. What can I make for dinner?",
        ground_truth="A recipe using chicken and rice as main ingredients, such as chicken and rice casserole, fried rice with chicken, or chicken rice bowl.",
        relevant_keywords=["chicken", "rice"]
    ),
    TestCase(
        question="What vegetarian recipes use beans for protein?",
        ground_truth="Vegetarian recipes featuring beans as a protein source, such as bean chili, bean burgers, or bean salad.",
        relevant_keywords=["bean", "vegetarian", "protein"]
    ),
    TestCase(
        question="I'm on a tight budget and have pasta. What's a cheap meal?",
        ground_truth="Budget-friendly pasta dishes like pasta with simple tomato sauce, aglio e olio, or pasta with vegetables.",
        relevant_keywords=["pasta", "budget", "cheap"]
    ),
]


# ============================================================================
# RETRIEVAL METHODS
# ============================================================================

def retrieve_semantic_only(vectorstore: Chroma, query: str, k: int = 5) -> List[Document]:
    """Pure semantic search using embeddings."""
    return vectorstore.similarity_search(query, k=k)


def retrieve_bm25_only(bm25_retriever: BM25Retriever, query: str, k: int = 5) -> List[Document]:
    """Pure BM25 keyword search."""
    bm25_retriever.k = k * 2  # Get more for better fusion
    return bm25_retriever.invoke(query)[:k]


def retrieve_hybrid_rrf(bm25_retriever: BM25Retriever, semantic_retriever, query: str, k: int = 5) -> List[Document]:
    """Hybrid search: BM25 + Semantic with RRF."""
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_retriever.invoke(query)
    fused = reciprocal_rank_fusion([bm25_results, semantic_results])
    return fused[:k]


# ============================================================================
# RAGAS EVALUATION
# ============================================================================

def create_ragas_dataset(test_cases: List[TestCase], retrieval_method, llm, method_name: str) -> Dataset:
    """
    Create a RAGAS-compatible dataset from test cases.

    RAGAS expects:
    - question: User query
    - answer: Generated answer from RAG
    - contexts: Retrieved documents (list of strings)
    - ground_truth: Expected answer (for context_recall)
    """
    print(f"\n🔍 Generating dataset for: {method_name}")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"   [{i}/{len(test_cases)}] Processing: \"{test_case.question[:50]}...\"")

        # Retrieve documents
        retrieved_docs = retrieval_method(test_case.question)

        # Format context for LLM
        context_str = format_context(retrieved_docs)

        # Generate answer using LLM
        answer = generate_recipe(
            llm,
            context_str,
            test_case.question,
            restrictions=""  # No restrictions for evaluation
        )

        # Prepare for RAGAS
        questions.append(test_case.question)
        answers.append(answer)
        contexts_list.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(test_case.ground_truth)

    # Create dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }

    return Dataset.from_dict(data)


def evaluate_retrieval_method(
    method_name: str,
    retrieval_func,
    vectorstore: Chroma,
    llm,
    test_cases: List[TestCase]
) -> Dict:
    """
    Evaluate a single retrieval method using RAGAS.

    Returns dict of metric scores.
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {method_name}")
    print(f"{'='*80}")

    # Create dataset
    dataset = create_ragas_dataset(test_cases, retrieval_func, llm, method_name)

    print(f"\n🧪 Running RAGAS evaluation...")
    print(f"   This may take a few minutes (using LLM for metrics)...")

    # Run RAGAS evaluation
    # Note: RAGAS uses LLMs to evaluate, so it needs the Gemini model
    results = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=llm,
        embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    print(f"\n✅ Evaluation complete for {method_name}")

    return results


def compare_methods(all_results: Dict[str, Dict]):
    """Print comparison table of all methods."""
    print("\n" + "="*100)
    print("📊 RAGAS EVALUATION RESULTS - COMPARISON")
    print("="*100)

    # Extract metrics
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    # Create comparison table
    print(f"\n{'Method':<25}", end="")
    for metric in metrics:
        print(f"{metric:<20}", end="")
    print()
    print("-" * 100)

    for method_name, results in all_results.items():
        print(f"{method_name:<25}", end="")
        for metric in metrics:
            value = results.get(metric, 0.0)
            print(f"{value:.4f}{' '*15}", end="")
        print()

    print("\n" + "="*100)

    # Calculate and display best performer for each metric
    print("\n🏆 Best Performers:")
    for metric in metrics:
        best_method = max(all_results.items(), key=lambda x: x[1].get(metric, 0))
        print(f"   {metric}: {best_method[0]} ({best_method[1].get(metric, 0):.4f})")


def save_results_to_csv(all_results: Dict[str, Dict], filename: str = "evaluation_results.csv"):
    """Save results to CSV for easy analysis."""
    # Convert to DataFrame
    df = pd.DataFrame(all_results).T
    df.index.name = "Method"

    # Save to CSV
    df.to_csv(filename)
    print(f"\n💾 Results saved to: {filename}")

    # Also save detailed report
    report_filename = filename.replace('.csv', '_report.txt')
    with open(report_filename, 'w') as f:
        f.write("SMART PANTRY - RAGAS EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")

        f.write("RESULTS:\n")
        f.write(df.to_string())
        f.write("\n\n")

        f.write("BEST PERFORMERS:\n")
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        for metric in metrics:
            best_method = max(all_results.items(), key=lambda x: x[1].get(metric, 0))
            f.write(f"  {metric}: {best_method[0]} ({best_method[1].get(metric, 0):.4f})\n")

    print(f"📄 Report saved to: {report_filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the RAGAS evaluation framework."""
    print("\n" + "="*80)
    print("🧪 SMART PANTRY - RAGAS EVALUATION FRAMEWORK")
    print("="*80)

    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ Vector database not found at {CHROMA_PATH}")
        print("   Run 'python ingest.py' first to create it.")
        sys.exit(1)

    # Initialize components
    print("\n📚 Initializing components...")
    vectorstore = initialize_vectorstore()
    bm25_retriever, semantic_retriever = initialize_hybrid_retriever(vectorstore)
    llm = initialize_llm()

    print(f"✓ Vector database loaded")
    print(f"✓ BM25 and Semantic retrievers ready")
    print(f"✓ LLM initialized ({GEMINI_MODEL})")

    # Define retrieval methods to evaluate
    methods = {
        "Semantic Only": lambda q: retrieve_semantic_only(vectorstore, q),
        "BM25 Only": lambda q: retrieve_bm25_only(bm25_retriever, q),
        "Hybrid (RRF)": lambda q: retrieve_hybrid_rrf(bm25_retriever, semantic_retriever, q)
    }

    print(f"\n📋 Test cases: {len(TEST_CASES)}")
    print(f"🔬 Methods to evaluate: {len(methods)}")

    # Run evaluation for each method
    all_results = {}

    for method_name, retrieval_func in methods.items():
        try:
            results = evaluate_retrieval_method(
                method_name,
                retrieval_func,
                vectorstore,
                llm,
                TEST_CASES
            )
            all_results[method_name] = results

        except Exception as e:
            print(f"❌ Error evaluating {method_name}: {e}")
            import traceback
            traceback.print_exc()

    # Display comparison
    if all_results:
        compare_methods(all_results)
        save_results_to_csv(all_results)

    print("\n✅ Evaluation complete!")
    print("\n💡 Next steps:")
    print("   1. Review the results above")
    print("   2. Implement cross-encoder reranking")
    print("   3. Re-run evaluation to measure improvement")
    print("   4. Tune hybrid search weights if needed")
    print("\n📖 Learn more about RAGAS: https://docs.ragas.io/")


if __name__ == "__main__":
    main()
