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
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv

# RAGAS imports
# NOTE: Using legacy ragas.metrics (not collections) because collections requires InstructorLLM
# Collections doesn't support LangChain's ChatGroq wrapper
from ragas import evaluate, RunConfig
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy
)
from datasets import Dataset

load_dotenv()

# Import from app.py
from app import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    reciprocal_rank_fusion,
    rerank_with_cross_encoder,
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


def retrieve_hybrid_with_reranking(bm25_retriever: BM25Retriever, semantic_retriever, query: str, k: int = 5) -> List[Document]:
    """Hybrid + Cross-Encoder: BM25 + Semantic with RRF, then cross-encoder reranking."""
    # First get hybrid results (retrieve more for reranking)
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_retriever.invoke(query)
    fused = reciprocal_rank_fusion([bm25_results, semantic_results])

    # Get top 10 candidates for reranking
    candidates = fused[:10]

    # Rerank with cross-encoder
    return rerank_with_cross_encoder(query, candidates, top_k=k)


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

        # Ensure answer is a string (not list or other type)
        if not isinstance(answer, str):
            answer = str(answer)

        # Prepare for RAGAS
        questions.append(test_case.question)
        answers.append(answer)
        contexts_list.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(test_case.ground_truth)

    # Create dataset using pandas to handle nested lists better
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }

    # Convert to DataFrame first, then to Dataset (handles nested lists better)
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)


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

    # Configure RAGAS with extended timeouts and better error handling
    run_config = RunConfig(
        timeout=300,  # 5 minutes per operation (increased from default 180s)
        max_retries=3,  # Retry up to 3 times on failures
        max_wait=30,  # Wait up to 30s between retries
        max_workers=2,  # Limit concurrent workers to avoid rate limits
        log_tenacity=False  # Disable retry logging for cleaner output
    )

    # Initialize all 4 RAGAS metrics for comprehensive RAG evaluation
    # Using legacy API (ragas.metrics) - LLM is passed to evaluate(), not to metrics
    # CRITICAL: Set strictness=1 for AnswerRelevancy to work with Groq
    # Groq only supports n=1, and AnswerRelevancy generates N questions by default (default=3)
    # See: https://github.com/explodinggradients/ragas/issues/1072
    metrics = [
        ContextPrecision(),             # Retrieval: Are retrieved docs relevant?
        ContextRecall(),                # Retrieval: Did we retrieve all relevant info?
        Faithfulness(),                 # Generation: Is answer grounded in context?
        AnswerRelevancy(strictness=1)   # Generation: strictness=1 for Groq compatibility
    ]

    # Run RAGAS evaluation
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        run_config=run_config,
        raise_exceptions=False  # Continue even if some metrics fail
    )

    print(f"\n✅ Evaluation complete for {method_name}")

    return results


def compare_methods(all_results):
    """Print comparison table of all methods."""
    print("\n" + "="*100)
    print("📊 RAGAS EVALUATION RESULTS - COMPARISON")
    print("="*100)

    # Extract metrics
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    # Convert EvaluationResult objects to metric dictionaries
    results_dict = {}
    for method_name, eval_result in all_results.items():
        # Convert to DataFrame and get mean scores
        df = eval_result.to_pandas()
        results_dict[method_name] = {metric: df[metric].mean() for metric in metrics}

    # Create comparison table
    print(f"\n{'Method':<25}", end="")
    for metric in metrics:
        print(f"{metric:<20}", end="")
    print()
    print("-" * 100)

    for method_name, scores in results_dict.items():
        print(f"{method_name:<25}", end="")
        for metric in metrics:
            value = scores.get(metric, 0.0)
            print(f"{value:.4f}{' '*15}", end="")
        print()

    print("\n" + "="*100)

    # Calculate and display best performer for each metric
    print("\n🏆 Best Performers:")
    for metric in metrics:
        best_method = max(results_dict.items(), key=lambda x: x[1].get(metric, 0))
        best_value = best_method[1].get(metric, 0)
        print(f"   {metric}: {best_method[0]} ({best_value:.4f})")


def save_results_to_csv(all_results, filename: str = "evaluation_results.csv"):
    """Save results to CSV for easy analysis."""
    # Convert EvaluationResult objects to metric dictionaries
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    results_dict = {}
    for method_name, eval_result in all_results.items():
        # Convert to DataFrame and get mean scores
        df_temp = eval_result.to_pandas()
        results_dict[method_name] = {metric: df_temp[metric].mean() for metric in metrics}

    # Convert to DataFrame
    df = pd.DataFrame(results_dict).T
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
            # Use DataFrame instead of EvaluationResult objects
            if metric in df.columns:
                best_method = df[metric].idxmax()
                best_value = df.loc[best_method, metric]
                if not pd.isna(best_value):
                    f.write(f"  {metric}: {best_method} ({best_value:.4f})\n")
                else:
                    f.write(f"  {metric}: No valid results\n")
            else:
                f.write(f"  {metric}: Metric not found\n")

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

    # Use Groq API for fast, reliable evaluation
    print("   Initializing Groq LLM (llama-3.1-8b-instant)...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Efficient 8B model for faster evaluation with lower token usage
        temperature=0,  # Deterministic for evaluation
        max_tokens=4096,  # Sufficient for recipe generation
        # Note: Groq only supports n=1 by default, no need to specify
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    print(f"✓ Vector database loaded")
    print(f"✓ BM25 and Semantic retrievers ready")
    print(f"✓ LLM initialized (Groq Llama-3.1-8B, n=1 for RAGAS compatibility)")

    # Define retrieval methods to evaluate
    methods = {
        "Semantic Only": lambda q: retrieve_semantic_only(vectorstore, q),
        "BM25 Only": lambda q: retrieve_bm25_only(bm25_retriever, q),
        "Hybrid (RRF)": lambda q: retrieve_hybrid_rrf(bm25_retriever, semantic_retriever, q),
        "Hybrid + Reranking": lambda q: retrieve_hybrid_with_reranking(bm25_retriever, semantic_retriever, q)
    }

    print(f"\n📋 Test cases: {len(TEST_CASES)}")
    print(f"🔬 Methods to evaluate: {len(methods)}")
    print(f"   1. Semantic Only (baseline)")
    print(f"   2. BM25 Only (keyword)")
    print(f"   3. Hybrid (RRF)")
    print(f"   4. Hybrid + Cross-Encoder Reranking ⭐")

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
