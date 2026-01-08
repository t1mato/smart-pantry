"""
Smart Pantry & Diet Guardian - Streamlit Application

This app helps you find recipes based on ingredients you have at home.
It retrieves recipes from your local PDF cookbook database and adapts them
to your dietary restrictions using AI.

Features:
- Ingredient-based recipe search
- Dietary restriction filtering
- Source transparency (shows which cookbook/page)
- Grounded in real recipes (no hallucinations)
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Vector database path (must match ingest.py)
CHROMA_PATH = "./chroma_db"

# Embedding model (MUST match the model used in ingest.py)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM configuration
# Using Google Gemini Flash (latest version) for recipe generation
# "gemini-flash-latest" automatically uses the newest flash model available
# Your API key has access to gemini-2.5-flash and newer models
GEMINI_MODEL = "gemini-flash-latest"

# Number of recipe chunks to retrieve from vector database
# Higher = more context but slower, Lower = faster but might miss recipes
NUM_RESULTS = 5

# Prompt template for recipe generation
RECIPE_PROMPT_TEMPLATE = """
You are a helpful cooking assistant. Based on the recipe context provided below,
create a recipe recommendation that uses the user's available ingredients and
respects their dietary restrictions.

RECIPE CONTEXT (from cookbooks):
{context}

USER'S INGREDIENTS:
{ingredients}

DIETARY RESTRICTIONS:
{restrictions}

INSTRUCTIONS:
1. If the context contains recipes that match the ingredients, recommend the best one
2. Adapt the recipe if needed to respect dietary restrictions
3. If no exact match exists, suggest the closest recipe and explain substitutions
4. Always cite the source (cookbook and page number from the context metadata)
5. Be specific about measurements and cooking steps
6. If restrictions make a recipe impossible (e.g., vegan cake with eggs), say so honestly

FORMAT YOUR RESPONSE AS:
📖 **Recipe Name**
🏷️ *Source: [Cookbook name, Page X]*

**Ingredients:**
- [List ingredients with measurements]

**Instructions:**
1. [Step-by-step instructions]

**Notes:**
- [Any substitutions made for dietary restrictions]
- [Tips or warnings]
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_vectorstore():
    """
    Initialize the ChromaDB vector store with the same embeddings used during ingestion.

    CRITICAL: The embedding model MUST match the one used in ingest.py.
    Mismatched embeddings will result in nonsensical similarity scores.

    Returns:
        Chroma: Vector store instance, or None if database doesn't exist

    Raises:
        FileNotFoundError: If ChromaDB hasn't been initialized yet
    """
    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector database not found at {CHROMA_PATH}. "
            f"Please run 'python ingest.py' first to create it."
        )

    # Initialize the same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load existing vector store
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore


def initialize_llm():
    """
    Initialize Google Gemini LLM for recipe generation and adaptation.

    Requires GOOGLE_API_KEY in environment variables.

    Returns:
        ChatGoogleGenerativeAI: LLM instance

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )

    # Initialize Gemini with temperature for creative recipe adaptation
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,  # Low temp for consistent, factual responses
        google_api_key=api_key
    )

    return llm


def search_recipes(vectorstore, ingredients, restrictions, num_results=NUM_RESULTS):
    """
    Search the vector database for recipes matching the ingredients.

    This performs semantic search, so it understands meaning:
    - "tomato" will match "tomatoes", "diced tomatoes", etc.
    - "chicken breast" will match "chicken", "poultry", etc.

    Args:
        vectorstore (Chroma): Vector database instance
        ingredients (str): User's available ingredients
        restrictions (str): Dietary restrictions
        num_results (int): Number of results to retrieve

    Returns:
        list: List of Document objects with recipe chunks and metadata
    """
    # Construct search query combining ingredients and restrictions
    # This helps find recipes that are more likely to be adaptable
    search_query = f"Recipes using: {ingredients}"
    if restrictions:
        search_query += f" that can be made {restrictions}"

    # Perform similarity search
    results = vectorstore.similarity_search(
        query=search_query,
        k=num_results
    )

    return results


def format_context(search_results):
    """
    Format search results into a context string for the LLM prompt.

    Includes both the recipe text and metadata (source, page number).

    Args:
        search_results (list): List of Document objects from vector search

    Returns:
        str: Formatted context string
    """
    context_parts = []

    for i, doc in enumerate(search_results, 1):
        # Extract metadata
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', 'Unknown page')

        # Format this result
        context_parts.append(
            f"--- Recipe Chunk {i} ---\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n".join(context_parts)


def generate_recipe(llm, context, ingredients, restrictions):
    """
    Generate a recipe recommendation using the LLM.

    Args:
        llm: Language model instance
        context (str): Recipe context from vector search
        ingredients (str): User's ingredients
        restrictions (str): Dietary restrictions

    Returns:
        str: Generated recipe recommendation
    """
    # Create prompt from template
    prompt = ChatPromptTemplate.from_template(RECIPE_PROMPT_TEMPLATE)

    # Format the prompt with user inputs
    formatted_prompt = prompt.format(
        context=context,
        ingredients=ingredients,
        restrictions=restrictions if restrictions else "None"
    )

    # Generate response
    response = llm.invoke(formatted_prompt)

    return response.content


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """
    Main Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="🍳 Smart Pantry & Diet Guardian",
        page_icon="🍳",
        layout="wide"
    )

    # Header
    st.title("🍳 Smart Pantry & Diet Guardian")
    st.markdown(
        "Find delicious recipes based on what's in your pantry, "
        "adapted to your dietary needs. All recipes are sourced from real cookbooks."
    )

    # Sidebar for configuration and info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
            This app uses:
            - 🔍 **Semantic Search**: Finds recipes by meaning, not just keywords
            - 📚 **Local Database**: Your PDF cookbooks stored locally
            - 🤖 **AI Adaptation**: Google Gemini adapts recipes to your needs
            - 🎯 **Source Citation**: Always shows which cookbook and page
            """
        )

        st.divider()

        st.header("⚙️ Settings")
        num_results = st.slider(
            "Number of recipes to search",
            min_value=1,
            max_value=10,
            value=NUM_RESULTS,
            help="More results = more context but slower"
        )

    # Initialize components
    try:
        with st.spinner("Loading vector database..."):
            vectorstore = initialize_vectorstore()

        with st.spinner("Initializing AI model..."):
            llm = initialize_llm()

    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info("👉 Run `python ingest.py` to create the recipe database first.")
        st.stop()

    except ValueError as e:
        st.error(f"❌ {e}")
        st.info("👉 Add your Google API key to the `.env` file:\n```\nGOOGLE_API_KEY=your_key_here\n```")
        st.stop()

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

    # Main input form
    st.divider()
    st.header("🥘 What's in your pantry?")

    col1, col2 = st.columns(2)

    with col1:
        ingredients = st.text_area(
            "Available Ingredients",
            placeholder="e.g., chicken breast, rice, bell peppers, onion",
            height=100,
            help="List the ingredients you have available"
        )

    with col2:
        restrictions = st.text_area(
            "Dietary Restrictions (Optional)",
            placeholder="e.g., vegetarian, gluten-free, no cilantro",
            height=100,
            help="Any dietary restrictions or preferences"
        )

    # Search button
    search_button = st.button("🔍 Find Recipes", type="primary", use_container_width=True)

    # Process search
    if search_button:
        if not ingredients.strip():
            st.warning("⚠️ Please enter at least one ingredient.")
            return

        # Show search status
        with st.spinner("🔍 Searching cookbook database..."):
            search_results = search_recipes(
                vectorstore,
                ingredients,
                restrictions,
                num_results=num_results
            )

        if not search_results:
            st.warning("😔 No recipes found. Try different ingredients or add more cookbooks.")
            return

        # Format context
        context = format_context(search_results)

        # Generate recipe
        with st.spinner("🤖 Adapting recipe to your needs..."):
            recipe = generate_recipe(llm, context, ingredients, restrictions)

        # Display result
        st.divider()
        st.header("📖 Your Recipe")
        st.markdown(recipe)

        # Show retrieved context in expander
        with st.expander("🔍 View Source Recipe Chunks"):
            st.markdown("*These are the cookbook excerpts used to generate your recipe:*")
            for i, doc in enumerate(search_results, 1):
                st.markdown(f"**Chunk {i}**")
                st.markdown(f"*Source: {doc.metadata.get('source', 'Unknown')} "
                          f"(Page {doc.metadata.get('page', 'Unknown')})*")
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.divider()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
