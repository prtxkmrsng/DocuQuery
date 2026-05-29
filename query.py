import os
import sys
import warnings
from dotenv import load_dotenv

# Suppress noisy background deprecation alerts
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ingest environment credentials from .env
load_dotenv()

def auto_discover_gemini_model() -> str:
    """
    Programmatically lists all active models available for the provided API key
    and returns the best available text generation model automatically.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ [Discovery Error] Missing API Key in environment variables.")
        print("👉 Please populate GOOGLE_API_KEY inside your local .env file.\n")
        sys.exit(1)

    print("[Query Engine] Interrogating Google AI Studio endpoint for dynamic model discovery...")
    try:
        # Initialize the native Google GenAI SDK client
        from google import genai
        client = genai.Client(api_key=api_key)
        
        discovered_text_models = []
        # Query the active models endpoint
        for model_meta in client.models.list():
            if "generateContent" in model_meta.supported_actions:
                # Strip the prefix 'models/' to match standard string formatting
                short_name = model_meta.name.replace("models/", "")
                discovered_text_models.append(short_name)
        
        # Define a prioritized preference cascade matching active models
        preference_cascade = [
            "gemini-2.5-flash",
            "gemini-3.5-flash", 
            "gemini-2.0-flash", 
            "gemini-1.5-flash",
            "gemini-pro"
        ]
        
        for priority_target in preference_cascade:
            if priority_target in discovered_text_models:
                print(f"🎯 Auto-Selected Active Model: '{priority_target}'")
                return priority_target
                
        # If no preferred targets found, fall back to any available generation model
        if discovered_text_models:
            print(f"🎯 Auto-Selected Discovered Model: '{discovered_text_models[0]}'")
            return discovered_text_models[0]
            
    except Exception as e:
        print(f"⚠️ [Discovery Warning] Programmatic listing failed or unauthorized: {e}")
    
    # Global ultimate safe fallback string if endpoint discovery is rate-limited
    print("👉 Falling back to standard default model path: 'gemini-2.5-flash'")
    return "gemini-2.5-flash"


def format_docs(docs):
    """Combines extracted page snippets into a single unified context block."""
    return "\n\n".join(doc.page_content for doc in docs)


def run_audit_query(user_question: str, vector_store_dir: str = "faiss_index"):
    """
    Loads the unified FAISS index, retrieves contextual elements via semantic matching,
    and constructs an LCEL pipe flow querying the auto-discovered Gemini model.
    """
    if not os.path.exists(vector_store_dir):
        print(f"[Query Error] Vector index database not found at '{vector_store_dir}'. Run ingest.py first!")
        return

    # Step 1: Initialize local sentence-embeddings transformer
    print("[Query Engine] Loading local embedding model into memory...")
    embedding_engine = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Step 2: Deserialization index maps safely from disk
    print("[Query Engine] Deserializing local FAISS index footprint...")
    db = FAISS.load_local(
        vector_store_dir, 
        embedding_engine, 
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Step 3: Run Dynamic Model Discovery 
    target_model_string = auto_discover_gemini_model()

    # Step 4: Establish connection to Google AI Studio Gemini API via discovered string
    llm = ChatGoogleGenerativeAI(
        model=target_model_string,
        temperature=0.1, 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Step 5: Define prompt guidelines
    system_prompt = (
        "You are a professional legal and financial audit assistant.\n"
        "Analyze the retrieved document context pieces below and provide a concise response.\n"
        "If you do not know the answer or if it is not explicitly mentioned in the context, "
        "state cleanly that the document does not contain the necessary information. Do not make anything up.\n\n"
        "Retrieved Document Context:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Step 6: Assemble the LCEL RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n[Query] Executing search across database logs for: '{user_question}'")
    retrieved_documents = retriever.invoke(user_question)
    answer = rag_chain.invoke(user_question)

    print("\n🚀 --- Grounded Audit Response ---")
    print(answer)
    
    print("\n📝 --- Source Citations & Audit Trail ---")
    for idx, doc in enumerate(retrieved_documents):
        page_num = doc.metadata.get("page", 0) + 1
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        print(f"[{idx + 1}] Source Doc: {source_file} | Page Link: {page_num}")
        print(f"    Content Snippet: {doc.page_content[:120].strip()}...")

if __name__ == "__main__":
    test_query = "What are the core guidelines or maintenance rules mentioned?"
    run_audit_query(test_query)