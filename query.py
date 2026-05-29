import os
import sys
import datetime
import warnings
from dotenv import load_dotenv

# Suppress background deprecation warnings cleanly
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient

# Load environmental variables
load_dotenv()

def auto_discover_gemini_model() -> str:
    """Programmatically discovers the best text generation model for the API key."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ [Discovery Error] Missing API Key in environment variables.")
        sys.exit(1)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        discovered_text_models = []
        for model_meta in client.models.list():
            if "generateContent" in model_meta.supported_actions:
                discovered_text_models.append(model_meta.name.replace("models/", ""))
        
        preference_cascade = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        for priority_target in preference_cascade:
            if priority_target in discovered_text_models:
                return priority_target
        if discovered_text_models:
            return discovered_text_models[0]
    except Exception:
        pass
    return "gemini-2.5-flash"


def format_docs(docs):
    """Combines extracted page snippets into a single context block."""
    return "\n\n".join(doc.page_content for doc in docs)


def log_transaction_to_mongodb(question: str, answer: str, source_documents: list):
    """
    Connects to the local Docker MongoDB container and saves a structured
    historical log entry of the conversation and context audit trail.
    """
    try:
        # Connect to local container port
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        db = client["docuquery_db"]
        logs_collection = db["conversation_history"]
        
        # Build out clear citation metadata arrays
        citations = []
        for doc in source_documents:
            citations.append({
                "source_file": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", 0) + 1,
                "text_snippet": doc.page_content[:200].strip()
            })
            
        # Structure payload object matching roadmap requirements
        log_entry = {
            "timestamp": datetime.datetime.utcnow(),
            "user_question": question,
            "generated_response": answer,
            "audit_trail_citations": citations
        }
        
        # Insert record into database space
        logs_collection.insert_one(log_entry)
        print("💾 [MongoDB] Transaction entry and context references archived successfully.")
    except Exception as e:
        print(f"⚠️ [MongoDB Log Warning] Failed to persist log entry to database: {e}")


def run_audit_query(user_question: str, vector_store_dir: str = "faiss_index"):
    """Loads FAISS index, matches contexts, passes to LLM, and records trail to MongoDB."""
    if not os.path.exists(vector_store_dir):
        print(f"[Query Error] Vector index database not found. Run ingest.py first!")
        return

    # Step 1: Initialize local embeddings
    embedding_engine = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Step 2: Load local vector store
    db = FAISS.load_local(vector_store_dir, embedding_engine, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Step 3: Auto-select model string and load Gemini
    target_model_string = auto_discover_gemini_model()
    print(f"[Query Engine] Executing via discovered engine target: '{target_model_string}'")
    
    llm = ChatGoogleGenerativeAI(
        model=target_model_string,
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Step 4: System Prompt Framework
    system_prompt = (
        "You are a professional legal and financial audit assistant.\n"
        "Analyze the retrieved document context pieces below and provide a concise response.\n"
        "If you do not know the answer, state that the document does not contain the information. Do not hallucinate.\n\n"
        "Retrieved Document Context:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Step 5: Assemble modern LCEL RAG Pipe Chain
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 6: Invoke and Parse Data Streams
    retrieved_documents = retriever.invoke(user_question)
    answer = rag_chain.invoke(user_question)

    print("\n🚀 --- Grounded Audit Response ---")
    print(answer)
    print("----------------------------------\n")
    
    # Step 7: Log everything straight into MongoDB database index
    log_transaction_to_mongodb(user_question, answer, retrieved_documents)

if __name__ == "__main__":
    query = "What are the core guidelines or maintenance rules mentioned?"
    run_audit_query(query)