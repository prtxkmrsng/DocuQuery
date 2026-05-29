import os
import sys
import warnings
from pathlib import Path

# Suppress noisy LangChain community deprecation alerts
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def batch_ingest_documents(folder_path_str: str, vector_store_dir: str = "faiss_index"):
    """
    Scans a directory for PDFs, pools text chunks across all files, 
    and builds a single unified vector database index.
    """
    folder_path = Path(folder_path_str)
    
    # Verify the source directory folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"\n❌ [Ingestion Error] Target folder directory not found at: '{folder_path_str}'")
        print("👉 Please create this folder locally and drop your target PDFs inside it.")
        sys.exit(1)
        
    # Gather all PDF path objects
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"\n⚠️ [Ingestion Warning] No .pdf files discovered inside '{folder_path_str}' directory.")
        return

    print(f"📂 [Ingestion] Discovered {len(pdf_files)} PDF document(s) for batch processing.")
    
    # Master collection array to hold all document chunks across all files
    all_chunks = []
    
    # Hierarchical text splitter configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Step 1: Iterate through files and pool text chunks
    for pdf_path in pdf_files:
        print(f"👉 Processing: {pdf_path.name}")
        try:
            # Pass the complete relative path string (e.g., "files/sample_report.pdf")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Extract and segment chunks for the current file
            file_chunks = text_splitter.split_documents(documents)
            all_chunks.extend(file_chunks)
            
            print(f"   Parsed {len(documents)} pages -> Generated {len(file_chunks)} text fragments.")
        except Exception as e:
            print(f"❌ [Parser Error] Skipped corrupted file {pdf_path.name}. Details: {e}")
            continue

    if not all_chunks:
        print("\n❌ Ingestion Aborted: No valid text matrices could be extracted from the folder.")
        return

    # Step 2: Load embedding engine into memory
    print(f"\n[Ingestion] Loading embedding model weights ('all-mpnet-base-v2') into RAM...")
    embedding_engine = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Step 3: Compute dense vector metrics for the full batch
    print(f"[Ingestion] Computing dense vector spaces for a total of {len(all_chunks)} combined chunks...")
    db = FAISS.from_documents(all_chunks, embedding_engine)
    
    # Step 4: Save unified footprint to disk
    db.save_local(vector_store_dir)
    print(f"\n🚀 [Ingestion Complete] Unified vector footprint saved locally under: '{vector_store_dir}/'")

if __name__ == "__main__":
    # Create the files directory if it doesn't exist for a cleaner user onboarding setup
    Path("./files").mkdir(exist_ok=True)
    
    # Execute batch extraction
    batch_ingest_documents("./files")