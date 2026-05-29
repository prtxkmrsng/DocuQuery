# DocuQuery 🚀

An adaptive, enterprise-grade Retrieval-Augmented Generation (RAG) legal and financial audit engine. The system ingests multi-document directories of complex PDFs, extracts semantic structures hierarchically, creates dense local vector indexes, and automates real-time model discovery via the Gemini API. Every query is evaluated using strict grounding rules to eliminate hallucinations and is recorded to a persistence logging tier for a comprehensive audit trail.

---

## 🏗️ Architectural Topology

- **Document Ingestion Engine:** Automated multi-file directory scanner using `PyPDFLoader` to parse layouts and aggregate page tokens.
- **Hierarchical Text Segmentation:** `RecursiveCharacterTextSplitter` chunking prose across logical structural gaps with a 100-character context overlap boundary.
- **Local Dense Vector Processing:** Localized token vector calculations powered by `sentence-transformers` and cataloged into an ultra-fast `FAISS` key-space index.
- **Dynamic Model Auto-Discovery:** Automated runtime capabilities mapping. The application utilizes the `google-genai` SDK on startup to query your API privileges, select the optimal text generation model, and fall back dynamically to maintain perfect service uptime.
- **Modern Orchestration Pipe:** High-throughput data piping managed via `LangChain Expression Language (LCEL)` for optimal low-latency text token streaming.
- **Audit Logging Tier:** Transaction tracking system that serializes user prompts, response content, and explicit document-page citations directly into a containerized `MongoDB` instance.

---

## 🛠️ Project Structure

```text
DocuQuery/
│
├── files/                      # Folder to store input pdf files
├── faiss_index/                # Folder where your vector databases will be generated
├── .gitattributes              # Line-ending normalizations
├── .gitignore                  # Active tracking exemptions map
├── .env                        # Local credential variables blueprint
├── ingest.py                   # Consolidated batch document ingestion pipeline
├── requirementts.txt                   # Prerequisites
└── query.py                    # Dynamic discovery inference and audit logging script
```
---
## 🚀 Local Windows Setup Instruction

### 1. Prerequisite Checklist

- Python: Ensure Python 3.10+ is active on your environment variable path.
- Environment Engine: Docker Desktop installed and running natively on Windows with the WSL2 execution backend.
- API Privileges: A functional Google AI Studio API key.

### 2. Dependency Configuration

Navigate to your project folder and set up your isolated virtual space:
```
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Establish Local Docker Database Infrastructure

Fire up an isolated, local MongoDB container network node to record conversation histories and metadata indexes:

```
docker run -d -p 27017:27017 --name docuquery_mongo mongo:latest
```

### 4. Configure Environmental Secrets

Create a .env file in the root directory:

```
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 5. Execute Directory Document Ingestion

Create a folder named ``files/`` in your root workspace and drop any selection of valid target reference PDFs inside it. Compile the unified vector index database:

```
python ingest.py
```

This builds your local faiss_index/ vector matrix database.

### 6. Run the Grounded Query Interface

Dispatch search questions across your processed document library:

```
python query.py
```

### 7. Inspect the Audit Logs

To review the immutable citation history and query records captured inside your MongoDB database, execute:

```
docker exec -it docuquery_mongo mongosh docuquery_db --eval "db.conversation_history.find().pretty()"
```
---