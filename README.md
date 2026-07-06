# AI Research Bot: Sovereign Parent-Child RAG for Structural Design

A production-grade, asynchronous Retrieval-Augmented Generation (RAG) system built natively using **FastAPI**, **PostgreSQL (pgvector)**, and **Google Gemini**. Designed to ingest structural steel design PDF documents (e.g., IS 800 standards) and provide context-rich, mathematically accurate engineering answers.

---

## 🏗️ System Architecture

```
                 +-----------------------------------+
                 |        Uploaded PDF File          |
                 +-----------------+-----------------+
                                   |
                                   v
                 +-----------------+-----------------+
                 |  Semantic Chunking (Parent text)  |
                 +-----------------+-----------------+
                                   |
                                   +---------------------------+
                                   |                           |
                                   v                           v
                      +------------+------------+  +-----------+-----------+
                      | Save to FileParent Table |  | Slice into sentences  |
                      |   (id, file_name, text)  |  |      (Children)       |
                      +-------------------------+  +-----------+-----------+
                                                               |
                                                               v
                                                   +-----------+-----------+
                                                   | Vectorize (all-MiniLM)|
                                                   +-----------+-----------+
                                                               |
                                                               v
                                                   +-----------+-----------+
                                                   | Save to FileChild Table|
                                                   | (parent_id, embedding)|
                                                   +-----------------------+

                                [ SEARCH FLOW ]
                                
[User Question] -> [Embed Vector] -> [Query FileChild (Cosine)] -> [JOIN FileParent] -> [Deduplicate Context] -> [Gemini LLM]
```

---

## 🚀 Key Engineering Implementations

### 1. Parent-Child RAG Schema (Relational Context Mapping)
To solve the classic RAG dilemma (small chunks match well but lack context; large chunks match poorly but have rich context), this system splits data into two linked PostgreSQL tables using **SQLAlchemy**:
* **Parents**: Large semantic paragraphs stored in the database without vectors.
* **Children**: Individual sentences mapped to the Parent ID, embedded using a 384-dimensional vector (`all-MiniLM-L6-v2`).
* *Retrieval*: The system queries child vectors to find the most relevant sentences, joins the parent table, and retrieves the **complete parent paragraph** as context for the LLM.

### 2. Non-Blocking CPU Concurrency (`asyncio.to_thread`)
In traditional setups, calling heavy local model calculations (like generating sentence embeddings via PyTorch) blocks FastAPI's single-threaded event loop, freezing the entire server. This system wraps all model encoding calls in `asyncio.to_thread()`, offloading CPU-bound tasks to a background thread pool to handle multiple concurrent requests smoothly.

### 3. Level 4 Punctuation-Robust Chunker
Rather than using naive regex splitters (which break on decimal numbers like `3.14` or abbreviations like `e.g.`), this system integrates the **NLTK `sent_tokenize`** library. It uses a **Hybrid Similarity Threshold** (relative 85th percentile distance bound by a minimum cosine distance of `0.35`) to prevent over-chopping short, single-topic paragraphs.

### 4. Production-Grade Telemetry & Error Tracking
* **Langfuse**: Captures complete trace telemetry (token count, latency, generation costs) for both the database vector searches and the final Gemini generation.
* **Sentry**: Integrated into the FastAPI app to track exceptions and runtime server crashes.
* **Depends(get_db)**: Enforces dependency injection to guarantee database connection closure and prevent leaks.

---

## 🛠️ Technology Stack

* **Framework**: FastAPI (Python)
* **Vector Database**: PostgreSQL with `pgvector` extension
* **ORM**: SQLAlchemy
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local Execution)
* **LLM**: Google Gemini 1.5 Flash
* **Telemetry**: Langfuse SDK & Sentry SDK
* **Libraries**: NLTK (Sentence Tokenization), aiofiles (Async file I/O)

---

## ⚙️ Local Setup

### 1. Environment Variables
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vectors_db
GEMINI_API_KEY=your_gemini_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
SENTRY_DSN=your_sentry_dsn
```

### 2. Run with Docker Compose
To spin up the PostgreSQL database container:
```bash
docker-compose up -d
```

### 3. Run the Backend Locally
Install dependencies:
```bash
pip install -r requirements.txt
```

Launch the FastAPI dev server:
```bash
uvicorn main:app --reload
```
