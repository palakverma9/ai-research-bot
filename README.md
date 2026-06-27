# AI Research Bot - RAG Backend

A FastAPI backend for Document Q&A using Retrieval-Augmented Generation (RAG).

## Features
* **PDF Processing**: Chunks and embeds text using `all-MiniLM-L6-v2`.
* **Vector DB**: Stores chunks and performs cosine search inside PostgreSQL (`pgvector`) in Docker.
* **Gemini integration**: Generates context-aware answers using `gemini-2.5-flash`.

## Setup

1. **Configure Environment**:
   Create a `.env` file:
   ```env
   GEMINI_API_KEY="YOUR_KEY"
   ```

2. **Start Database (Docker)**:
   ```bash
   docker compose up -d
   ```

3. **Install & Run**:
   ```bash
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

## API Docs
Visit `http://localhost:8000/docs` to upload PDFs and ask questions via the `/ask_question/` endpoint.
