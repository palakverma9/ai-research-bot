import re
import os
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, text
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from dotenv import load_dotenv
load_dotenv()

# Sentry Instrumentation
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[FastApiIntegration()],
        traces_sample_rate=1.0,
    )

# Langfuse Instrumentation
from langfuse import observe, get_client, propagate_attributes
langfuse_client = get_client()

from google import genai
ai_client = genai.Client()

model = SentenceTransformer('all-MiniLM-L6-v2')

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectors_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

# row that will be stored in database, each row will have an id, file_name, embedding and text
class FileVector(Base):
    __tablename__ = "file_vectors"

    id = Column(Integer, primary_key=True)
    file_name = Column(String)
    embedding = Column(Vector(dim=384))
    text = Column(String)

Base.metadata.create_all(bind=engine)

class QueryRequest(BaseModel):
    question: str



async def split_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1:
        return sentences
    
    sentence_embeddings = model.encode(sentences)

    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        vec1 = sentence_embeddings[i].reshape(1, -1)
        vec2 = sentence_embeddings[i + 1].reshape(1, -1)
        sim = cosine_similarity(vec1, vec2)[0][0]
        similarities.append(sim)
    
    distances = [1.0 - sim for sim in similarities]

    threshold = np.percentile(distances, 85)

    chunks = []
    current_chunk = [sentences[0]]
    for i in range(len(sentences) - 1):
        distance = distances[i]
        
    
        if distance > threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

    


async def text_to_vector(chunks):
    vectors = model.encode(chunks)
    return vectors.tolist()



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = FastAPI()


@app.post("/uploads/")
@observe(name="pdf_indexing")
async def upload_file(file: UploadFile = File(...)):
    with propagate_attributes(user_id="palak_engineer"):
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Uploaded file is not in pdf format.")
        
        filelocation = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Span 1: Save File to Disk
        with langfuse_client.start_as_current_observation(name="save_file_to_disk", as_type="span") as span:
            try:
                async with aiofiles.open(filelocation, "wb") as buffer:
                    while content := await file.read(1024 * 1024):
                        await buffer.write(content)
                span.update(output={"status": "success"})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {str(e)}")

        # Span 2: Parse PDF Text
        with langfuse_client.start_as_current_observation(name="pdf_text_extraction", as_type="span") as span:
            try:
                reader = PdfReader(filelocation)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() or ""
                span.update(output={"character_count": len(text_content)})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")
        
        # Span 3: Semantic Chunking & Vector Embeddings
        with langfuse_client.start_as_current_observation(name="semantic_chunking_and_embedding", as_type="span") as span:
            try:
                chunks = await split_text(text_content)
                file_vectors = await text_to_vector(chunks)
                span.update(output={"chunks_count": len(chunks)})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise HTTPException(status_code=500, detail=f"Failed during vector embedding: {str(e)}")
        
        # Span 4: Database Save
        with langfuse_client.start_as_current_observation(name="database_save", as_type="span") as span:
            try:
                db = SessionLocal()
                for vector, chunk in zip(file_vectors, chunks):
                    file_vector = FileVector(file_name=file.filename, embedding=vector, text=chunk)
                    db.add(file_vector)
                db.commit()
                db.close()
                span.update(output={"status": "success"})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise HTTPException(status_code=500, detail=f"Database save operation failed: {str(e)}")

            return {"filename": file.filename, "saved_chunks": len(chunks)}


@app.get("/users/")
async def get_all_chunks():
    db = SessionLocal()
    chunks = db.query(FileVector).all()
    db.close()
    return [
        {
            "id": chunk.id,
            "file_name": chunk.file_name,
            "text": chunk.text,
            
        }
        for chunk in chunks
    ]


@app.post("/ask_question/")
@observe(name="ask_question_rag")
async def ask_question(question: QueryRequest):
    with propagate_attributes(user_id="palak_engineer"):
        # Span 1: Generate Embedding for the Question
        with langfuse_client.start_as_current_observation(name="generate_question_embedding", as_type="span") as span:
            try:
                question_vector = (await text_to_vector([question.question]))[0]
                span.update(output={"vector_dimensions": len(question_vector)})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to generate embedding for question: {str(e)}")

        # Span 2: PostgreSQL Vector Search
        with langfuse_client.start_as_current_observation(name="postgres_vector_search", as_type="span") as span:
            try:
                db = SessionLocal()
                distance_fn = FileVector.embedding.cosine_distance(question_vector)
                results = db.query(FileVector, distance_fn).order_by(distance_fn).limit(10).all()
                db.close()
                
                similarities = []
                seen_text = set()

                for row, dist in results:
                    normalised_text = row.text.strip()

                    if normalised_text not in seen_text:
                        seen_text.add(normalised_text)

                        similarity_score = 1.0 - float(dist)
                        similarities.append({
                            "id": row.id,
                            "file_name": row.file_name,
                            "text": row.text,
                            "similarity": similarity_score
                        })
                    if len(similarities) == 3:
                        break
                
                span.update(output={"retrieved_chunks": similarities})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise HTTPException(status_code=500, detail=f"Database search failed: {str(e)}")
        
        context_texts = [item["text"] for item in similarities]
        context = "\n---\n".join(context_texts)

        prompt = f"""You are a helpful structural steel design assistant. Use the following context retrieved from the uploaded documents to answer the user's question. 
    If a specific standard code parameter, formula, or safety factor (like γm0) is mentioned in the context but its numerical value is missing, you are allowed to use your pre-trained knowledge of the IS 800 steel code to fill in the constant and calculate the answer. State clearly if you used your pre-trained knowledge for a constant."

    Context:
    {context}
    User Question: {question.question}

    Answer:"""
        
        # Generation: Gemini LLM Call
        with langfuse_client.start_as_current_observation(
            name="gemini_generation",
            as_type="generation",
            model="gemini-2.5-flash",
            input=prompt,
        ) as generation:
            try:
                response = ai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                
                usage = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = {
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count,
                    }
                    
                generation.update(
                    output=response.text,
                    usage=usage
                )
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                raise HTTPException(status_code=500, detail=f"Gemini generation failed: {str(e)}")

        return {
            "answer": response.text,
            "context_used": similarities
        }
