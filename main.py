import os
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, text
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel

#for semantic chunking
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from dotenv import load_dotenv
load_dotenv()

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
async def upload_file(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file is not in pdf format.")
    
   
    filelocation = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        async with aiofiles.open(filelocation, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                await buffer.write(content)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {str(e)}")

    
    try:
        reader = PdfReader(filelocation)
        text_content = ""

        for page in reader.pages:
          text_content += page.extract_text() or ""

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {str(e)}")
    
    try:
        chunks = await split_text(text_content)
        file_vectors = await text_to_vector(chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed during vector embedding: {str(e)}")
    
    try:
        db = SessionLocal()
        for vector, chunk in zip(file_vectors, chunks):
            file_vector = FileVector(file_name=file.filename, embedding=vector, text=chunk)
            db.add(file_vector)
        db.commit()
        db.close()
    except Exception as e:
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
async def ask_question(question: QueryRequest):
    question_vector = (await text_to_vector([question.question]))[0]

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
                "id":row.id,
                "file_name":row.file_name,
                "text":row.text,
                "similiarity":similarity_score
            })
        if len(similarities) == 3:
            break
    
    context_texts = [item["text"] for item in similarities]
    context = "\n---\n".join(context_texts)


    prompt = f"""You are a helpful structural steel design assistant. Use the following context retrieved from the uploaded documents to answer the user's question. 
If a specific standard code parameter, formula, or safety factor (like γm0) is mentioned in the context but its numerical value is missing, you are allowed to use your pre-trained knowledge of the IS 800 steel code to fill in the constant and calculate the answer. State clearly if you used your pre-trained knowledge for a constant."

Context:
{context}
User Question: {question.question}

Answer:"""
    

    response = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return {
        "answer": response.text,
        "context_used": similarities
    }

