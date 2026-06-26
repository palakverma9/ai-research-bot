import os
import aiofiles
from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, text
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel

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


async def split_text(text, chunk_size=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks


async def text_to_vector(chunks):
    vectors = model.encode(chunks)
    return vectors.tolist()


# creates a folder to store the uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = FastAPI()


@app.post("/uploads/")
async def upload_file(file: UploadFile = File(...)):
    filelocation = os.path.join(UPLOAD_FOLDER, file.filename)

    async with aiofiles.open(filelocation, "wb") as buffer:
        while content := await file.read(1024 * 1024):
            await buffer.write(content)

    reader = PdfReader(filelocation)
    text_content = ""

    for page in reader.pages:
        text_content += page.extract_text() or ""

    chunks = await split_text(text_content)
    file_vectors = await text_to_vector(chunks)

    db = SessionLocal()

    for vector, chunk in zip(file_vectors, chunks):
        file_vector = FileVector(file_name=file.filename, embedding=vector, text=chunk)
        db.add(file_vector)

    db.commit()
    db.close()

    return {"filename": file.filename, "saved_chunks": len(chunks)}


@app.get("/users/")
async def get_all_chunks():
    db = SessionLocal()
    chunks = db.query(FileVector).all()
    db.close()
    return [
        {
            "id": chunk.id,
            "text": chunk.text,
            # .tolist() converts the NumPy array to a standard list so FastAPI doesn't crash
            "embedding": chunk.embedding.tolist() if chunk.embedding is not None else None
        }
        for chunk in chunks
    ]


@app.post("/find_similiar_chunks/")
async def find_similiar_chunks(question: QueryRequest):
    question_vector = (await text_to_vector([question.question]))[0]

    db = SessionLocal()

    distance_fn = FileVector.embedding.cosine_distance(question_vector)
    results = db.query(FileVector, distance_fn).order_by(distance_fn).limit(3).all()
    db.close()
    
    similarities = []
    for row, dist in results:
        similarity_score = 1.0 - float(dist)
        similarities.append({
            "id": row.id,
            "file_name": row.file_name,
            "text": row.text,
            "similiarity": similarity_score
        })

    return similarities