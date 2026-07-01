FROM python:3.11-slim

# Install system dependencies required for compiling psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

EXPOSE 8000

# Start Uvicorn pointing to main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]