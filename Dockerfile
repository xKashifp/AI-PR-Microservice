FROM python:3.11-slim

WORKDIR /app

# System deps for faiss + torch
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required directories
RUN mkdir -p data/faiss_index ml/models

# Train classifier at build time
RUN python -m ml.train_classifier

# Open permissions on /app for runtime DB/index writing
RUN chmod -R 777 /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
