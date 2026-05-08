FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch for smaller image; remove flag for GPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Train TF-IDF model at build time so container is ready to serve
RUN python src/train.py --model tfidf --data data/answers.csv

EXPOSE 5000

ENV CAD_MODEL=tfidf
ENV FLASK_ENV=production

CMD ["python", "ui/app.py"]
