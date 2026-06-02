FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api-v2.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api-v2.txt

COPY . .

ENV YOLO_MODEL_PATH="/app/runs/detect/train4/weights/best_clean.pt"
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["python", "api_v2.py"]