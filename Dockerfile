# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy app
COPY . /app

# Install python deps
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install fastapi uvicorn[standard] python-multipart openpyxl imageio torch torchvision pillow SimpleITK pydicom pandas SQLAlchemy psycopg2-binary

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
