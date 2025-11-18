# Dockerfile (project root)
FROM python:3.11-slim

# avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# system deps needed for Tesseract, Pillow, and other libs
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libtiff5-dev \
    libpng-dev \
    libtesseract-dev \
    tesseract-ocr \
    libleptonica-dev \
    libicu-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# create app dir
WORKDIR /app

# copy only lock/files first for better cache
COPY pyproject.toml requirements.txt /app/

# upgrade pip + install python deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# copy whole project
COPY . /app

# expose port (Render sets $PORT)
ENV PORT=8000

# start the app (adjust module:path if different)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --loop=uvloop --http=httptools"]
