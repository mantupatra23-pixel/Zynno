# Dockerfile (project root)
FROM python:3.11-slim

# install system deps including tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

# create app dir
WORKDIR /app

# copy files
COPY pyproject.toml requirements.txt /app/
# install python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

# expose port (Render sets $PORT env var)
ENV PYTHONUNBUFFERED=1

# start app (adjust module:path if different)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
