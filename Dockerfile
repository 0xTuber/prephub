# Course Builder Docker Image
# Build: docker build -t course-builder .
# Run: docker run -v ./data:/app/data -e GOOGLE_API_KEY=your-key course-builder generate --certification "Your Cert"

FROM python:3.11-slim

# Install system dependencies for MinerU and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default data directory
VOLUME ["/app/data"]

# Entry point
ENTRYPOINT ["course-builder"]
CMD ["--help"]
