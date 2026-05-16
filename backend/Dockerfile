FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy project specification files first to cache dependencies
COPY pyproject.toml uv.lock* ./

# Install project dependencies
RUN uv sync --frozen || uv sync

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# The default command runs the FastAPI backend. 
# Railway provides a $PORT environment variable dynamically.
CMD uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
