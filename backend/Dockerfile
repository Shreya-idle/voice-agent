FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build-essential, curl, portaudio for pyaudio)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock* ./

# Sync dependencies using uv (ignoring dev dependencies)
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command starts the FastAPI app using uv
CMD uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
