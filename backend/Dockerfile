FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file to cache dependencies
COPY requirements.txt ./

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# The default command runs the FastAPI backend. 
# Railway provides a $PORT environment variable dynamically.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
