FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "if [ -z \"$PORT\" ] || [ \"$PORT\" = \"undefined\" ]; then PORT=8000; fi; uv run uvicorn main:app --host 0.0.0.0 --port $PORT"]
