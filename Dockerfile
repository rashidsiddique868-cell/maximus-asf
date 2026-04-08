FROM python:3.11-slim

# Metadata
LABEL maintainer="OpenEnv Hackathon"
LABEL description="Autonomous Traffic Control OpenEnv"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY traffic_env.py .
COPY server.py .
COPY openenv.yaml .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Environment defaults
ENV TRAFFIC_TASK=basic_flow
ENV HOST=0.0.0.0
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Launch
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
