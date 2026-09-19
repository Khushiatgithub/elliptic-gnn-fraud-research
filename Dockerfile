# Multi-stage Dockerfile for FastAPI backend + React static frontend

# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend runtime
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and results
COPY backend/ ./backend/
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/
COPY data/ ./data/

# Copy built frontend assets from stage 1 into backend static
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port (default 8000, customizable via PORT env var)
ENV PORT=8000
EXPOSE 8000

# Run FastAPI via Uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
