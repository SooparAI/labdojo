# Lab Dojo Production Dockerfile
# Multi-stage build: Node.js for React UI, Python for FastAPI server

# Stage 1: Build the React UI
FROM node:22-slim AS ui-builder
WORKDIR /app/labdojo-ui
RUN npm install -g pnpm
COPY labdojo-ui/package.json labdojo-ui/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY labdojo-ui/ ./
RUN pnpm build

# Stage 2: Python runtime
FROM python:3.12-slim AS runtime
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir aiohttp fastapi uvicorn pydantic

# Copy application code
COPY labdojo.py science_catalog.json ./
COPY labdojo/ ./labdojo/

# Copy built React UI
COPY --from=ui-builder /app/labdojo-ui/dist ./labdojo-ui/dist

# Expose port (Railway sets PORT env var)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/status')" || exit 1

# Run the server
CMD ["python", "labdojo.py"]
