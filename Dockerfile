# =============================================================================
# Stage 1: builder — install all Python dependencies
# =============================================================================
FROM python:3.11-slim AS builder

# Build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the package manifest first so layer is cached if deps don't change
COPY pyproject.toml ./

# Create a minimal stub so pip can resolve the package without the full source
RUN mkdir -p src && touch src/__init__.py

# Install all dependencies into a prefix directory that we can copy cleanly
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --prefix=/install --no-cache-dir -e . || \
    pip install --prefix=/install --no-cache-dir \
        pydantic>=2.0 \
        pyyaml>=6.0 \
        structlog>=24.0 \
        python-dotenv>=1.0 \
        polars>=1.0 \
        pandas>=2.2 \
        duckdb>=1.0 \
        ccxt>=4.0 \
        yfinance>=0.2 \
        newsapi-python>=0.2 \
        feedparser>=6.0 \
        aiohttp>=3.9 \
        websockets>=12.0 \
        pandas-ta>=0.3 \
        numpy>=1.26 \
        scipy>=1.12 \
        hmmlearn>=0.3 \
        anthropic>=0.40 \
        apscheduler>=3.10 \
        streamlit>=1.38 \
        plotly>=5.22 \
        python-telegram-bot>=21.0 \
        pytest>=8.0 \
        pytest-asyncio>=0.23 \
        pytest-cov>=5.0

# =============================================================================
# Stage 2: runtime — lean image with only what's needed to run
# =============================================================================
FROM python:3.11-slim AS runtime

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd --gid 1000 trader && \
    useradd --uid 1000 --gid trader --shell /bin/bash --create-home trader

WORKDIR /app

# Copy application source — only the directories needed at runtime
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY config/    ./config/
COPY dashboard/ ./dashboard/

# Create data directory and set ownership
RUN mkdir -p /app/data/logs && \
    chown -R trader:trader /app

USER trader

# Python path so `from src.x import y` resolves correctly
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Streamlit listens on 8501; health check server on 8080
EXPOSE 8501 8080

# Health check — hits the lightweight health endpoint started by run_multiagent.py
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "scripts/run_multiagent.py", "--paper"]
