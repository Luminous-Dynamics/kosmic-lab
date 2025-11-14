# Kosmic Lab - Production Docker Image
# Multi-stage build for minimal final image size

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.11-slim as builder

LABEL maintainer="Kosmic Lab Team <kosmic-lab@example.org>"
LABEL description="Kosmic Lab - AI-Accelerated Consciousness Research Platform"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without dev dependencies for production)
RUN poetry install --only main --no-root && \
    rm -rf $POETRY_CACHE_DIR

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim as runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash kosmic && \
    mkdir -p /app /data /logs && \
    chown -R kosmic:kosmic /app /data /logs

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=kosmic:kosmic /app/.venv /app/.venv

# Copy application code
COPY --chown=kosmic:kosmic . .

# Set environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

# Switch to non-root user
USER kosmic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import core; import fre; print('healthy')" || exit 1

# Expose dashboard port
EXPOSE 8050

# Default command: show help
CMD ["python", "-c", "print('🌊 Kosmic Lab Container Ready!\\n\\nAvailable commands:\\n  python examples/01_hello_kosmic.py\\n  make fre-run\\n  make dashboard\\n  make help\\n')"]
