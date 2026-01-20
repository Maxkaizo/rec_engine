
FROM python:3.12-slim

# Install system dependencies for C-extensions (implicit, surprise)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --frozen ensures we use the exact versions from uv.lock
# --no-cache reduces image size
RUN uv sync --frozen --no-cache

# Place the virtual environment's bin at the front of the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code and models
# Note: In a real CI/CD, we might not copy 'data/' if it's huge, 
# but for the capstone we need the models to be present.
COPY src/ ./src/
COPY models/ ./models/

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
