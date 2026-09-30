# Use the STABLE "Bookworm" version of Debian to avoid "Trixie" errors
FROM python:3.11-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Install basic system tools
# We removed 'software-properties-common' to fix the build error
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port (Railway will map $PORT dynamically)
EXPOSE 8000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

# Command to run the high-performance FastAPI engine
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]