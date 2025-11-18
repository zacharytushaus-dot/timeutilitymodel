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

# Streamlit specific: expose the port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# The Command to run the app
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0