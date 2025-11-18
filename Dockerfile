# Use a lightweight Python Linux image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install basic system tools (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (this makes re-builds faster by caching installed packages)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Streamlit specific: expose the port
EXPOSE 8501

# Healthcheck to tell the cloud provider the app is alive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# The Command to run the app
# We map the cloud's dynamic PORT to Streamlit's server port
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0