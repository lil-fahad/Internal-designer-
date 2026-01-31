# AI Interior Designer Docker Image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/saved models/checkpoints data/cache

# Expose port
EXPOSE 5000

# Set default environment variables
ENV HOST=0.0.0.0
ENV PORT=5000
ENV DEBUG=False

# Run the application
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5000", "--workers", "4"]
