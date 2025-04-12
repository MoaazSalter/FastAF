# Use an official Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV and general use
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files into the image
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the gRPC port
EXPOSE 50051

# Run your server
CMD ["python", "server.py"]
