# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY ./app /app

# Install system dependencies (if needed, for PIL or other libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
# RUN apt-get update --allow-insecure-repositories && \
#     apt-get install -y --no-install-recommends \
#         libgl1-mesa-glx && \
#     rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5002

# Command to run the Flask app
CMD ["python", "app.py"]
