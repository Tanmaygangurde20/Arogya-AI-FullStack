# Use official Python image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]