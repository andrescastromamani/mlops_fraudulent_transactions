# Oficial Python image
FROM python:3.10.21-slim

# Dont generate .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Workdir for the application
WORKDIR /app

# Install dependencies for building Python packages and other necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to install dependencies
COPY requirements.txt .

# Install dependencies from environment.yml using pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . /app

# Install the local package in editable mode
RUN pip install --no-cache-dir -e .

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "mlops_fraudulent_transactions.api:app", "--host", "0.0.0.0", "--port", "8000"]