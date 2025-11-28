# Use slim Python image to save memory
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (leverages Docker cache)
COPY requirements.txt .

# Install dependencies (no cache to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port for Render
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
