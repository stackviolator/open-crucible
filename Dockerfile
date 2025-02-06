# Use an official PyTorch image with CUDA support.
FROM pytorch/pytorch:1.13.1-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies.
# (This image already has torch installed, but you can add other dependencies.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Expose the port FastAPI will run on.
EXPOSE 8000

# Run the FastAPI app using Uvicorn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

