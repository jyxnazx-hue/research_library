# Use official Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy everything from your local directory into the container
COPY . .

# Install dependencies and your local project as a package
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Expose the port defined in your openenv.yaml
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]