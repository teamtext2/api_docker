# Use python:3.10-slim as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Set environment variables
ENV HF_HOME=/code/huggingface
ENV PORT=7860

# Install git & curl for diagnostic purposes
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Qwen2.5-0.5B-Instruct tokenizer & model weights to build the cache.
# This prevents downloading 1GB from Hugging Face every time the container starts up or wakes up from sleep.
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# Ensure directory is writable by user 1000 (Hugging Face default runtime user)
RUN chmod -R 777 /code

# Copy the rest of the application files
COPY . .

# Expose Space port
EXPOSE 7860

# Start application
CMD ["python", "app.py"]
