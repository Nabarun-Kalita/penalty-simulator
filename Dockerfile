# Dockerfile for deploying the Penalty Simulator API on Hugging Face Spaces
# HF Spaces uses port 7860 by convention for Docker spaces, not 8000.

FROM python:3.12-slim

# Avoid Python writing .pyc files and ensure logs flush immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for some Python packages (xgboost needs libgomp, Pillow may need libjpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a user with UID 1000 for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

# Install Python dependencies first (better Docker layer caching)
COPY --chown=user:user requirements.txt .
RUN pip install --user --upgrade pip && \
    pip install --user -r requirements.txt

# Copy the rest of the project
COPY --chown=user:user . .

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

# Start the FastAPI app via uvicorn on port 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
