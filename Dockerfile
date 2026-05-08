FROM python:3.12-slim

# soundfile needs libsndfile1; librosa needs ffmpeg (via audioread) for MP3 decode.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Stream stdout/stderr in real time so the launch URL appears immediately.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV OUTPUT_DIR=/app/outputs
RUN mkdir -p /app/outputs

EXPOSE 7860
CMD ["python", "-m", "src.app"]
