FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt numba
COPY config /app/config
COPY scenarios /app/scenarios
COPY backend /app/backend
WORKDIR /app/backend
ENV PYTHONPATH=/app/backend PYTHONUTF8=1
# train the card classifier at build time so the API starts fast
RUN python -m evaluation.classifier_eval > /dev/null
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
