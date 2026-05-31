FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
COPY . .
RUN chown -R 1000:1000 /app
ENV STORAGE_MODE=local
ENV RUNS_DIR=/runs
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
