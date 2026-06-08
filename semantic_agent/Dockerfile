FROM python:3.12-slim
WORKDIR /app/semantic_agent
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# HF_HOME points to a Docker volume so the model is cached across runs
# and never re-downloaded during docker compose build
ENV HF_HOME=/app/hf_home
COPY . .
RUN chown -R 1000:1000 /app
ENV VECTOR_STORE_URL=http://localhost:8010
ENV REGISTRY_PATH=/tools_registry
CMD ["sh", "-c", "python mcp_server.py & uvicorn a2a_server:app --host 0.0.0.0 --port 8088"]
