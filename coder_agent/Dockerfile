FROM python:3.11-slim
WORKDIR /app/coder_agent
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt websockets
COPY . .
RUN chown -R 1000:1000 /app
ENV SANDBOX_MODE=local
ENV SANDBOX_URL=http://sandbox:8080
CMD ["python", "-m", "uvicorn", "a2a_server:app", "--host", "0.0.0.0", "--port", "8089"]
