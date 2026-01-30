FROM python:3.11-slim
WORKDIR /app
COPY postprocess-service /app/postprocess-service
WORKDIR /app/postprocess-service
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
