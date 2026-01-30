FROM python:3.11-slim
WORKDIR /app
COPY image-service /app/image-service
WORKDIR /app/image-service
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
