FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/
COPY app /app/app
COPY data /app/data
COPY config.yaml /app/config.yaml

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
