FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY data /app/data

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "entity_linking_agent.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
