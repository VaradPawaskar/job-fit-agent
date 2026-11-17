FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY ui ./ui

RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000 8501

# Default: the API. docker-compose overrides the command for the UI.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
