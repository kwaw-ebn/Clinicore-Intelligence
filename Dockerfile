# Root-level Dockerfile to build the full app (frontend served by Flask static)
FROM python:3.11-slim

WORKDIR /app

# system deps for xgboost sometimes need build tools - keep minimal
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip
RUN pip install -r backend/requirements.txt

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["gunicorn", "backend.server:app", "--bind", "0.0.0.0:5000", "--workers=2"]
