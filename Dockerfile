# fleet-v2 coordinator image (box1). One Python image, no build step for the dashboard.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY fleet2/ ./fleet2/
COPY coordinator/ ./coordinator/
COPY config/ ./config/
COPY deploy/ ./deploy/

RUN pip install --no-cache-dir ".[coordinator]"

ENV FLEET_BIND=0.0.0.0:8090 \
    FLEET_DEPLOY_DIR=/app/deploy \
    FLEET_LIMITS_FILE=/app/config/limits.toml

EXPOSE 8090

CMD ["python", "-m", "coordinator.main"]
