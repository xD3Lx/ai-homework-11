# Personal Finance Coach — Streamlit demo image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    FINANCE_COACH_DB=/tmp/finance_coach.db \
    FINANCE_COACH_NOW=2025-11-30

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code + data
COPY src/ ./src/
COPY app/ ./app/
COPY evals/ ./evals/
COPY starter/ ./starter/

EXPOSE 8080

# Streamlit health endpoint for Fly checks: /_stcore/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py"]
