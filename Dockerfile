# Wardrobe Intelligence Lab: self-contained image for Hugging Face Spaces (Docker SDK) or any container host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOME=/home/user LOOKROOM_DATA_DIR=/tmp/wil-data HF_HOME=/tmp/hf-cache \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY --chown=user . .
USER user

# Bake pretrained models into the image at build time, so runtime never depends on the Hub.
# A failed download does not fail the build: the app degrades to its documented fallbacks.
# (BiRefNet is used offline for catalog curation only: ~45 s/image on 2 vCPU is too slow for requests.)
RUN (python scripts/setup_models.py || echo "WARN: vision models unavailable -> heuristic fallback") \
 && rm -rf /tmp/hf-cache

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--server.maxUploadSize=5"]
