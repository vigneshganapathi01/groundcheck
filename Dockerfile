# GroundCheck on Hugging Face Spaces (Docker SDK). Also runs anywhere Docker does:
#   docker build -t groundcheck . && docker run -p 7860:7860 groundcheck
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    GROUNDCHECK_PRELOAD=1 \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch (~200 MB instead of ~2 GB with CUDA), then the app's dependencies.
COPY app/requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

USER user
# Bake the base model into the image so the first check doesn't wait for a download.
RUN python -c "from transformers import AutoTokenizer, AutoModelForTokenClassification as M; \
m='KRLabsOrg/lettucedect-base-modernbert-en-v1'; AutoTokenizer.from_pretrained(m); M.from_pretrained(m)"

COPY --chown=user app/ ./app/
EXPOSE 7860
CMD ["python", "app/server.py"]
