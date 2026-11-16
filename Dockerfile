FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9696

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (also required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user fraud_pipeline.py predict.py ./
COPY --chown=user models/model.joblib models/model.joblib

EXPOSE 9696

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 120 predict:app"]
