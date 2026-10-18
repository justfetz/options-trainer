# Options trainer: tested pricing engine + retrieval + (later) an evaluated agent.
# Listens on 0.0.0.0:$PORT (7861 by default).
# Required env for hosting: PUBLIC_DEMO_ORIGIN (the site's own https origin). Optional:
# PUBLIC_DEMO_EMBED_ORIGINS. The corpus index is NOT in the image; mount it from a private bucket.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7861
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && mkdir -p /app/artifacts && chown -R app:app /app

USER app
EXPOSE 7861
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7861\")}/api/health').read()" || exit 1
CMD ["python", "-m", "trainer.server"]
