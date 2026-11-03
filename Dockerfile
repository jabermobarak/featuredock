FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FD_DB=/data/events.db
WORKDIR /app
COPY pyproject.toml README.md ./
COPY featuredock ./featuredock
RUN python -m pip install --no-cache-dir . \
    && useradd --uid 10001 --create-home worker \
    && mkdir /data && chown worker:worker /data
COPY artifacts ./artifacts
COPY docker-entrypoint.sh ./
USER worker
EXPOSE 8020
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8020/ready', timeout=3)"
CMD ["sh", "docker-entrypoint.sh"]
