FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Editable install on purpose: config.py finds config/ relative to the source tree.
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install -e .

# One-time seed: if the volume has no database yet, start from the bundled snapshot so
# the seen-sets and pending candidates survive the move. Never overwrites a live DB.
COPY seed ./seed

ENV SCOUT_DB_PATH=/data/scout.db

CMD ["sh", "-c", "mkdir -p /data && { [ -f \"$SCOUT_DB_PATH\" ] || { [ -f seed/scout.db ] && cp seed/scout.db \"$SCOUT_DB_PATH\" && echo 'seeded database from snapshot'; } || true; } && exec scout run"]
