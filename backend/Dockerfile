FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv /opt/director-venv
ENV PATH="/opt/director-venv/bin:$PATH"
COPY pyproject.toml ./
COPY app ./app
RUN pip install --upgrade pip setuptools wheel \
    && pip install .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/director-venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 director \
    && useradd --system --uid 10001 --gid director --home-dir /app --shell /usr/sbin/nologin director \
    && mkdir -p /app /data /tmp/director \
    && chown -R director:director /app /data /tmp/director

WORKDIR /app
COPY --from=builder /opt/director-venv /opt/director-venv
COPY --chown=director:director app ./app
COPY --chown=director:director alembic ./alembic
COPY --chown=director:director alembic.ini ./alembic.ini
COPY --chown=director:director pyproject.toml ./pyproject.toml

USER director
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
