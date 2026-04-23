FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-editable 2>/dev/null || uv sync --no-dev --no-editable

# Copy server source and config
COPY server/ ./server/
COPY config.yaml ./

EXPOSE 8090

CMD ["uv", "run", "python", "-m", "server.main"]
