FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . /app
ENV PORT=8000
EXPOSE 8000
ENTRYPOINT ["python", "-m", "server.main"]
CMD ["--host", "0.0.0.0"]
