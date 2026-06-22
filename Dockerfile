FROM python:3.12-slim

WORKDIR /app

# Install the package (offline: numpy only, no model downloads, no API keys).
COPY pyproject.toml README.md ./
COPY tuneforge ./tuneforge
COPY evals ./evals
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run the full offline optimizer benchmark and print the results.
CMD ["python", "-m", "evals.harness"]
