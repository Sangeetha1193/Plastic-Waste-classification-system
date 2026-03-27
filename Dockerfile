# -----------------------------------------------------------------------------
# Stage 1 - builder
# Install Python deps into an isolated layer so the final image stays lean.
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /install

# System libs needed to compile some Python wheels (opencv, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix folder — copied into the final image cleanly
RUN pip install --upgrade pip && \
    pip install --prefix=/deps --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2 - runtime
# Lean image - no build tools, no pip cache.
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Runtime system libs only (opencv needs libglib, torch needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /deps /usr/local

WORKDIR /app

# Copy application code
COPY app.py .
COPY templates/      templates/
COPY static/         static/

# Copy model weights
# If best.pt is large (>100 MB), consider mounting via a volume instead:
#   docker run -v $(pwd)/models:/app/models waste-api
COPY models/best.pt  models/best.pt

# Create output dirs the app may write to
RUN mkdir -p outputs/predictions outputs/evaluation

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Environment (overridable at runtime with -e flags)
ENV MODEL_PATH=models/best.pt \
    CONF_THRESH=0.25 \
    IOU_THRESH=0.45 \
    PORT=5000 \
    FLASK_DEBUG=false \
    # Stops ultralytics from trying to hit the internet on startup
    YOLO_OFFLINE=1 \
    # Prevent OpenCV from looking for a display (headless server)
    DISPLAY="" \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Gunicorn for production (more robust than Flask dev server)
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
