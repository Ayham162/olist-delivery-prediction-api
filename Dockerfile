# python:3.12-slim (minor-version pin, not an exact patch like 3.12.1-slim)
# — matches the interpreter this was developed against (3.12.1) while still
# getting Debian security patches on rebuild, which an exact-patch pin won't.
FROM python:3.12-slim

WORKDIR /app

# Runtime deps only — requirements-dev.txt (pytest/black/flake8) has no
# business in a production image. Copied before the rest of the source so
# this layer caches independently of code changes (the standard Docker
# speedup: `pip install` only reruns when requirements.txt itself changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Editable install (-e), deliberately, even in this "final" image: config.py
# resolves PROJECT_ROOT from its own __file__ location relative to src/,
# which only stays correct if the package keeps living under /app/src rather
# than getting physically copied into site-packages (a real, non-editable
# `pip install .` did exactly that and broke config/model/data path
# resolution at container startup — caught by actually running the
# container, not just building it).
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

COPY app/ app/
COPY config/ config/
COPY models/ models/
COPY data/ data/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
