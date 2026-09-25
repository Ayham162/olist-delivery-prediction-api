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
# --timeout 120: pip's default (15s per read) is too tight for some of these
# wheels (scikit-learn, pandas, mlflow's deps) over a slower connection —
# hit real ReadTimeoutErrors against files.pythonhosted.org building this,
# not a hypothetical concern.
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

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

# models/ and data/ are DVC-tracked, not committed to git — a genuinely
# fresh `git clone` doesn't have them (confirmed by actually cloning into a
# scratch directory and checking, not assumed), so `COPY models/ models/`
# from the host build context would fail on any machine but this one, where
# they already happen to exist locally from an earlier `dvc pull`/copy. The
# local DVC remote (../dvc-storage) is a path on this machine only, same
# problem CI hit (see .github/workflows/ci.yml) — fetched from the same
# GitHub Release asset here instead, so the image is self-sufficient on a
# truly clean clone with nothing more than `docker compose up --build`.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fL https://github.com/Ayham162/olist-delivery-prediction-api/releases/download/ci-fixtures-v1/ci-artifacts.tar.gz \
       | tar -xz models data \
    && apt-get purge -y curl && apt-get autoremove -y

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
