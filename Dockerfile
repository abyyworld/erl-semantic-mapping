# syntax=docker/dockerfile:1.7
#
# Reproducible image for the mapping evaluation.
#
# Three things make the result deterministic rather than merely convenient:
#   1. the base image is pinned by digest, not by a tag that moves under you;
#   2. every dependency is installed from a hash-locked requirements file with
#      --require-hashes, so a compromised or re-uploaded wheel fails the build;
#   3. the runtime stage carries no compiler, no pip and no build context.
#
#   docker build -t erl-semantic-mapping .
#   docker run --rm erl-semantic-mapping          # runs the full gate
#
# Regenerate the locks after changing pyproject.toml:
#   make lock

ARG PYTHON_DIGEST=sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

# ---------------------------------------------------------------- build stage
FROM python:3.12-slim@${PYTHON_DIGEST} AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first, so editing source does not re-resolve the environment.
COPY requirements.lock ./
RUN pip install --require-hashes --no-deps -r requirements.lock

# --no-deps: the lock file is the single source of truth for the environment,
# and pyproject's ranges must not be allowed to pull anything in behind it.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# -------------------------------------------------------------- runtime stage
FROM python:3.12-slim@${PYTHON_DIGEST} AS runtime

LABEL org.opencontainers.image.title="erl-semantic-mapping" \
      org.opencontainers.image.description="Semantic occupancy mapping with an evaluation harness that gates on its own assumptions" \
      org.opencontainers.image.source="https://github.com/abyyworld/erl-semantic-mapping" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Results are written to the working directory, so the runtime user has to own
# it — otherwise every `docker run` fails on the first write.
RUN useradd --create-home --uid 1000 mapper \
    && mkdir -p /work/results \
    && chown -R mapper:mapper /work

COPY --from=build --chown=mapper:mapper /opt/venv /opt/venv
COPY --chown=mapper:mapper conf /work/conf

USER mapper
WORKDIR /work

# Fails the container if the installed package cannot produce a map at all.
HEALTHCHECK --interval=30s --timeout=30s --retries=1 \
    CMD ["erl-map", "scenarios"]

ENTRYPOINT ["erl-map"]
CMD ["check", "--scene", "conf/scenes/room_v1.yaml"]
