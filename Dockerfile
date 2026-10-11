# Engineering document QA API.
#   docker build -t rag .
#   docker run --rm -p 8000:8000 --env-file .env -v "$PWD/data:/app/data:ro" rag
# The index is built beforehand (python -m rag.cli build-index) and mounted; the
# API key comes from --env-file at run time and is never part of the image.
# The meshpilot package is installed too, so the code imports as in the
# repository, but no checkpoint, mesh or prediction enters the image and no
# model runs during the build.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home app
WORKDIR /app

# Dependencies first, so these layers are reused when only the code changes.
# CPU-only torch: the default PyPI wheel bundles CUDA libraries the image does not need.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# The other pinned dependencies are read from pyproject.toml. The project itself
# is installed only after its code is copied, so setuptools checks the real
# package directories instead of placeholders.
COPY pyproject.toml ./
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; print('\n'.join(p['dependencies'] + p['optional-dependencies']['embed']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# The embedding model is downloaded at build time, so the container starts
# without reaching Hugging Face. Answers still need the Anthropic API.
USER app
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"
ENV HF_HUB_OFFLINE=1

# The code and the corpus manifest, installed in place: rag.config finds the
# manifest next to the code. The PDFs and the index stay outside the image.
USER root
COPY README.md ./
COPY --chown=app rag ./rag
COPY --chown=app meshpilot ./meshpilot
COPY --chown=app corpus ./corpus
RUN pip install --no-deps -e .
USER app
ENV RAG_DATA_DIR=/app/data

EXPOSE 8000
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
