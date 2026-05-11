# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pylib** is a reusable Python 3.12 library for AI-powered document processing, RAG (Retrieval-Augmented Generation), workflow automation, and database management. It is consumed by other projects via editable installs or Git tags.

## Commands

```bash
# Install dependencies
uv sync

# Sanity check
uv run python -c "from lib import tools; print('Success!')"

# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/lib/tools/test_tools.py

# Add a dependency
uv add <package-name>

# Fix dependency conflicts
uv sync --reinstall
```

## Architecture

The library is in `src/lib/` and organized into layers:

- **`tools.py`** — Foundation utility module used by everything. File I/O (`readText`, `readJson`, `readYaml`, etc.), nested dict access (`g()` / `gi()`), `deep_merge()`, `Spy` context manager for tracing, `NOW()` (mockable datetime), caching helpers, and type conversion utilities (`to_seconds()`, `from_metric()`, etc.).

- **`configurations.py`** — Environment-aware config loading. Reads `config.yaml` + `credentials.yaml`, detects environment via env var matching, and merges all layers. `get_config_credentials_environment()` is the main entry point.

- **`ai/`** — AI pipeline components:
  - `fileconvert.py` — Converts PDF, DOCX, EPUB, RTF, RDF → text/markdown
  - `corpus.py` — Enumerates a document folder for ingestion
  - `vectordb.py` / `vectordb_chroma.py` / `vectordb_numpy.py` — Abstract VectorDb base class with ChromaDB and NumPy backends; handles chunking, embedding, and incremental updates
  - `modelstack.py` — LLM abstraction; `ModelStack.from_config()` creates Ollama or AWS Bedrock instances; includes `clean_json()` and `clean_fence()` helpers for LLM output cleanup
  - `rag.py` — Wires VectorDb + ModelStack; `sync_corpus_to_vdb()` for indexing, `query()` for Q&A
  - `raptor.py` — RAPTOR hierarchical clustering/summarization algorithm
  - `rerankerlib.py` / `rerankers1.py` — Document reranking after retrieval

- **`database/`** — Persistence layer:
  - `databases.py` — Abstract `Database` base class + `SqliteDatabase` implementation
  - `migrations.py` — Schema migration generation; compares desired vs. actual schema and writes SQL DDL to `data/migrations/`

- **`workflow/engine.py`** — YAML-configured state machine engine with environment-based transitions, callback hooks, and save/restore state. See `docs/workflow.md` for format.

- **`book.py`** — Multi-format document reader (TXT, PDF, EPUB, MOBI, DOCX, RTF, etc.)

## Configuration

`config.yaml` defines `environments` (with test conditions for detecting prod vs local) and `all` (shared config merged into every environment). `credentials.yaml` holds secrets and is not committed. Both files are discovered by walking up the directory tree.

## Using pylib in Another Project

```bash
# Editable (live development)
uv add --editable ../pylib

# Pinned to a Git tag
uv add "pylib @ git+https://github.com/robertscotthoward/pylib@v1.0.0"

# Latest from GitHub main
uv add "pylib @ git+https://github.com/robertscotthoward/pylib@main"
uv lock --upgrade-package pylib
```

## Tests

Tests live in `tests/lib/` mirroring the `src/lib/` structure. Key suites: `tools/test_tools.py`, `workflow/test_engine.py`, `ai/test_raptor*.py`, `ai/vectordb/test_vectordb_chroma.py`.
