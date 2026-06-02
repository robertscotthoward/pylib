# pylib — Product Requirements Document

## Overview

**pylib** is a reusable Python 3.12 library and CLI for AI-powered document processing, RAG, workflow automation, and database management. It is consumed by other projects via editable installs or Git tags.

---

## Features

### convert (Added: 2026-05-01)

* **Context / Why**: Projects need to convert legacy office documents (PDF, DOCX, DOC, EPUB, RTF) into LLM-friendly markdown before ingestion.
* **Purpose / What**: Recursively scans a folder for convertible files and produces a sibling `.md` file for each. Files that already have a `.md` sibling are skipped.
* **Usage / How**: `uv run -m src.main convert FOLDER [--filter ".pdf|report*.docx"]`

---

### requirements (Added: 2026-06-02)

* **Context / Why**: Meeting transcripts and design documents contain scattered, implicit requirements. Manually extracting and deduplicating them is slow and error-prone.
* **Purpose / What**: For each `*.md` file in a folder, invokes an LLM to extract structured requirements into a `*.requirements.md` sibling. Once all per-file extractions are complete, consolidates them into a single topologically-ordered `requirements.md` with unique IDs, dependency declarations, acceptance criteria, and source traceability.
* **Usage / How**: `uv run -m src.main requirements FOLDER [--model-class ollama|bedrock] [--model MODEL] [--host HOST] [--region REGION] [--force]`

---

## Ad-hoc & Experimental Features

*(None yet)*
