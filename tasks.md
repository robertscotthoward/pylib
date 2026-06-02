# pylib — Task Tracker

## Features

- [x] Feature: convert
  - [x] Typer CLI scaffold (`src/main.py`)
  - [x] Recursive folder scan with extension filter
  - [x] PDF / DOCX / DOC / EPUB / RTF → markdown via `fileconvert.py`
  - [x] Skip files that already have a `.md` sibling
  - [x] Summary output (converted / skipped / errors)

- [ ] Feature: requirements
  - [x] Integrate into PRD.md (Why/What/How)
  - [x] Scaffold implementation / Core logic in `src/main.py`
  - [x] Per-file LLM extraction → `F.requirements.md`
  - [x] Skip `F.md` when `F.requirements.md` already exists (unless `--force`)
  - [x] Consolidation pass: merge all `*.requirements.md`, deduplicate, topological sort
  - [x] Final `requirements.md` output with REQ-XXX IDs, dependencies, acceptance criteria, sources
  - [ ] Add robust error handling & tests
  - [ ] Verify functionality & update CLI documentation
