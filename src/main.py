from pathlib import Path
import fnmatch
from typing import Optional
import typer

app = typer.Typer(no_args_is_help=True)

# ── prompts used by the requirements command ─────────────────────────────────

_EXTRACT_PROMPT = """\
You are a requirements analyst. Read the document below and extract every
requirement that is stated or clearly implied.

Output ONLY a markdown document with this structure:
1. A level-1 heading that is the filename (provided below).
2. A single paragraph summarising the document's purpose and scope.
3. One level-2 section per requirement, each formatted EXACTLY as:

## [REQ-NNN] <Short Title>

* **Description:** WHO wants WHAT and WHY. Use MUST / SHOULD / MAY / MUST NOT.
* **Priority:** P0 (Must Have) | P1 (Should Have) | P2 (Nice to Have)
* **Acceptance Criteria:**
  * <criterion 1>
  * <criterion 2>
* **Source:** <section heading or timestamp where this was mentioned, or "Inferred">

Rules:
- One discrete requirement per section. Break compound requirements into
  separate sections with sequential NNN values.
- NNN starts at 001 for this document.
- Do NOT include implementation details in the description; capture them as
  a separate requirement that depends on the general one.
- Do NOT emit anything outside the markdown document.

Filename: {filename}

---
{content}
{optionalprompt}"""

_CONSOLIDATE_PROMPT = """\
You are a senior systems architect. You have been given a set of per-document
requirements files extracted from meeting transcripts and design documents.

Your task:
1. Merge all requirements into one master list.
2. Eliminate exact duplicates and near-duplicates (keep the most complete version).
3. Assign globally unique IDs: REQ-001, REQ-002, … (re-number from scratch).
4. Declare dependencies: if requirement B cannot be built without A, add
   `[depends on: REQ-XXX]` at the end of the Description line.
5. Order requirements topologically so every dependency appears BEFORE the
   requirement that depends on it.
6. Detect circular dependencies; if found, break the cycle at the least
   critical link and add a note.
7. Ensure every requirement is atomic (one unit of work).
8. Use MUST / SHOULD / MAY / MUST NOT as appropriate.

Output ONLY a markdown document with:
- A level-1 heading: `# Requirements`
- A short introductory paragraph (≤4 sentences) describing the product.
- One level-2 section per requirement using EXACTLY this template:

## [REQ-NNN] <Short Title>

* **Description:** WHO wants WHAT and WHY. [depends on: REQ-XXX, REQ-YYY]
* **Priority:** P0 (Must Have) | P1 (Should Have) | P2 (Nice to Have)
* **Acceptance Criteria:**
  * <criterion 1>
  * <criterion 2>
* **Source:** <filename and/or timestamp>

Do NOT emit anything outside the markdown document.

--- BEGIN REQUIREMENTS FILES ---
{combined}
--- END REQUIREMENTS FILES ---
{optionalprompt}"""


@app.callback()
def main():
    """pylib CLI tools."""


CONVERTIBLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".rtf", ".rdf", ".epub"}


def _matches_patterns(name: str, patterns: list[str]) -> bool:
    """True if name matches any pattern; bare extensions (e.g. '.pdf') act as suffix filters."""
    lower = name.lower()
    for pat in patterns:
        if pat.startswith(".") and "*" not in pat and "?" not in pat:
            if lower.endswith(pat.lower()):
                return True
        else:
            if fnmatch.fnmatch(lower, pat.lower()):
                return True
    return False


@app.command()
def convert(
    folder: Path = typer.Argument(
        ...,
        help="Folder to recursively scan and convert files to markdown.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    filter: Optional[str] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Pipe-delimited filename patterns to process (e.g. '.pdf|report*.docx'). "
             "When omitted, all supported file types are processed.",
    ),
):
    """Convert all supported files in FOLDER to markdown, skipping files that already have a .md sibling."""
    from lib.ai.fileconvert import get_markdown, convert_doc_to_docx

    patterns: list[str] = [p.strip() for p in filter.split("|") if p.strip()] if filter else []

    converted = 0
    skipped = 0
    errors = 0

    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue
        if patterns and not _matches_patterns(file_path.name, patterns):
            continue
        if file_path.suffix.lower() not in CONVERTIBLE_EXTENSIONS:
            continue

        md_path = file_path.with_suffix(".md")
        if md_path.exists():
            skipped += 1
            continue

        typer.echo(f"Converting: {file_path}")
        try:
            convert_path = file_path

            if file_path.suffix.lower() == ".doc":
                docx_path = file_path.with_suffix(".docx")
                convert_doc_to_docx(str(file_path))
                if not docx_path.exists():
                    typer.echo(f"  Error: .doc to .docx conversion failed for {file_path.name}", err=True)
                    errors += 1
                    continue
                convert_path = docx_path

            markdown = get_markdown(str(convert_path))
            if markdown:
                md_path.write_text(markdown, encoding="utf-8")
                converted += 1
            else:
                typer.echo(f"  Warning: no content extracted from {file_path.name}", err=True)
                errors += 1
        except Exception as e:
            typer.echo(f"  Error converting {file_path.name}: {e}", err=True)
            errors += 1

    typer.echo(f"\nDone: {converted} converted, {skipped} skipped (already have .md), {errors} errors")


def _build_modelstack(model_class: str, model: str, host: str, region: str):
    from lib.ai.modelstack import ModelStack
    if model_class == "ollama":
        cfg = {
            "class": "ollama",
            "host": host or "http://localhost:11434",
            "model": model or "llama3.1:8b",
        }
    elif model_class == "bedrock":
        cfg = {
            "class": "bedrock",
            "model": model or "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "region": region or "us-east-1",
            "context-window": 200000,
        }
    else:
        typer.echo(f"Unknown model class '{model_class}'. Use 'ollama' or 'bedrock'.", err=True)
        raise typer.Exit(1)
    return ModelStack.from_config(cfg)


@app.command()
def requirements(
    folder: Path = typer.Argument(
        ...,
        help="Folder containing *.md source files to process.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    model_class: str = typer.Option(
        "bedrock",
        "--model-class",
        "-c",
        help="LLM backend: 'ollama' or 'bedrock'.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name/ID. Defaults to a sensible value per model-class.",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Ollama host URL (ollama only).",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        "-r",
        help="AWS region (bedrock only).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-F",
        help="Re-extract even if *.requirements.md already exists.",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Extra instruction appended to every LLM prompt.",
    ),
):
    """Extract requirements from *.md files in FOLDER and produce a consolidated requirements.md."""

    # Collect source markdown files, skipping *.requirements.md and requirements.md itself
    source_files = sorted(
        f for f in folder.glob("*.md")
        if not f.name.endswith(".requirements.md") and f.name != "requirements.md"
    )

    if not source_files:
        typer.echo("No *.md source files found in the folder.")
        raise typer.Exit(0)

    ms = _build_modelstack(model_class, model, host, region)

    # ── Phase 1: per-file extraction ─────────────────────────────────────────
    req_files: list[Path] = []
    for src in source_files:
        req_path = src.with_suffix("").with_suffix("") if src.suffix == ".md" else src
        req_path = folder / (src.stem + ".requirements.md")

        if req_path.exists() and not force:
            typer.echo(f"Skipping (already extracted): {src.name}")
            req_files.append(req_path)
            continue

        typer.echo(f"Extracting: {src.name} → {req_path.name}")
        content = src.read_text(encoding="utf-8")
        extract_prompt = _EXTRACT_PROMPT.format(
            filename=src.name,
            content=content,
            optionalprompt=prompt or "",
        )
        try:
            result = ms.query(extract_prompt, max_tokens=8192)
            req_path.write_text(result, encoding="utf-8")
            req_files.append(req_path)
            typer.echo(f"  Written: {req_path.name}")
        except Exception as e:
            typer.echo(f"  Error extracting {src.name}: {e}", err=True)

    if not req_files:
        typer.echo("No requirements files produced. Aborting consolidation.", err=True)
        raise typer.Exit(1)

    # ── Phase 2: consolidation ────────────────────────────────────────────────
    typer.echo("\nConsolidating all requirements files…")
    combined_parts = []
    for rf in sorted(req_files):
        combined_parts.append(f"### {rf.name}\n\n{rf.read_text(encoding='utf-8')}")
    combined = "\n\n---\n\n".join(combined_parts)

    consolidation_prompt = _CONSOLIDATE_PROMPT.format(
        combined=combined,
        optionalprompt=prompt or "",
    )
    try:
        final = ms.query(consolidation_prompt, max_tokens=16384)
    except Exception as e:
        typer.echo(f"Error during consolidation: {e}", err=True)
        raise typer.Exit(1)

    out_path = folder / "requirements.md"
    out_path.write_text(final, encoding="utf-8")
    typer.echo(f"\nDone. Consolidated requirements written to: {out_path}")


if __name__ == "__main__":
    app()
