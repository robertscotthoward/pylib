from pathlib import Path
import fnmatch
import os
import shutil
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


CONVERTIBLE_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".rtf", ".rdf", ".epub",
    ".xlsx", ".xls", ".pptx", ".ppt",
}

# Still images get a vision-model description; see fileconvert.image_to_markdown.
# Video is deliberately absent -- describing it needs frame sampling, not one still.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}

# Audio is transcribed by a speech model; see fileconvert.audio_to_markdown.
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma", ".aiff"}

# Both are network calls per file, so they are queued to the worker pool rather than
# converted inline.
REMOTE_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS
CONVERTIBLE_EXTENSIONS |= REMOTE_EXTENSIONS

# Text formats the clean command will touch when walking a folder. Binary formats are
# excluded because reading them as text and writing the result back would destroy them.
# An explicitly named file is always cleaned, whatever its extension.
CLEANABLE_EXTENSIONS = {
    ".txt", ".text", ".md", ".markdown", ".rst", ".log",
    ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".ini", ".cfg", ".conf",
}


_PARTIAL_HASH_BYTES = 65536


def _build_dedupe_index(folder: Path):
    """Index convertible files by (size, extension, hash of first 64KB).

    Two thirds of a large corpus share a size with nothing, and those cannot have a
    duplicate, so only files whose size collides are ever opened. The partial hash is a
    filter, not proof: a match is confirmed byte-for-byte before anything is reused.

    Returns (key by path, paths by key, number of files hashed). Every convertible file
    is indexed regardless of --filter, so a donor can come from anywhere in the tree.
    """
    import hashlib
    from collections import defaultdict

    by_size = defaultdict(list)
    for path in folder.rglob("*"):
        if not path.is_file() or path.name.startswith("._"):
            continue
        extension = path.suffix.lower()
        if extension not in CONVERTIBLE_EXTENSIONS:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size:
            by_size[(size, extension)].append(path)

    key_by_path: dict = {}
    paths_by_key: dict = defaultdict(list)
    hashed = 0
    for (size, extension), paths in by_size.items():
        if len(paths) < 2:
            continue
        for path in paths:
            try:
                with open(path, "rb") as handle:
                    head = handle.read(_PARTIAL_HASH_BYTES)
            except OSError:
                continue
            key = (size, extension, hashlib.sha256(head).hexdigest())
            key_by_path[path] = key
            paths_by_key[key].append(path)
            hashed += 1
    return key_by_path, paths_by_key, hashed


def _find_duplicate_md(file_path: Path, key_by_path: dict, paths_by_key: dict):
    """Return an existing .md belonging to a byte-identical file, or None.

    The candidate is compared in full with filecmp before its .md is offered up.
    Attaching one document's text to a different document would be a worse outcome
    than converting the same bytes twice.
    """
    import filecmp

    key = key_by_path.get(file_path)
    if key is None:
        return None
    for other in paths_by_key.get(key, ()):
        if other == file_path:
            continue
        other_md = other.with_name(other.name + ".md")
        try:
            if not other_md.is_file() or other_md.stat().st_size == 0:
                continue
            if filecmp.cmp(str(file_path), str(other), shallow=False):
                return other_md
        except OSError:
            continue
    return None


def _write_error_marker(md_path: Path, source_path: Path):
    """Write an empty .md file to mark a failed conversion, so a future non-forced
    run doesn't keep retrying it. Stamped with the source's mtime so it isn't
    treated as stale (and re-attempted) until the source itself changes.
    """
    md_path.write_bytes(b"")
    st = source_path.stat()
    os.utime(md_path, (st.st_atime, st.st_mtime))


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
def clean(
    path: Path = typer.Argument(
        ...,
        help="File to clean, or folder to clean recursively.",
        exists=True,
        resolve_path=True,
    ),
    filter: Optional[str] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Pipe-delimited filename patterns to process (e.g. '.md|notes*.txt'). "
             "Folder mode only; when omitted, all supported text types are processed.",
    ),
    keep_nonsense: bool = typer.Option(
        False,
        "--keep-nonsense",
        help="Repair encoding with ftfy but keep every line, skipping nostril's gibberish check.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Report what would change without writing files or creating backups.",
    ),
):
    """Clean up text files with ftfy and nostril.

    Each file is read, cleaned, and rewritten only when the cleaning changed something.
    The original is renamed to FILE.bak first, so nothing is lost. Unchanged files are
    left alone, which makes the command safe to re-run.
    """
    from lib.ai.fileconvert import clean_text, read_text_best_effort

    patterns: list[str] = [p.strip() for p in filter.split("|") if p.strip()] if filter else []

    if path.is_file():
        targets = [path]
    else:
        targets = [
            p for p in sorted(path.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in CLEANABLE_EXTENSIONS
            and not p.name.endswith(".bak")
            and not p.name.startswith("._")
            and (not patterns or _matches_patterns(p.name, patterns))
        ]

    cleaned = 0
    unchanged = 0
    errors = 0

    for file_path in targets:
        try:
            original = read_text_best_effort(file_path)
            result = clean_text(original, drop_nonsense=not keep_nonsense)
        except Exception as e:
            typer.echo(f"  Error cleaning {file_path.name}: {e}", err=True)
            errors += 1
            continue

        if result == original:
            unchanged += 1
            continue

        removed = original.count("\n") - result.count("\n")
        detail = f" ({removed} lines dropped)" if removed > 0 else ""
        if dry_run:
            typer.echo(f"Would clean: {file_path}{detail}")
            cleaned += 1
            continue

        backup_path = file_path.with_name(file_path.name + ".bak")
        os.replace(file_path, backup_path)
        # newline='' keeps the \n that ftfy normalised to, instead of letting Windows
        # rewrite them as \r\n, which would make every run report a change.
        file_path.write_text(result, encoding="utf-8", newline="")
        typer.echo(f"Cleaned: {file_path}{detail} (original saved to {backup_path.name})")
        cleaned += 1

    verb = "would be cleaned" if dry_run else "cleaned"
    typer.echo(f"\nDone: {cleaned} {verb}, {unchanged} already clean, {errors} errors")


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
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-convert every matching file, even when its .md is up to date.",
    ),
    retry_errors: bool = typer.Option(
        False,
        "--retry-errors",
        help="Also retry files whose previous conversion failed, i.e. left an empty .md. "
             "Files that converted successfully are still skipped.",
    ),
    redo_ocr: bool = typer.Option(
        False,
        "--redo-ocr",
        help="Also re-convert scanned PDFs that require OCR, even when their .md looks up "
             "to date. Use after changing the OCR repair model. Text-layer PDFs and every "
             "other file type are left alone.",
    ),
    ocr_workers: int = typer.Option(
        8,
        "--ocr-workers",
        min=1,
        help="How many OCR repair calls to run concurrently. The repair is a network "
             "round trip, so raising this shortens a scan-heavy run; 1 makes it sequential.",
    ),
    ocr_processes: Optional[int] = typer.Option(
        None,
        "--ocr-processes",
        min=1,
        help="How many CPU processes Tesseract may use to OCR pages of one PDF. "
             "Defaults to min(8, cpu count); 1 disables the pool.",
    ),
    older_than: Optional[str] = typer.Option(
        None,
        "--older-than",
        help="With --redo-ocr, only re-convert when the existing .md predates this date "
             "(YYYY-MM-DD or YYYY-MM-DDTHH:MM). Use it to redo scans from before a model "
             "change without touching newer, already-good output.",
    ),
    dedupe: bool = typer.Option(
        False,
        "--dedupe",
        help="Reuse an existing .md when a byte-identical copy of the file has already "
             "been converted, instead of converting the same bytes twice. Costs one "
             "indexing pass up front.",
    ),
    fix_names_first: bool = typer.Option(
        False,
        "--fix-names",
        help="Before converting, migrate old-style 'XXX.md' output to the 'XXX.EXT.md' "
             "convention. Runs first so the migrated files are then seen as up to date.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Report what would happen without renaming, deleting, or writing anything.",
    ),
):
    """Convert all supported files in FOLDER to markdown.

    XXX.ext converts to XXX.ext.md, not XXX.md, so a folder holding both report.doc and
    report.docx gets two distinct outputs instead of one clobbering the other. Use
    `fix-names` to migrate a corpus that already has the old XXX.md outputs.

    A file is converted when its .md sibling is missing or older than the file itself,
    so edited sources are re-converted and up-to-date ones are skipped.

    A failed conversion leaves an empty .md stamped with the source's mtime, which reads
    as up to date and is skipped on later runs. --retry-errors treats those empty files
    as missing and retries only them, without re-doing work that already succeeded.

    Scanned PDFs are handled in two stages. Tesseract runs inline because it takes about a
    second, while the LLM repair that follows is a much slower network call, so those are
    queued to a worker pool and collected at the end. Fast documents keep converting while
    the repairs are in flight.
    """
    from concurrent.futures import ThreadPoolExecutor
    from lib.ai.fileconvert import (
        get_markdown, convert_doc_to_docx, needs_conversion, pdf_needs_ocr,
        reformat_ocr_markdown, image_to_markdown, audio_to_markdown,
    )

    patterns: list[str] = [p.strip() for p in filter.split("|") if p.strip()] if filter else []

    # Renames a dry run only pretended to make, so the conversion preview below can
    # still judge staleness against where each .md would have ended up.
    planned_renames: dict = {}
    if fix_names_first:
        # Runs before the walk so migrated files are already correctly named, and the
        # conversion pass then sees them as up to date instead of redoing the work.
        planned_renames = _fix_names(folder, dry_run)
        typer.echo("")

    if ocr_processes:
        # fileconvert reads this per call, so setting it here reaches the OCR pool.
        os.environ["PYLIB_OCR_WORKERS"] = str(ocr_processes)

    cutoff = None
    if older_than:
        from datetime import datetime
        try:
            cutoff = datetime.fromisoformat(older_than).timestamp()
        except ValueError:
            typer.echo(f"Invalid --older-than '{older_than}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM.", err=True)
            raise typer.Exit(1)

    key_by_path: dict = {}
    paths_by_key: dict = {}
    if dedupe:
        typer.echo("Indexing duplicates...")
        key_by_path, paths_by_key, hashed = _build_dedupe_index(folder)
        typer.echo(
            f"Indexed {hashed} files sharing a size, in {len(paths_by_key)} groups.\n"
        )

    converted = 0
    updated = 0
    retried = 0
    skipped = 0
    empty_sources = 0
    errors = 0
    repaired = 0
    deduped = 0

    # Scanned PDFs whose LLM repair is still in flight: (future, raw markdown, paths, flags).
    pending: list[tuple] = []

    def record_success(is_error_marker: bool, rebuild: bool):
        nonlocal converted, updated, retried
        if is_error_marker:
            retried += 1
        elif rebuild:
            updated += 1
        else:
            converted += 1

    def flush_pending(block: bool):
        """Write finished OCR repairs to disk.

        Called after every file with block=False so each document lands as soon as its
        repair returns. That keeps progress durable: interrupt the run and everything
        already written is complete, so re-running skips it instead of redoing it.
        """
        nonlocal repaired, errors
        ready = list(pending) if block else [job for job in pending if job[0].done()]
        for job in ready:
            future, raw, md_path, source_path, was_error_marker, was_rebuild = job
            pending.remove(job)
            try:
                text = future.result()
            except Exception as e:
                # Keep the raw OCR rather than losing the page to a failed network call.
                typer.echo(f"  Warning: OCR repair failed for {source_path.name}, keeping raw OCR: {e}", err=True)
                text = raw
            text = text or raw
            if not (text and text.strip()):
                # Nothing came back and there is no raw OCR to fall back on.
                typer.echo(f"  Warning: no content extracted from {source_path.name}", err=True)
                _write_error_marker(md_path, source_path)
                errors += 1
                continue
            try:
                md_path.write_text(text, encoding="utf-8")
                record_success(was_error_marker, was_rebuild)
                repaired += 1
                typer.echo(f"  Done: {source_path.name}")
            except Exception as e:
                typer.echo(f"  Error writing {md_path.name}: {e}", err=True)
                errors += 1

    pool = ThreadPoolExecutor(max_workers=ocr_workers)
    try:
        for file_path in sorted(folder.rglob("*")):
            # Land any repair that finished while the previous files were converting.
            flush_pending(block=False)
            if not file_path.is_file():
                continue
            if patterns and not _matches_patterns(file_path.name, patterns):
                continue
            if file_path.suffix.lower() not in CONVERTIBLE_EXTENSIONS:
                continue
            # macOS '._name.doc' sidecars carry a document's name but only resource-fork
            # metadata, so every converter fails on them.
            if file_path.name.startswith("._"):
                continue
            # A zero-byte source holds no document for any parser to read. Skipping it is
            # not a failure, so it gets no error marker and does not inflate the error count.
            if file_path.stat().st_size == 0:
                empty_sources += 1
                continue

            md_path = file_path.with_name(file_path.name + ".md")
            # In a dry run the rename has not happened, so judge staleness against the file
            # the rename would have moved. Outside a dry run this is always md_path itself.
            check_md = planned_renames.get(md_path, md_path)
            # An empty .md is the marker a failed conversion leaves behind, never a real result.
            is_error_marker = check_md.is_file() and check_md.stat().st_size == 0
            is_pdf = file_path.suffix.lower() == ".pdf"
            scanned = None  # resolved lazily; opening every PDF twice is wasted work

            if force:
                stale = True
            elif retry_errors and is_error_marker:
                stale = True
            else:
                stale = needs_conversion(str(file_path), str(check_md))
                if not stale and redo_ocr and is_pdf:
                    # Only pay for opening the PDF once the date filter has passed.
                    fresh_enough = cutoff is not None and check_md.exists() and check_md.stat().st_mtime >= cutoff
                    if not fresh_enough:
                        scanned = pdf_needs_ocr(str(file_path))
                        stale = scanned
            if not stale:
                skipped += 1
                continue
            rebuild = check_md.exists()

            if is_error_marker:
                action, intent = "Retrying", "Would retry"
            elif rebuild:
                action, intent = "Re-converting", "Would re-convert"
            else:
                action, intent = "Converting", "Would convert"

            # Reuse the output of an identical file before doing any conversion work.
            donor_md = _find_duplicate_md(file_path, key_by_path, paths_by_key) if dedupe else None
            if donor_md is not None:
                if dry_run:
                    typer.echo(f"Would copy from duplicate: {file_path.name} <- {donor_md.name}")
                else:
                    shutil.copyfile(donor_md, md_path)
                    typer.echo(f"Copied from duplicate: {file_path.name} <- {donor_md.name}")
                deduped += 1
                continue

            if dry_run:
                typer.echo(f"{intent}: {file_path}")
                record_success(is_error_marker, rebuild)
                continue

            typer.echo(f"{action}: {file_path}")
            try:
                convert_path = file_path

                if file_path.suffix.lower() == ".doc":
                    docx_path = file_path.with_suffix(".docx")
                    convert_doc_to_docx(str(file_path))
                    if not docx_path.exists():
                        typer.echo(f"  Error: .doc to .docx conversion failed for {file_path.name}", err=True)
                        _write_error_marker(md_path, file_path)
                        errors += 1
                        continue
                    convert_path = docx_path

                remote_worker = None
                if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    remote_worker = image_to_markdown
                elif file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    remote_worker = audio_to_markdown
                if remote_worker is not None:
                    # One network call per file, so queue it and keep walking rather than
                    # blocking the main thread on every photo or recording in the tree.
                    future = pool.submit(remote_worker, str(convert_path))
                    pending.append((future, "", md_path, file_path, is_error_marker, rebuild))
                    continue

                if scanned is None and is_pdf:
                    scanned = pdf_needs_ocr(str(convert_path))

                if scanned:
                    # OCR inline (about a second), then hand the slow repair to the pool and
                    # move on to the next file instead of blocking on the network.
                    raw = get_markdown(str(convert_path), repair_ocr=False)
                    if raw and raw.strip():
                        future = pool.submit(reformat_ocr_markdown, raw)
                        pending.append((future, raw, md_path, file_path, is_error_marker, rebuild))
                        continue
                    markdown = raw
                else:
                    markdown = get_markdown(str(convert_path))

                if markdown:
                    md_path.write_text(markdown, encoding="utf-8")
                    record_success(is_error_marker, rebuild)
                else:
                    typer.echo(f"  Warning: no content extracted from {file_path.name}", err=True)
                    _write_error_marker(md_path, file_path)
                    errors += 1
            except Exception as e:
                typer.echo(f"  Error converting {file_path.name}: {e}", err=True)
                _write_error_marker(md_path, file_path)
                errors += 1

        # Collect whatever OCR repairs are still running after the walk finished.
        if pending:
            typer.echo(f"\nCollecting {len(pending)} OCR repair(s) still in flight...")
        flush_pending(block=True)
    finally:
        pool.shutdown(wait=True)

    summary = [
        f"{converted} " + ("would be converted" if dry_run else "converted"),
        f"{updated} re-converted (source newer than .md)",
        f"{retried} recovered (previous error)",
        f"{repaired} via OCR",
        f"{deduped} copied from a duplicate",
        f"{skipped} skipped (.md up to date)",
        f"{empty_sources} skipped (empty source file)",
        f"{errors} errors",
    ]
    typer.echo("\nDone: " + ", ".join(summary))
    if errors and not retry_errors:
        typer.echo("Re-run with --retry-errors to retry only the files that failed.")


@app.command("fix-names")
def fix_names(
    folder: Path = typer.Argument(
        ...,
        help="Folder to recursively scan for old-style same-stem .md files.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Report what would be renamed/deleted without changing anything.",
    ),
):
    """Migrate old-style 'XXX.md' conversion output to the collision-proof 'XXX.EXT.md'
    convention that `convert` now writes.

    For every XXX.md found, looks at every other file in the same folder whose stem
    (case-insensitive) is also XXX:

    - Exactly one such file XXX.EXT: unambiguous, XXX.md is renamed to XXX.EXT.md.
    - Two or more such files (e.g. report.doc and report.docx both present): ambiguous,
      since XXX.md could have come from any of them. It is deleted rather than guessed
      at; the source files are untouched, so re-running `convert` regenerates correct,
      distinctly-named output for each one.
    - None: XXX.md isn't a conversion output (e.g. a hand-written note) and is left alone.
    """
    _fix_names(folder, dry_run)


def _fix_names(folder: Path, dry_run: bool) -> dict:
    """Rename old-style XXX.md outputs to XXX.EXT.md. Shared by the `fix-names` command
    and `convert --fix-names`, so both behave identically.

    Returns {planned new path: existing old path} for a dry run, empty otherwise. It lets
    `convert --fix-names --dry-run` report what it would do *after* the rename, instead of
    claiming it would re-convert every file whose .md has not moved yet.
    """
    planned: dict = {}
    renamed = 0
    deleted = 0
    left_alone = 0

    for md_path in sorted(folder.rglob("*.md")):
        stem_lower = md_path.stem.lower()
        siblings = [
            p for p in md_path.parent.iterdir()
            if p.is_file() and p != md_path and p.stem.lower() == stem_lower
        ]

        if not siblings:
            left_alone += 1
            continue

        if len(siblings) == 1:
            new_path = siblings[0].with_name(siblings[0].name + ".md")
            if new_path.exists():
                typer.echo(f"  Skipping (target already exists): {md_path} -> {new_path.name}", err=True)
                left_alone += 1
                continue
            if dry_run:
                typer.echo(f"Would rename: {md_path} -> {new_path.name}")
                planned[new_path] = md_path
            else:
                md_path.rename(new_path)
                typer.echo(f"Renamed: {md_path} -> {new_path.name}")
            renamed += 1
            continue

        names = ", ".join(p.name for p in siblings)
        if dry_run:
            typer.echo(f"Would delete (ambiguous, matches {names}): {md_path}")
        else:
            md_path.unlink()
            typer.echo(f"Deleted (ambiguous, matches {names}): {md_path}")
        deleted += 1

    verb = "would be renamed" if dry_run else "renamed"
    verb2 = "would be deleted" if dry_run else "deleted"
    typer.echo(f"\nNames: {renamed} {verb}, {deleted} {verb2} (ambiguous), {left_alone} left alone")
    return planned


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
