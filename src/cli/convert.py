from pathlib import Path
import typer

# Extensions that get_markdown() explicitly dispatches to (not text passthrough)
CONVERTIBLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".rtf", ".rdf", ".epub"}


def convert(
    folder: Path = typer.Argument(
        ...,
        help="Folder to recursively scan and convert files to markdown.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
):
    """Convert all supported files in FOLDER to markdown, skipping files that already have a .md sibling."""
    from lib.ai.fileconvert import get_markdown, convert_doc_to_docx

    converted = 0
    skipped = 0
    errors = 0

    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
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

            # .doc files must be converted to .docx first
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
