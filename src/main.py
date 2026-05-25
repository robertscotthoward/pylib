from pathlib import Path
import fnmatch
from typing import Optional
import typer

app = typer.Typer(no_args_is_help=True)


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


if __name__ == "__main__":
    app()
