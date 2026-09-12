import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer
from src.main import convert, clean

app = typer.Typer(no_args_is_help=True)


def _print_last_commit():
    """Show this checkout's last commit and when it was made, so a stale install is obvious."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M:%S"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            typer.echo(f"pylib @ {result.stdout.strip()}")
    except (OSError, subprocess.SubprocessError):
        pass


@app.callback()
def main():
    """pylib CLI tools."""
    _print_last_commit()


app.command("convert")(convert)
app.command("clean")(clean)

if __name__ == "__main__":
    app()
