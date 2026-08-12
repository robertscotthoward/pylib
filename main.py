import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer
from src.main import convert, clean

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():
    """pylib CLI tools."""


app.command("convert")(convert)
app.command("clean")(clean)

if __name__ == "__main__":
    app()
