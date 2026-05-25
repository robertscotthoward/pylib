import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer
from src.main import convert

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():
    """pylib CLI tools."""


app.command("convert")(convert)

if __name__ == "__main__":
    app()
