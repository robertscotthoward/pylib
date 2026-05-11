import sys
from pathlib import Path

# Ensure src/ is on the path so cli modules are importable when running as a script
sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer
from cli.convert import convert

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():
    """pylib CLI tools."""


app.command("convert")(convert)

if __name__ == "__main__":
    app()
