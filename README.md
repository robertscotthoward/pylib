# Lib

This project was created:
```
uv init --python 3.12
uv add chromadb striprtf pypdf[crypto] python-docx pyyaml rdflib ebooklib requests ruamel-yaml langchain-text-splitters boto3 FlagEmbedding
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```


# Sanity
```
uv run python -c "from lib import tools; print('Success!')"
```


# Run Unit Tests from CLI
```
uv run pytest
```


# Run Unit Tests from IDE
* Select launch setting "Test"


# Use in another project
```
Create a project folder.
uv init --python 3.13
uv add --editable ../../lib

# Test:
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

# ====================================================================================================
# UV Template

Use this process to create new Python projects in a convenient manner, such that:
* A virtual environment is created with the powerful [uv](https://docs.astral.sh/uv/guides/tools/)
* Fast: very fast virtual environment creations, dependency installation, re-creations


# Global Environment

## uv
If you want all your projects to be on D drive, then create environment variable: UV_CACHE_DIR=D:\uv
This makes all your projects on D drive use hard links for libraries, which is very fast.


Install `uv`, e.g. scoop, pip, pipx, etc. Verify with `uv --help`


# New Project
You want to create a brand new uv-driven python project "Project1" that uses git.
Here are some good reproducible steps to make a good project.

```
:: IF LOCAL THEN
md Project1
cd Project1
git init
:: ELSE
:: Create the "Project1" repo in Github
:: git clone the repo
cd Project1
:: END


uv init --python 3.10
:: Or just `uv init` to use the latest version of Python. 3.10 tends to be more AI stable.

:: To explicitly create the venv, but uv add PACKAGE does this automatically if venv does not exist.
uv venv .venv



.venv\Scripts\activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

## Specific Scenario

Here's an example from the command line for creating a new local project.
```
md Project1
cd Project1
copy D:\github\zinclusive\tech\python\uv-template
git init
uv init --python 3.10
uv add requests pyyaml pandas
.venv\Scripts\activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
code .
```

# Copy all the "bat" files into project folder

These bat files are convenient. Here's what they do:
* `activate` - from terminal, quickly activate the virtual environment
* `deactivate` - from terminal, quickly deactivate the virtual environment
* `dev-cmd` - activates the environment, then starts a terminal
* `dev-cursor` - activates the environment, starts a terminal, then starts Cursor
* `dev` - activates the environment, starts a terminal, then starts VS Code
* `reset-venv` - deletes the entire .venv folder, runs sync, and installs pip. Do this if you change python versions.

# Change Python Version

If you want to change the underlying python runtime:

* Change the version in `pyproject.toml` and `.python-version`.
* Run:
```
rm -rf .venv
uv sync
call .venv\Scripts\activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```
