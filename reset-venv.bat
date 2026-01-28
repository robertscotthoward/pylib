rm -rf .venv
uv sync
call .venv\Scripts\activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
