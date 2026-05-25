For the pylib/main.py file, make it a CLI using Typer that can act as a tool in addition to being a library for reuse.
Create files as needed in the src/cli folder but avoid changing files in the src/lib folder - else propose the changes and ask permission.

For the first use case, make this call:
`python main.py convert FOLDER`
that will recursively scan the FOLDER for all files that can be converted to markdown but does not have a corresponding sibling .md file and then read the file and convert it creating the .md file.
