"""Compile repository Python files excluding virtualenv and hidden dirs."""
import compileall
import os

SKIP = {".venv", ".git", "__pycache__"}

printed = False
for entry in os.listdir('.'):
    if entry in SKIP or entry.startswith('.'):  # skip hidden and configured
        continue
    path = os.path.join('.', entry)
    if os.path.isdir(path):
        compileall.compile_dir(path, quiet=1)
    elif entry.endswith('.py'):
        compileall.compile_file(path, quiet=1)
print("compile_repo_done")
