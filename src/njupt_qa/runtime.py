"""Resolve the installed Ollama executable without a machine-specific path."""
import os
from pathlib import Path
import shutil


def ollama_binary():
    override = os.environ.get('OLLAMA_BIN')
    candidate = shutil.which(override or 'ollama')
    if candidate:
        return candidate
    if override:
        path = Path(override).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise RuntimeError('Ollama executable not found. Install Ollama and add it to PATH, or set OLLAMA_BIN.')
