import os
import pytest
from njupt_qa.runtime import ollama_binary


def test_ollama_resolves_from_path_without_a_fixed_install_prefix(tmp_path, monkeypatch):
    binary = tmp_path / 'ollama'
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    monkeypatch.delenv('OLLAMA_BIN', raising=False)
    monkeypatch.setenv('PATH', str(tmp_path))
    assert ollama_binary() == str(binary)


def test_explicit_ollama_override_takes_precedence(tmp_path, monkeypatch):
    binary = tmp_path / 'custom-ollama'
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    monkeypatch.setenv('OLLAMA_BIN', str(binary))
    assert ollama_binary() == str(binary)


def test_missing_ollama_has_an_actionable_error(monkeypatch):
    monkeypatch.delenv('OLLAMA_BIN', raising=False)
    monkeypatch.setenv('PATH', '')
    with pytest.raises(RuntimeError, match='OLLAMA_BIN'):
        ollama_binary()
