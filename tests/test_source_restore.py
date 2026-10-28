import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location('restore_sources', Path(__file__).resolve().parents[1] / 'scripts/restore_sources.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def setup_restore(tmp_path, monkeypatch, content, raw_file='data/raw/doc.html'):
    expected = b'frozen original'
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/corpus_manifest.json').write_text(json.dumps({'sources': [{
        'doc_id': 'doc', 'url': 'https://www.njupt.edu.cn/doc.html',
        'raw_file': raw_file, 'raw_sha256': hashlib.sha256(expected).hexdigest(),
    }]}))
    monkeypatch.setattr(restore, 'ROOT', tmp_path)
    monkeypatch.setattr(restore.time, 'sleep', lambda _: None)
    client_type = httpx.Client
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b'' if r.url.path == '/robots.txt' else content))
    monkeypatch.setattr(restore.httpx, 'Client', lambda **kwargs: client_type(transport=transport, **kwargs))


def test_restoring_original_bytes_preserves_cleaned_data(tmp_path, monkeypatch):
    setup_restore(tmp_path, monkeypatch, b'frozen original')
    clean = tmp_path / 'data/processed/documents.jsonl'
    clean.parent.mkdir()
    clean.write_text('frozen cleaned corpus')
    restore.main()
    assert (tmp_path / 'data/raw/doc.html').read_bytes() == b'frozen original'
    assert clean.read_text() == 'frozen cleaned corpus'


def test_changed_remote_snapshot_does_not_overwrite_existing_file(tmp_path, monkeypatch):
    setup_restore(tmp_path, monkeypatch, b'updated policy')
    target = tmp_path / 'data/raw/doc.html'
    target.parent.mkdir()
    target.write_bytes(b'local file to preserve')
    with pytest.raises(SystemExit):
        restore.main()
    assert target.read_bytes() == b'local file to preserve'
    report = json.loads((tmp_path / 'results/source_restore.json').read_text())
    assert report['checks'][0]['status'] == 'snapshot_changed'


def test_manifest_cannot_write_outside_raw_directory(tmp_path, monkeypatch):
    setup_restore(tmp_path, monkeypatch, b'frozen original', 'outside.html')
    with pytest.raises(SystemExit):
        restore.main()
    assert not (tmp_path / 'outside.html').exists()
