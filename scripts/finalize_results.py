"""Audit saved experiment artifacts and build a reproducibility index; no inference."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sqlite3
import sys

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    documents = rows('data/processed/documents.jsonl')
    topics = Counter(d['topic'] for d in documents)
    assert len(documents) == 56 and len(topics) == 5 and min(topics.values()) >= 8
    assert all(sha(ROOT / d['raw_file']) == d['raw_sha256'] for d in documents)
    chunks = rows('data/processed/chunks.jsonl')
    vectors = np.load(ROOT / 'data/vectors.npy', mmap_mode='r')
    assert vectors.shape == (len(chunks), 512)
    with sqlite3.connect(ROOT / 'data/knowledge.sqlite') as con:
        assert con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0] == len(chunks)
    counts = {split: len(rows(f'data/sft/{split}.jsonl')) for split in ['train', 'valid', 'test']}
    assert counts == {'train': 1200, 'valid': 150, 'test': 150}
    provenance = rows('data/sft/provenance.jsonl')
    groups = {split: set() for split in counts}
    for item in provenance:
        groups[item['split']].update([item['fact_group_id'], *item.get('extra_fact_groups', [])])
    assert not (groups['train'] & groups['valid'] or groups['train'] & groups['test'] or groups['valid'] & groups['test'])
    campus = rows('data/eval/campus_100.jsonl')
    assert len(campus) == 100
    assert not {r['fact_group_id'] for r in campus if r['answerable']} & (groups['train'] | groups['valid'] | groups['test'])
    audit = read('data/sft/audit.json')
    assert audit['max_tokens'] <= 1536 and audit['exact_duplicate_samples'] == 0
    assert audit['acceptance_documents_excluded_from_sft']
    csv_counts = {}
    for folder, modes, expected in [
        ('dev', ['raw', 'sft', 'rag', 'sft_rag'], 20),
        ('test', ['raw', 'sft', 'rag', 'sft_rag'], 80),
        ('test_calculator', ['rag', 'sft_rag'], 80),
        ('calculation_24', ['rag', 'rag_calculator', 'sft_rag', 'sft_rag_calculator'], 24),
        ('', ['acceptance'], 4),
    ]:
        for mode in modes:
            name = str(Path('results') / folder / f'{mode}.csv')
            with (ROOT / name).open(encoding='utf-8-sig', newline='') as stream:
                table = list(csv.reader(stream))
            assert table[0] == ['问题', '答案', '参考资料']
            assert len(table) == expected + 1 and all(len(row) == 3 for row in table)
            csv_counts[name] = expected
    with httpx.Client(timeout=5, trust_env=False) as client:
        health = client.get('http://127.0.0.1:8765/api/health').json()
        tags = client.get('http://127.0.0.1:11435/api/tags').json()['models']
    digests = {tag['name'].split(':')[0]: tag['digest'] for tag in tags}
    main_results = {}
    subgroups = {}
    for mode in ['raw', 'sft', 'rag', 'sft_rag']:
        output = rows(f'results/test/{mode}.jsonl')
        assert len(output) == 80 and all(not r.get('error') for r in output)
        tag = 'njupt-qwen3-sft' if mode in ['sft', 'sft_rag'] else 'njupt-qwen3-base'
        assert all(r['model_digest'] == digests[tag] for r in output)
        main_results[mode] = read('results/test/summary.json')[mode]['accuracy']
        partitions = {
            'stable_history_identity': [r for r in output if r['case']['answerable'] and r['case']['topic'] == '学校概况'],
            'dated_information_and_policy': [r for r in output if r['case']['answerable'] and r['case']['topic'] != '学校概况'],
            'heldout_sentence_ids': [r for r in output if r['case']['answerable']],
            'unanswerable': [r for r in output if not r['case']['answerable']],
        }
        subgroups[mode] = {name: {'cases': len(group), 'accuracy': sum(r['grade']['correct'] for r in group) / len(group)}
                           for name, group in partitions.items()}
    save('results/test/subgroups.json', {
        'annotation': 'Post-hoc descriptive grouping, without changing questions, answers or scoring. School overview facts are stable history/identity; other positive cases concern dated figures or service policies. Sentence IDs are held out but a later semantic audit found repeated school-spirit and founding-year facts on other pages; do not interpret this as all facts being unseen. Groups overlap.',
        'modes': subgroups,
    })
    training = read('results/training_summary.json')
    assert training['completed'] and training['actual_epochs'] == 3
    assert sha(ROOT / 'models/lora-adapter/adapters.safetensors') == training['adapter_sha256']
    assert read('results/acceptance_summary.json')['all_passed']
    assert read('results/offline_verification.json')['passed']
    verification = read('results/model_verification.json')
    assert verification['all_56_adapted_layers_changed'] and verification['base_source_conversion_verified']
    assert verification['fused_weight_max_abs_error'] < 0.01
    assert '18 passed' in (ROOT / 'results/pytest.log').read_text()
    for mode in ['rag_calculator', 'sft_rag_calculator']:
        assert read('results/calculation_24/summary.json')[mode]['correct'] == 24
    shutil.copy2(ROOT / 'models/lora-adapter/metrics.jsonl', ROOT / 'results/training_metrics.jsonl')
    critical = [
        'pyproject.toml', 'uv.lock', 'requirements-lock.txt', 'frontend/pnpm-lock.yaml',
        'config/model_sources.json', 'data/processed/documents.jsonl', 'data/processed/chunks.jsonl',
        'data/sft/train.jsonl', 'data/sft/valid.jsonl', 'data/sft/test.jsonl',
        'data/eval/campus_100.jsonl', 'data/eval/calculation_24.jsonl',
        'models/lora-adapter/adapters.safetensors', 'models/qwen3-base-q8_0.gguf', 'models/qwen3-sft-q8_0.gguf',
    ]
    save('results/reproducibility.json', {
        'python': sys.version, 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ['mlx', 'mlx-lm', 'torch', 'transformers', 'onnxruntime', 'fastapi', 'numpy']},
        'artifact_sha256': {name: sha(ROOT / name) for name in critical}, 'ollama_model_digests': digests,
    })
    completion = {
        'completed_at': datetime.now().astimezone().isoformat(), 'passed': True,
        'documents': len(documents), 'topic_counts': dict(topics), 'chunks': len(chunks),
        'sft_counts': counts, 'sft_fact_group_overlap': 0, 'evaluation_fact_overlap': 0,
        'csv_rows': csv_counts, 'main_test_rule_accuracy': main_results,
        'acceptance': '4/4', 'calculation_with_tools': '24/24 in both RAG modes',
        'tests': '18 passed', 'offline_process_sandbox': True, 'health': health,
        'limitation': 'SFT reduced held-out template loss but degraded main campus test performance. Calculator results measure evidence expansion plus deterministic arithmetic, not SFT arithmetic ability.',
        'ui_verification': ['responsive desktop/mobile', 'evidence and local PDF page', 'rename', 'copy', 'Markdown export', 'stop and regenerate without duplicate turns'],
    }
    save('results/completion.json', completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
