"""Meaningful dataset gates for the optional tuning experiment artifacts."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ['sft_aligned', 'sft_aligned_long', 'sft_aligned_rich_long']


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]


def test_tuning_real_facts_and_distractors_are_disjoint_from_validation():
    original = rows('data/sft/provenance.jsonl')
    valid = {g for p in original if p['split'] != 'train' for g in [p['fact_group_id'], *p['extra_fact_groups']]}
    for folder in FOLDERS:
        provenance = rows(f'data/{folder}/provenance.jsonl')
        real = {g for p in provenance for g in [p['fact_group_id'], *p.get('evidence_fact_groups', [])] if not g.startswith('aligned-')}
        assert not real & valid


def test_tuning_official_answer_spans_are_present_in_sources():
    units = {u['unit_id']: u for u in rows('data/tuning/source_candidates.jsonl')}
    norm = lambda text: re.sub(r'\s+', '', text)
    for split, partition in [('train', 'train'), ('train_extra', 'train'), ('validation', 'valid'), ('fresh_test', 'test')]:
        with (ROOT / f'data/tuning/{split}_annotations.tsv').open() as stream:
            for item in csv.DictReader(stream, delimiter='\t'):
                unit = units[item['unit_id']]
                assert unit['split'] == partition
                assert all(norm(fact) in norm(unit['text']) for fact in item['required_facts'].split('|'))


def test_tuning_train_has_no_exact_evaluation_question():
    protected = {c['question'] for c in rows('data/eval/campus_100.jsonl') + rows('data/eval/acceptance.jsonl') + rows('data/tuning/validation_60.jsonl') + rows('data/tuning/fresh_test_60.jsonl')}
    for folder in FOLDERS:
        for item in rows(f'data/{folder}/train.jsonl'):
            question = item['messages'][-2]['content'].split('\n\n问题：')[-1].split('\n请只依据')[0]
            assert question not in protected


def test_tuning_lengths_hashes_and_group_variant_limits():
    from collections import Counter
    for folder in FOLDERS:
        provenance = rows(f'data/{folder}/provenance.jsonl')
        audit = json.loads((ROOT / f'data/{folder}/audit.json').read_text())
        assert len(provenance) == 1200
        assert max(p['tokens'] for p in provenance) <= 1536
        assert max(Counter(p['fact_group_id'] for p in provenance).values()) <= 3
        assert hashlib.sha256((ROOT / f'data/{folder}/train.jsonl').read_bytes()).hexdigest() == audit['train_sha256']


def test_tuning_refusals_do_not_claim_unrelated_citations():
    for folder in FOLDERS:
        provenance = rows(f'data/{folder}/provenance.jsonl')
        samples = rows(f'data/{folder}/train.jsonl')
        for p in provenance:
            if p['category'] == 'refusal':
                assert not re.search(r'\[\[C\d+\]\]', samples[p['row']]['messages'][-1]['content'])
