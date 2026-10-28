"""Evidence-aligned data and learning-rate controlled comparison."""
from pathlib import Path
import json
import subprocess
import sys
from run_tuning_phase2 import stage_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, 'scripts/tune_lora.py', *args], cwd=ROOT, check=True)


def main():
    smoke = ROOT / 'models/tuning/aligned-long-smoke/training_summary.json'
    if not smoke.exists():
        run('train', '--name', 'aligned-long-smoke', '--data', 'data/sft_aligned_long', '--iters', '20', '--save-every', '20')
    assert json.loads(smoke.read_text())['peak_memory_gb'] < 10, 'Reprocess long data with checkpointing before training'
    for name, data, lr in [
        ('aligned-short-lr3e5', 'data/sft_aligned', 3e-5),
        ('aligned-short-lr1e4', 'data/sft_aligned', 1e-4),
        ('aligned-long-lr3e5', 'data/sft_aligned_long', 3e-5),
        ('aligned-long-lr1e4', 'data/sft_aligned_long', 1e-4),
    ]:
        if not (ROOT / 'models/tuning' / name / 'training_summary.json').exists():
            run('train', '--name', name, '--data', data, '--lr', str(lr), '--iters', '1200')
        candidates = []
        for step in [200, 400, 800, 1200]:
            adapter = stage_checkpoint(name, step)
            candidate = f'{name}-s{step:04d}'
            run('eval', '--name', candidate, '--adapter', adapter, '--modes', 'sft_rag')
            summary = json.loads((ROOT / 'results/tuning' / candidate / 'summary.json').read_text())
            candidates.append((summary['sft_rag']['accuracy'], step, candidate, adapter))
        for _, step, candidate, adapter in sorted(candidates, reverse=True)[:2]:
            run('eval', '--name', 'validation-' + candidate, '--adapter', adapter,
                '--cache', 'data/tuning/validation_cache.jsonl', '--modes', 'sft_rag', 'sft')
    print('PHASE 3 COMPLETE', flush=True)


if __name__ == '__main__':
    main()
