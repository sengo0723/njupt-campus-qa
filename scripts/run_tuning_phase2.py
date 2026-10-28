"""Learning-rate sweep on original data, selected using task development sets."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, 'scripts/tune_lora.py', *args], cwd=ROOT, check=True)


def stage_checkpoint(run_name, step):
    source = ROOT / 'models/tuning' / run_name
    target = ROOT / 'models/tuning' / f'{run_name}-s{step:04d}'
    target.mkdir(exist_ok=True)
    shutil.copy2(source / 'adapter_config.json', target / 'adapter_config.json')
    shutil.copy2(source / f'{step:07d}_adapters.safetensors', target / 'adapters.safetensors')
    return str(target.relative_to(ROOT))


def main():
    run('eval', '--name', 'validation-base', '--cache', 'data/tuning/validation_cache.jsonl', '--modes', 'sft_rag', 'sft')
    for name, strength in [('legacy-0400', 1.), ('legacy-0800', 1.), ('legacy-0800', .1)]:
        run('eval', '--name', f'validation-{name}-{strength}', '--adapter', f'models/tuning/{name}',
            '--strength', str(strength), '--cache', 'data/tuning/validation_cache.jsonl', '--modes', 'sft_rag')
    for name, lr in [('original-lr1e5', 1e-5), ('original-lr3e5', 3e-5), ('original-lr5e5', 5e-5)]:
        if not (ROOT / 'models/tuning' / name / 'training_summary.json').exists():
            run('train', '--name', name, '--lr', str(lr), '--iters', '1200')
        candidates = []
        for step in [200, 400, 800, 1200]:
            adapter = stage_checkpoint(name, step)
            candidate = f'{name}-s{step:04d}'
            run('eval', '--name', candidate, '--adapter', adapter)
            summary = json.loads((ROOT / 'results/tuning' / candidate / 'summary.json').read_text())
            candidates.append((summary['sft_rag']['accuracy'], step, candidate, adapter))
        for _, step, candidate, adapter in sorted(candidates, reverse=True)[:2]:
            run('eval', '--name', 'validation-' + candidate, '--adapter', adapter,
                '--cache', 'data/tuning/validation_cache.jsonl', '--modes', 'sft_rag', 'sft')
    print('PHASE 2 COMPLETE', flush=True)


if __name__ == '__main__':
    main()
