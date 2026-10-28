"""Adaptive data/rank/layer/schedule experiments after phase3 completes."""
from pathlib import Path
import json
import subprocess
import sys
from run_tuning_phase2 import stage_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, 'scripts/tune_lora.py', *args], cwd=ROOT, check=True)


def main():
    subprocess.run([sys.executable, 'scripts/summarize_tuning.py'], cwd=ROOT, check=True)
    candidates = json.loads((ROOT / 'results/tuning/leaderboard.json').read_text())['candidates']
    eligible = [c for c in candidates if c['eligible_for_final_selection']]
    assert eligible, 'Phase3 must finish before choosing a foundation'
    best = eligible[0]
    lr = best['config']['learning_rate']
    data = str(Path(best['config']['data']).relative_to(ROOT))
    settings = [
        ('rich-all-r16', 'data/sft_aligned_rich_long', lr, 28, 16, 1200, 0, None),
        ('rich-all-r8', 'data/sft_aligned_rich_long', lr, 28, 8, 1200, 0, None),
        ('rich-last8-r16', 'data/sft_aligned_rich_long', lr, 8, 16, 1200, 0, None),
        ('rich-last8-r8-mlp', 'data/sft_aligned_rich_long', 1e-5, 8, 8, 1200, 10,
         ['self_attn.q_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.down_proj']),
        ('aligned-twoepochs-schedule', data, lr, 28, 16, 2400, 10, None),
    ]
    (ROOT / 'results/tuning/phase4_design.json').write_text(json.dumps({
        'foundation': best['name'], 'foundation_joint_accuracy': best['joint_task_accuracy'],
        'selection_data': 'development20 + validation60 only', 'settings': settings,
    }, ensure_ascii=False, indent=2))
    for name, dataset, rate, layers, rank, iters, warmup, keys in settings:
        if not (ROOT / 'models/tuning' / name / 'training_summary.json').exists():
            command = ['train', '--name', name, '--data', dataset, '--lr', str(rate),
                       '--layers', str(layers), '--rank', str(rank), '--iters', str(iters), '--warmup', str(warmup)]
            if keys:
                command += ['--keys', *keys]
            run(*command)
        scores = []
        for step in ([200, 400, 800, 1200] if iters == 1200 else [400, 800, 1200, 1600, 2400]):
            adapter = stage_checkpoint(name, step)
            candidate = f'{name}-s{step:04d}'
            run('eval', '--name', candidate, '--adapter', adapter, '--modes', 'sft_rag')
            summary = json.loads((ROOT / 'results/tuning' / candidate / 'summary.json').read_text())
            scores.append((summary['sft_rag']['accuracy'], step, candidate, adapter))
        for _, step, candidate, adapter in sorted(scores, reverse=True)[:2]:
            run('eval', '--name', 'validation-' + candidate, '--adapter', adapter,
                '--cache', 'data/tuning/validation_cache.jsonl', '--modes', 'sft_rag', 'sft')
    subprocess.run([sys.executable, 'scripts/summarize_tuning.py'], cwd=ROOT, check=True)
    print('PHASE 4 COMPLETE', flush=True)


if __name__ == '__main__':
    main()
