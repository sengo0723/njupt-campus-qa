"""Build an auditable development-only leaderboard from completed runs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / 'results/tuning'


def main():
    groups = {}
    for protocol_file in sorted(TUNING.glob('*/protocol.json')):
        summary_file = protocol_file.with_name('summary.json')
        if not summary_file.exists():
            continue
        protocol = json.loads(protocol_file.read_text())
        if 'cache_sha256' not in protocol:
            continue
        summary = json.loads(summary_file.read_text())
        adapter = protocol['adapter']
        fingerprint = 'base'
        config = {}
        if adapter:
            weights = ROOT / adapter / 'adapters.safetensors'
            fingerprint = hashlib.sha256(weights.read_bytes()).hexdigest()
            config = json.loads((weights.parent / 'adapter_config.json').read_text())
        key = (fingerprint, protocol['strength'])
        group = groups.setdefault(key, {'name': protocol_file.parent.name.removeprefix('validation-'),
                                       'adapter': adapter, 'adapter_sha256': fingerprint,
                                       'strength': protocol['strength'], 'runs': {},
                                       'config': {k: config.get(k) for k in ['learning_rate', 'num_layers', 'lora_parameters', 'data', 'lr_schedule']}})
        for mode, metrics in summary.items():
            n = metrics['cases']
            if metrics['completed'] != n:
                continue
            partition = {20: 'development20', 60: 'validation60'}.get(n)
            if partition:
                group['runs'][partition + '/' + mode] = {'folder': protocol_file.parent.name, **metrics}
    leaderboard = []
    for group in groups.values():
        runs = group['runs']
        dev, valid = runs.get('development20/sft_rag'), runs.get('validation60/sft_rag')
        group['joint_task_accuracy'] = ((dev['accuracy']*20+valid['accuracy']*60)/80) if dev and valid else None
        group['eligible_for_final_selection'] = bool(group['adapter'] and dev and valid and 'sft_aligned' in (group['config'].get('data') or ''))
        leaderboard.append(group)
    leaderboard.sort(key=lambda row: (row['joint_task_accuracy'] if row['joint_task_accuracy'] is not None else -1), reverse=True)
    (TUNING / 'leaderboard.json').write_text(json.dumps({
        'selection': 'Development20 + validation60 only. No original test80 or fresh test60 scores enter ranking.',
        'candidates': leaderboard,
    }, ensure_ascii=False, indent=2))
    for row in leaderboard:
        def score(name):
            result = row['runs'].get(name)
            return f'{result["accuracy"]*100:.2f}%' if result else '-'
        print(row['name'], 'strength', row['strength'], 'dev', score('development20/sft_rag'),
              'validation', score('validation60/sft_rag'), 'joint', row['joint_task_accuracy'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in leaderboard:
        d, v = row['runs'].get('development20/sft_rag'), row['runs'].get('validation60/sft_rag')
        if not (d and v):
            continue
        base = not row['adapter']
        color = '#243f69' if base else '#0b9488' if 'aligned' in row['name'] else '#d6813a'
        ax.scatter(d['accuracy']*100, v['accuracy']*100, color=color, s=110 if base else 40, marker='*' if base else 'o', alpha=.8)
        if base:
            ax.annotate('Base RAG', (d['accuracy']*100, v['accuracy']*100), xytext=(5, 5), textcoords='offset points')
    ax.set(xlabel='Original development20 rule accuracy (%)', ylabel='Natural validation60 rule accuracy (%)',
           title='LoRA tuning: development evidence only', xlim=(0, 102), ylim=(0, 102))
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(TUNING / 'development_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
