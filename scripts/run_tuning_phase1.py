"""Existing-checkpoint and adapter-strength development comparison."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
jobs = [(f'legacy-{step:04d}', f'models/tuning/legacy-{step:04d}', 1.) for step in [400, 800, 1200, 2400, 3600]]
jobs += [(f'legacy-0800-strength-{strength}', 'models/tuning/legacy-0800', strength) for strength in [.1, .25, .5, .75]]
for name, adapter, strength in jobs:
    subprocess.run([sys.executable, 'scripts/tune_lora.py', 'eval', '--name', name,
                    '--adapter', adapter, '--strength', str(strength)], cwd=ROOT, check=True)
print('PHASE 1 COMPLETE', flush=True)
