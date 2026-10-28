"""Wait for the final trained artifact, then run all comparisons sequentially."""
from pathlib import Path
import json,time,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
data_hash=hashlib.file_digest((ROOT/'data/sft/train.jsonl').open('rb'),'sha256').hexdigest()
deadline=time.monotonic()+3600
while time.monotonic()<deadline:
 try:
  p=ROOT/'results/training_summary.json';s=json.loads(p.read_text());loss=ROOT/'results/sft_heldout_loss.json'
  if s['dataset_sha256']==data_hash and loss.exists() and loss.stat().st_mtime>p.stat().st_mtime:break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(5)
else:raise RuntimeError('Final LoRA pipeline did not finish')
from njupt_qa.paths import save_json
from njupt_qa.evaluation import evaluate,retrieval_evaluation,acceptance
print('Final model ready; dataset',data_hash,flush=True)
evaluate('dev')
evaluate('test')
evaluate('test',['rag','sft_rag'],calculator=True)
retrieval_evaluation()
acceptance()
save_json(ROOT/'results/benchmark_complete.json',{'complete':True,'training_dataset_sha256':data_hash,'ended_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'main_test_cases':80,'main_modes':['raw','sft','rag','sft_rag'],'calculator_modes':['rag','sft_rag']})
print('ALL BENCHMARKS COMPLETE',flush=True)
