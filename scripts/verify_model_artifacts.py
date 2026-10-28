"""Verify the actual adapter update and fused weight arithmetic, plus local manifests."""
from pathlib import Path
import json,hashlib
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
checks=[]
with safe_open(ROOT/'models/qwen3-base-hf/model.safetensors',framework='pt') as orig, \
     safe_open(ROOT/'models/qwen3-base-mlx/model.safetensors',framework='pt') as base, \
     safe_open(ROOT/'models/qwen3-sft-hf/model.safetensors',framework='pt') as fused, \
     safe_open(ROOT/'models/lora-adapter/adapters.safetensors',framework='pt') as adapter:
    # Verify unchanged source conversion without materializing the full embedding.
    for key in ['model.embed_tokens.weight','model.layers.0.self_attn.q_proj.weight','model.norm.weight']:
        x=orig.get_tensor(key);y=base.get_tensor(key)
        checks.append({'name':key,'base_conversion_max_abs_error':float((x.float()-y.float()).abs().max())})
    for layer in range(28):
        for proj in ['q_proj','v_proj']:
            name=f'model.layers.{layer}.self_attn.{proj}'
            w=base.get_tensor(name+'.weight');a=adapter.get_tensor(name+'.lora_a');b=adapter.get_tensor(name+'.lora_b')
            delta=((2*b.float().T)@a.float().T).to(w.dtype)
            expected=w+delta;actual=fused.get_tensor(name+'.weight')
            err=float((expected.float()-actual.float()).abs().max())
            changed=int(torch.count_nonzero(actual-w))
            checks.append({'name':name,'fused_max_abs_error':err,'changed_weight_entries':changed,'adapter_nonzero':bool(torch.count_nonzero(b))})
summary={'base_source_conversion_verified':all(c.get('base_conversion_max_abs_error',0)==0 for c in checks),
    'all_56_adapted_layers_changed':all(c.get('changed_weight_entries',1)>0 for c in checks),
    'fused_weight_max_abs_error':max(c.get('fused_max_abs_error',0) for c in checks),'checks':checks,
    'gguf_files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in (ROOT/'models').glob('*.gguf')}}
(ROOT/'results/model_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
assert summary['base_source_conversion_verified'] and summary['all_56_adapted_layers_changed']
assert summary['fused_weight_max_abs_error']<.01
