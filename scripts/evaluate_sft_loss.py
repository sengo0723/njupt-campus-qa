"""Independent held-out assistant-token NLL, evaluated locally in MLX."""
from pathlib import Path
import argparse,functools,json
import mlx.core as mx
from mlx_lm.utils import load,load_adapters
from mlx_lm.lora import CONFIG_DEFAULTS,build_parser,load_dataset
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import evaluate
ROOT=Path(__file__).resolve().parents[1]
defaults={**CONFIG_DEFAULTS,**{k:v for k,v in vars(build_parser().parse_args([])).items() if v is not None}}
args=argparse.Namespace(**(defaults|{'data':str(ROOT/'data/sft'),'mask_prompt':True,'model':str(ROOT/'models/qwen3-base-mlx')}))
model,tokenizer=load(args.model,trust_remote_code=False)
raw=getattr(tokenizer,'_tokenizer',tokenizer);raw.apply_chat_template=functools.partial(raw.apply_chat_template,enable_thinking=False)
_,_,dataset=load_dataset(args,tokenizer);dataset=CacheDataset(dataset)
results={}
for name in ['base','sft']:
 if name=='sft':load_adapters(model,str(ROOT/'models/lora-adapter'))
 loss=evaluate(model,dataset,batch_size=1,num_batches=-1,max_seq_length=1536)
 results[name]={'assistant_token_nll':float(loss),'perplexity':float(mx.exp(mx.array(loss))),'examples':len(dataset)}
 print(name,results[name],flush=True)
(ROOT/'results/sft_heldout_loss.json').write_text(json.dumps(results,indent=2))
