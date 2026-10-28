"""Local MLX LoRA runner with real metrics, checkpoint selection and memory preflight."""
from pathlib import Path
import argparse,functools,json,math,shutil,time,hashlib
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load
from mlx_lm.lora import CONFIG_DEFAULTS,build_parser,load_dataset,train_model
from mlx_lm.tuner.callbacks import TrainingCallback

ROOT=Path(__file__).resolve().parents[1]

class Recorder(TrainingCallback):
 def __init__(self,model,out):
  self.model=model;self.out=out;self.rows=[];self.best=float('inf');self.best_step=None;self.start=time.monotonic()
 def save(self):
  (self.out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
 def on_train_loss_report(self,info):
  if not math.isfinite(float(info['train_loss'])):raise RuntimeError('Non-finite training loss')
  self.rows.append({'event':'train',**info,'wall_seconds':time.monotonic()-self.start});self.save()
 def on_val_loss_report(self,info):
  self.rows.append({'event':'validation',**info,'wall_seconds':time.monotonic()-self.start});self.save()
  if info['iteration']>0 and float(info['val_loss'])<self.best:
   self.best=float(info['val_loss']);self.best_step=info['iteration']
   mx.save_safetensors(str(self.out/'best_adapters.safetensors'),dict(tree_flatten(self.model.trainable_parameters())))

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--iters',type=int,default=3600);o=p.parse_args()
 defaults={**CONFIG_DEFAULTS,**{k:v for k,v in vars(build_parser().parse_args([])).items() if v is not None}}
 out=ROOT/'models'/('lora-smoke' if o.smoke else 'lora-adapter');out.mkdir(parents=True,exist_ok=True)
 config={'model':str(ROOT/'models/qwen3-base-mlx'),'data':str(ROOT/'data/sft'),'train':True,'test':False,
   'fine_tune_type':'lora','optimizer':'adamw','optimizer_config':{'adamw':{'weight_decay':0.01}},
   'num_layers':28,'batch_size':1,'grad_accumulation_steps':8,'iters':20 if o.smoke else o.iters,
   'learning_rate':1e-4,'max_seq_length':1536,'seed':42,'mask_prompt':True,'grad_checkpoint':False,
   'steps_per_report':5 if o.smoke else 20,'steps_per_eval':20 if o.smoke else 400,
   'save_every':20 if o.smoke else 400,'val_batches':5 if o.smoke else -1,
   'adapter_path':str(out),'lora_parameters':{'keys':['self_attn.q_proj','self_attn.v_proj'],'rank':16,'scale':2.0,'dropout':0.05},
   'report_to':None,'resume_adapter_file':None,'trust_remote_code':False}
 if not o.smoke:
  smoke=json.loads((ROOT/'results/training_smoke.json').read_text())
  if smoke['peak_memory_gb']>10:config.update(max_seq_length=1024,grad_checkpoint=True)
 args=argparse.Namespace(**(defaults|config));np.random.seed(42);mx.random.seed(42)
 model,tokenizer=load(args.model,tokenizer_config={'trust_remote_code':False},trust_remote_code=False)
 # Train exactly the same non-thinking chat template used to construct the data.
 raw=getattr(tokenizer,'_tokenizer',tokenizer)
 raw.apply_chat_template=functools.partial(raw.apply_chat_template,enable_thinking=False)
 train_set,valid_set,test_set=load_dataset(args,tokenizer)
 recorder=Recorder(model,out);begin=time.monotonic()
 train_model(args,model,train_set,valid_set,recorder)
 elapsed=time.monotonic()-begin
 if (out/'best_adapters.safetensors').exists():
  shutil.copy2(out/'adapters.safetensors',out/'last_adapters.safetensors');shutil.copy2(out/'best_adapters.safetensors',out/'adapters.safetensors')
 rows=recorder.rows
 peak=max([float(x.get('peak_memory',0)) for x in rows]+[mx.get_peak_memory()/1e9])
 params=sum(int(v.size) for _,v in tree_flatten(model.trainable_parameters()))
 summary={'kind':'smoke' if o.smoke else 'formal','completed':True,'iterations':args.iters,'max_epochs':3,
  'actual_epochs':args.iters/len(train_set),'training_samples':len(train_set),'validation_samples':len(valid_set),
  'elapsed_seconds':elapsed,'peak_memory_gb':peak,'trainable_parameters':params,'best_step':recorder.best_step,
  'best_validation_loss':recorder.best if math.isfinite(recorder.best) else None,'config':config,
  'dataset_sha256':hashlib.file_digest((ROOT/'data/sft/train.jsonl').open('rb'),'sha256').hexdigest(),
  'adapter_sha256':hashlib.file_digest((out/'adapters.safetensors').open('rb'),'sha256').hexdigest()}
 result=ROOT/'results'/('training_smoke.json' if o.smoke else 'training_summary.json');result.write_text(json.dumps(summary,indent=2))
 print('TRAINING COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
