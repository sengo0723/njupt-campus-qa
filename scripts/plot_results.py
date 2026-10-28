from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results'
rows=[json.loads(x) for x in (ROOT/'models/lora-adapter/metrics.jsonl').read_text().splitlines()]
train=[x for x in rows if x['event']=='train'];valid=[x for x in rows if x['event']=='validation']
summary=json.loads((out/'training_summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
ax.plot([x['iteration'] for x in train],[x['train_loss'] for x in train],color='#4978db',alpha=.8,label='Training loss')
ax.plot([x['iteration'] for x in valid],[x['val_loss'] for x in valid],color='#df8c39',marker='o',markersize=4,label='Validation loss')
ax.axvline(summary['best_step'],ls='--',color='#8190a6',lw=1,label=f'Selected checkpoint: {summary["best_step"]}')
ax.set(xlabel='Micro-steps (batch size 1; gradient accumulation 8)',ylabel='Assistant-token cross entropy',title='Qwen3-0.6B LoRA SFT: actual training run',yscale='log')
ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9);fig.savefig(out/'training_loss.png',dpi=180);plt.close(fig)
scores=json.loads((out/'test/summary.json').read_text());modes=['raw','sft','rag','sft_rag']
fig,ax=plt.subplots(figsize=(7.5,4.3),layout='constrained')
values=[scores[m]['accuracy']*100 for m in modes]
bars=ax.bar(['Base','LoRA SFT','Base + RAG','LoRA SFT + RAG'],values,color=['#b5c1d4','#91b8b2','#4978db','#8e8bc2'],width=.6)
for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+1.5,f'{v:.2f}%',ha='center',fontsize=10)
ax.set(ylim=(0,100),ylabel='Rule-based reference-fact score (%)',title='Frozen campus test: 80 questions, calculator disabled')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
print('Saved actual training curves and frozen-test comparison')
