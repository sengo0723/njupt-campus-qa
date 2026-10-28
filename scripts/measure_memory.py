"""Sample the real web backend, project Ollama daemon and its model runner RSS."""
from pathlib import Path
import csv,json,threading,time
import httpx,psutil
ROOT=Path(__file__).resolve().parents[1]
services=json.loads((ROOT/'results/services.json').read_text());samples=[];finished=threading.Event()
def sample():
 while not finished.is_set():
  pids={services['web_pid'],services['ollama_pid']}
  try:pids.update(p.pid for p in psutil.Process(services['ollama_pid']).children(recursive=True))
  except psutil.Error:pass
  sizes={}
  for pid in pids:
   try:sizes[str(pid)]=psutil.Process(pid).memory_info().rss
   except psutil.Error:pass
  samples.append({'seconds':time.monotonic()-begin,'rss_bytes':sum(sizes.values()),'by_pid':sizes})
  finished.wait(.1)
begin=time.monotonic();thread=threading.Thread(target=sample,daemon=True);thread.start();result=None;sid=None
try:
 with httpx.Client(timeout=180,trust_env=False) as client:
  with client.stream('POST','http://127.0.0.1:8765/api/chat/stream',json={'question':'按2026年章程，工科类优势学科专业学费加住宿费，假设标准四年不变，预收额合计多少？','mode':'sft_rag','calculator':True}) as r:
   r.raise_for_status();event=None
   for line in r.iter_lines():
    if line.startswith('event:'):event=line[6:].strip()
    elif line.startswith('data:'):
     d=json.loads(line[5:].strip())
     if event=='meta':sid=d['session_id']
     if event=='error':raise RuntimeError(d['message'])
     if event=='done':result=d
  # This session was created only for the measurement and its id came from meta.
  if sid:client.delete('http://127.0.0.1:8765/api/sessions/'+sid)
finally:
 finished.set();thread.join()
assert result and '31520' in result['answer']
report={'passed':True,'peak_summed_process_rss_bytes':max(x['rss_bytes'] for x in samples),
 'peak_summed_process_rss_gb':max(x['rss_bytes'] for x in samples)/1e9,'sample_interval_seconds':.1,
 'processes':'Actual web backend + project-specific Ollama daemon + its runner children',
 'measurement':'Sum of OS-reported RSS; shared pages may be counted more than once. This is not a measurement of unique unified-memory allocation.',
 'elapsed_ms':result['elapsed_ms'],'model_digest':result.get('model_digest'),'sample_count':len(samples)}
(ROOT/'results/inference_memory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (ROOT/'results/inference_memory_samples.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['seconds','summed_process_rss_bytes'])
 for x in samples:w.writerow([x['seconds'],x['rss_bytes']])
print(json.dumps(report,ensure_ascii=False,indent=2))
