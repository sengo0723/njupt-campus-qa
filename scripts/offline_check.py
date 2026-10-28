"""macOS process-level network isolation for both Ollama and the web backend."""
from pathlib import Path
import json,os,subprocess,time,re
import httpx
from njupt_qa.runtime import ollama_binary
ROOT=Path(__file__).resolve().parents[1];profile=ROOT/'config/offline.sb'
PY=str(ROOT/'.venv/bin/python');sandbox=['/usr/bin/sandbox-exec','-f',str(profile)]
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
 OLLAMA_HOST='127.0.0.1:11436',OLLAMA_MODELS=str(ROOT/'models/ollama'),OLLAMA_NUM_PARALLEL='1',OLLAMA_MAX_LOADED_MODELS='1',OLLAMA_NO_CLOUD='1',
 NJUPT_OLLAMA_URL='http://127.0.0.1:11436',NJUPT_PROJECT_ROOT=str(ROOT),NJUPT_SESSION_DB=str(ROOT/'results/offline_sessions.sqlite'))
def wait(url):
 for _ in range(150):
  try:
   r=httpx.get(url,timeout=2,trust_env=False)
   if r.status_code==200:return r.json()
  except Exception:pass
  time.sleep(.2)
 raise RuntimeError('Offline service failed to start '+url)
def stream_answer(payload):
 result=None
 with httpx.Client(timeout=180,trust_env=False) as client:
  with client.stream('POST','http://127.0.0.1:8766/api/chat/stream',json=payload) as r:
   r.raise_for_status();name=None
   for line in r.iter_lines():
    if line.startswith('event:'):name=line[6:].strip()
    if line.startswith('data:'):
     d=json.loads(line[5:].strip())
     if name=='error':raise RuntimeError(d['message'])
     if name=='done':result=d
 if result is None:raise RuntimeError('No completed offline answer')
 return result
def main():
 probe="import socket,json;\ntry:\n socket.create_connection(('1.1.1.1',443),timeout=3);print(json.dumps({'blocked':False}))\nexcept OSError as e:\n print(json.dumps({'blocked':True,'error':str(e)}))"
 r=subprocess.run(sandbox+[PY,'-c',probe],capture_output=True,text=True,env=env,cwd=ROOT,check=True)
 report={'external_probe':json.loads(r.stdout),'network_policy':profile.read_text(),'sandboxed_processes':['ollama','uvicorn'],'checks':[]}
 assert report['external_probe']['blocked']
 procs=[]
 try:
  for args,logname in [([ollama_binary(),'serve'],'offline_ollama.log'),([PY,'-m','uvicorn','njupt_qa.app:app','--host','127.0.0.1','--port','8766'],'offline_server.log')]:
   log=(ROOT/'results'/logname).open('w');procs.append(subprocess.Popen(sandbox+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
  wait('http://127.0.0.1:11436/api/version');health=wait('http://127.0.0.1:8766/api/health');report['health']=health
  home=httpx.get('http://127.0.0.1:8766/',trust_env=False)
  assets=re.findall(r'(?:src|href)="(/assets/[^\"]+)"',home.text)
  report['static_assets']=[{'path':a,'status':httpx.get('http://127.0.0.1:8766'+a,trust_env=False).status_code} for a in assets]
  for mode in ['raw','sft','rag','sft_rag']:
   result=stream_answer({'question':'南邮的校训是什么？','mode':mode,'calculator':False})
   report['checks'].append({'kind':'mode','mode':mode,'answer':result['answer'],'model_digest':result.get('model_digest'),'elapsed_ms':result['elapsed_ms']})
   print('Offline mode',mode,'completed',flush=True)
  from njupt_qa.paths import read_jsonl
  for case in read_jsonl(ROOT/'data/eval/acceptance.jsonl'):
   result=stream_answer({'question':case['question'],'mode':'sft_rag','calculator':True})
   passed=all(x in result['answer'] for x in case['expected']) and bool(result['calculation'])
   report['checks'].append({'kind':'acceptance','id':case['id'],'passed':passed,'answer':result['answer'],'calculation':result['calculation'],'citations':result['citations']})
   print('Offline acceptance',case['id'],passed,flush=True)
  # Test both local HTML snapshots and PDF originals while outbound traffic is denied.
  documents=httpx.get('http://127.0.0.1:8766/api/documents',trust_env=False).json()
  report['source_files']=[]
  for kind in ['html','pdf']:
   d=next(x for x in documents if x['format']==kind)
   rr=httpx.get('http://127.0.0.1:8766/api/sources/'+d['doc_id']+'/file',trust_env=False,timeout=20)
   report['source_files'].append({'format':kind,'status':rr.status_code,'bytes':len(rr.content)})
  report['passed']=report['external_probe']['blocked'] and all(x['status']==200 for x in report['static_assets']+report['source_files']) and all(x.get('passed',True) for x in report['checks'])
 finally:
  for p in reversed(procs):
   p.terminate()
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill()
 (ROOT/'results/offline_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 assert report['passed'];print('OFFLINE VERIFICATION PASSED',flush=True)
if __name__=='__main__':main()
