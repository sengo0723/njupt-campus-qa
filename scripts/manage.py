"""Start/stop only this project's local services. Existing system Ollama is untouched."""
from pathlib import Path
import argparse,json,os,subprocess,time,sys,webbrowser
import httpx,psutil
from njupt_qa.runtime import ollama_binary
ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
STATE=RESULTS/'services.json'

def request(url):
 try:return httpx.get(url,timeout=2,trust_env=False).json()
 except Exception:return None
def wait_ready(url,seconds=40):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  result=request(url)
  if result is not None:return result
  time.sleep(.3)
 raise RuntimeError('Service did not become ready: '+url)

def owned(proc):
 try:
  env=proc.environ()
  return env.get('NJUPT_PROJECT_ROOT')==str(ROOT) or env.get('OLLAMA_MODELS')==str(ROOT/'models/ollama')
 except (psutil.Error,OSError):return False

def start(open_browser=False,model_server_only=False):
 state=json.loads(STATE.read_text()) if STATE.exists() else {}
 env=os.environ.copy();env.update(NJUPT_PROJECT_ROOT=str(ROOT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
     OLLAMA_HOST='127.0.0.1:11435',OLLAMA_MODELS=str(ROOT/'models/ollama'),OLLAMA_NUM_PARALLEL='1',OLLAMA_MAX_LOADED_MODELS='1',OLLAMA_NO_CLOUD='1')
 if not request('http://127.0.0.1:11435/api/version'):
  log=(RESULTS/'ollama.log').open('a')
  p=subprocess.Popen([ollama_binary(),'serve'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state['ollama_pid']=p.pid;STATE.write_text(json.dumps(state));wait_ready('http://127.0.0.1:11435/api/version')
 else:
  found_owned=False
  for proc in psutil.process_iter(['pid','name']):
   if 'ollama' in (proc.info['name'] or '').lower() and owned(proc):
    try:
     if 'serve' in proc.cmdline() and proc.environ().get('OLLAMA_HOST')=='127.0.0.1:11435':
      state['ollama_pid']=proc.pid;found_owned=True;break
    except psutil.Error:pass
  if not found_owned:raise RuntimeError('Port 11435 belongs to another project; stop that project before starting this copy.')
 if model_server_only:
  STATE.write_text(json.dumps(state,indent=2));print('项目 Ollama 已就绪：127.0.0.1:11435');return
 tags=request('http://127.0.0.1:11435/api/tags') or {};names={m['name'].split(':')[0] for m in tags.get('models',[])}
 for tag,file in [('njupt-qwen3-base','base.Modelfile'),('njupt-qwen3-sft','sft.Modelfile')]:
  if tag not in names:
   gguf=ROOT/'models'/('qwen3-base-q8_0.gguf' if tag.endswith('base') else 'qwen3-sft-q8_0.gguf')
   if not gguf.exists():
    print('Model not prepared; its modes will be unavailable:',tag);continue
   subprocess.run([ollama_binary(),'create',tag,'-f',str(ROOT/'config'/file)],env=env,cwd=ROOT,check=True)
 web_ready=request('http://127.0.0.1:8765/api/health')
 if web_ready:
  try:web_owned=owned(psutil.Process(state.get('web_pid',-1)))
  except psutil.Error:web_owned=False
  if not web_owned:raise RuntimeError('Port 8765 belongs to another project; stop that project before starting this copy.')
 if not web_ready:
  log=(RESULTS/'server.log').open('a')
  p=subprocess.Popen([sys.executable,'-m','uvicorn','njupt_qa.app:app','--host','127.0.0.1','--port','8765'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state['web_pid']=p.pid;STATE.write_text(json.dumps(state));wait_ready('http://127.0.0.1:8765/api/health')
 STATE.write_text(json.dumps(state,indent=2));print('南邮知问已启动：http://127.0.0.1:8765')
 if open_browser:webbrowser.open('http://127.0.0.1:8765')

def stop():
 state=json.loads(STATE.read_text()) if STATE.exists() else {}
 for key in ['web_pid','ollama_pid']:
  if key not in state:continue
  try:
   proc=psutil.Process(state[key])
   if owned(proc):proc.terminate();proc.wait(15);print('Stopped',key)
  except psutil.Error:pass
 STATE.write_text('{}')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop','status','model-server']);p.add_argument('--open',action='store_true');a=p.parse_args()
 if a.action=='start':start(a.open)
 elif a.action=='stop':stop()
 elif a.action=='model-server':start(model_server_only=True)
 else:print(json.dumps(request('http://127.0.0.1:8765/api/health'),ensure_ascii=False,indent=2))
