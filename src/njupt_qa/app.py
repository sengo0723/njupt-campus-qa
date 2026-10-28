from __future__ import annotations
from datetime import datetime
from html import escape
from typing import Literal
import json,sqlite3,time,uuid,os,threading
from pathlib import Path
import httpx
from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse,FileResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .paths import ROOT,DATA,RESULTS,read_jsonl
from .engine import get_engine,OLLAMA,MODEL_NAMES

app=FastAPI(title='南邮知问',version='0.1.0')
SESSION_DB=Path(os.environ.get('NJUPT_SESSION_DB',str(DATA/'sessions.sqlite')))
ACTIVE_REQUESTS={}
ACTIVE_LOCK=threading.Lock()

def connection():
    db=sqlite3.connect(SESSION_DB,timeout=20);db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,title TEXT NOT NULL,updated_at TEXT NOT NULL,messages TEXT NOT NULL)')
    return db

@app.get('/api/health')
def health():
    docs=read_jsonl(DATA/'processed/documents.jsonl');models=[];error=None
    try:
        with httpx.Client(timeout=3,trust_env=False) as c:models=c.get(OLLAMA+'/api/tags').json().get('models',[])
    except Exception as e:error=str(e)[:120]
    names={m['name'].split(':')[0] for m in models}
    return {'status':'ok' if not error else 'model_unavailable','documents':len(docs),
        'model_ready':all(m in names for m in set(MODEL_NAMES.values())),
        'models':sorted(names),'index_ready':(DATA/'vectors.npy').exists(),'local_only':True,'error':error}

@app.get('/api/sessions')
def sessions():
    with connection() as db:return [dict(x) for x in db.execute('SELECT id,title,updated_at FROM sessions ORDER BY updated_at DESC')]

@app.get('/api/sessions/{sid}')
def session(sid:str):
    with connection() as db:r=db.execute('SELECT * FROM sessions WHERE id=?',(sid,)).fetchone()
    if not r:raise HTTPException(404,'会话不存在')
    return {'id':r['id'],'title':r['title'],'messages':json.loads(r['messages'])}

class Rename(BaseModel):title:str=Field(min_length=1,max_length=120)
@app.patch('/api/sessions/{sid}')
def rename(sid:str,body:Rename):
    with connection() as db:r=db.execute('UPDATE sessions SET title=? WHERE id=?',(body.title,sid));db.commit()
    if not r.rowcount:raise HTTPException(404,'会话不存在')
    return {'ok':True}

@app.delete('/api/sessions/{sid}')
def delete(sid:str):
    with connection() as db:db.execute('DELETE FROM sessions WHERE id=?',(sid,));db.commit()
    return {'ok':True}

@app.get('/api/sources/{doc_id}')
def source(doc_id:str):
    d=next((d for d in get_engine().documents if d['doc_id']==doc_id),None)
    if not d:raise HTTPException(404,'资料不存在')
    return d

@app.get('/api/sources/{doc_id}/file')
def source_file(doc_id:str):
    d=source(doc_id)
    p=(ROOT/d['raw_file']).resolve()
    if not p.is_relative_to(DATA/'raw') or not p.is_file():raise HTTPException(404,'原文文件不存在')
    if d['format']=='pdf':return FileResponse(p,media_type='application/pdf',headers={'Content-Disposition':'inline'})
    # Display a script-free text snapshot on the application origin.
    return HTMLResponse('<!doctype html><meta charset="utf-8"><title>'+escape(d['title'])+'</title><style>body{max-width:900px;margin:45px auto;padding:0 24px;font:16px/1.9 system-ui;color:#334}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#315df0}</style><h1>'+escape(d['title'])+'</h1><a rel="noreferrer" target="_blank" href="'+escape(d['url'],quote=True)+'">官方来源</a><p>发布日期：'+escape(d.get('published_at') or '未标注')+' · 本地公开资料快照</p><pre>'+escape(d['text'])+'</pre>')

@app.get('/api/documents')
def documents():return [{k:d[k] for k in ['doc_id','title','topic','url','published_at','years','format']} for d in get_engine().documents]

class ChatRequest(BaseModel):
    question:str=Field(min_length=1,max_length=2000)
    session_id:str|None=None
    mode:Literal['raw','rag','sft','sft_rag']='sft_rag'
    topic:Literal['学校概况','本科教学','招生','图书馆服务','学生事务']|None=None
    year:str|None=Field(default=None,pattern=r'^20\d{2}$')
    calculator:bool=True
    regenerate:bool=False

def persist(sid,title,messages):
    with connection() as db:
        db.execute('INSERT INTO sessions VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,messages=excluded.messages',
          (sid,title,datetime.now().astimezone().isoformat(),json.dumps(messages,ensure_ascii=False)));db.commit()

def persist_current(sid,title,messages,request_id):
    with ACTIVE_LOCK:
        if ACTIVE_REQUESTS.get(sid)!=request_id:return False
        persist(sid,title,messages)
        return True

@app.post('/api/chat/stream')
def chat(body:ChatRequest):
    sid=body.session_id or uuid.uuid4().hex
    if body.session_id:
        old=session(sid);history=old['messages'];title=old['title']
    else:history=[];title=body.question[:32]
    if body.regenerate:
        if len(history)>=2 and history[-2]['role']=='user' and history[-2]['content']==body.question:history=history[:-2]
        elif history and history[-1]['role']=='user' and history[-1]['content']==body.question:history=history[:-1]
    request_id=uuid.uuid4().hex
    with ACTIVE_LOCK:ACTIVE_REQUESTS[sid]=request_id
    def stream():
        messages=history+[{'role':'user','content':body.question}];answer='';refs=[];completed=False
        persist_current(sid,title,messages,request_id)
        def event(name,data):return 'event: '+name+'\ndata: '+json.dumps(data,ensure_ascii=False)+'\n\n'
        yield event('meta',{'session_id':sid,'request_id':request_id})
        try:
            for name,data in get_engine().generate(body.question,body.mode,body.topic,body.year,history,body.calculator):
                if name=='token':answer+=data['text']
                if name=='evidence':refs=data['citations']
                if name=='done':
                    completed=True;answer=data['answer'];refs=data['citations']
                    messages.append({'role':'assistant','content':answer,'citations':refs,'mode':body.mode,'elapsed_ms':data['elapsed_ms'],'calculation':data['calculation']})
                    persist_current(sid,title,messages,request_id)
                    with (RESULTS/'chat_traces.jsonl').open('a') as f:f.write(json.dumps({'request_id':request_id,'session_id':sid,'question':body.question,**data},ensure_ascii=False)+'\n')
                yield event(name,data)
        except GeneratorExit:raise
        except Exception as e:
            yield event('error',{'message':str(e)[:500]})
        finally:
            if not completed:
                messages.append({'role':'assistant','content':answer or '生成已停止或未完成。','citations':refs,'mode':body.mode,'stopped':True})
                persist_current(sid,title,messages,request_id)
            with ACTIVE_LOCK:
                if ACTIVE_REQUESTS.get(sid)==request_id:ACTIVE_REQUESTS.pop(sid,None)
    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

dist=ROOT/'frontend/dist'
if dist.exists():app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')
