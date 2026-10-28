from __future__ import annotations
import json,os,re,threading,time
from urllib.parse import urlsplit
import httpx
from .paths import DATA,MODELS,RESULTS,read_jsonl
from .retrieval import Retriever
from .calculations import calculate,render_calculation
from .dataset import SYSTEM

OLLAMA=os.environ.get('NJUPT_OLLAMA_URL','http://127.0.0.1:11435')
if urlsplit(OLLAMA).hostname not in ['127.0.0.1','localhost','::1']:
    raise ValueError('Inference endpoint must be local loopback')
MODEL_NAMES={'raw':'njupt-qwen3-base','rag':'njupt-qwen3-base','sft':'njupt-qwen3-sft','sft_rag':'njupt-qwen3-sft'}
GENERATION={'num_ctx':8192,'num_predict':768,'temperature':0,'seed':42,'repeat_penalty':1.1}
MODEL_LOCK=threading.Lock()

def production_refusal(question):
    if re.search('实时|现在.*空座|保证.*录取|私人.*(手机|电话)',question):return '当前公开资料不能提供实时状态、私人联系方式或保证录取的结论。请查询学校相应官方系统或联系主管部门。'
    if re.search('成绩|课表|到账|余额|借阅历史|宿舍门牌|个人电话',question) and re.search('我的|我本人|个人|某位|学号|同学',question):
        return '这需要本人登录学校官方系统查询。当前公开资料无法确认个人结果，请勿提供账户密码或其他同学的个人信息。'
    if any(int(y)>2026 for y in re.findall(r'20\d{2}',question)):
        return '本地公开资料快照尚无该年份的有效政策，当前证据不足，不能推测未来招生计划或办理安排。'
    return None

class Engine:
    def __init__(self):
        self.documents=read_jsonl(DATA/'processed/documents.jsonl');self.retriever=None;self._init_lock=threading.Lock()
    def retrieval(self):
        with self._init_lock:
            if self.retriever is None:self.retriever=Retriever()
        return self.retriever
    def prepare(self,question,mode,topic=None,year=None,history=None,calculator=True,retrieval_mode='full',guardrails=True):
        history=history or [];refusal=production_refusal(question) if guardrails else None
        citations=[];timings={};calc=None
        if mode in ['rag','sft_rag'] and not refusal:
            query=question
            if len(question)<18 or re.search(r'^(那|它|这些|上述|再|其中)',question):
                previous=next((m['content'] for m in reversed(history) if m['role']=='user'),'')
                query=previous+'\n追问：'+question
            citations,timings=self.retrieval().retrieve(query,topic,year,retrieval_mode)
            # History expands retrieval, but its previous operation must not
            # override the intent of a new follow-up question.
            if calculator:calc=calculate(question,self.documents,citations)
            if calc:
                citations=calc['sources']
                for i,c in enumerate(citations):c['id']='C'+str(i+1)
            if not citations:refusal='当前所选主题和年份下没有足够的公开资料，无法给出可靠结论。请调整筛选或补充资料。'
        blocks=[]
        for c in citations:
            loc=f'PDF第{c["page"]}页' if c.get('page') else '网页正文'
            # Keep full calculation tables; ordinary windows were already token-bounded.
            content=c['content'][:5500 if calc else 1400]
            blocks.append(f'[[{c["id"]}]] 标题：{c["title"]}；位置：{loc}；发布日期：{c.get("published_at") or "未标注"}\n{content}')
        context='\n\n'.join(blocks)
        prompt=f'参考资料：\n{context}\n\n问题：{question}' if citations else question
        if calc:prompt+='\n\n本地计算器已核验：'+calc['formula']+'。请按该结果解释，不重新猜测数值。'
        if mode in ['rag','sft_rag']:
            prompt+='\n请只依据参考资料回答。引用格式为[[C1]]，使用实际提供的编号。资料不含答案时明确说证据不足，不得推测。'
        messages=[{'role':'system','content':SYSTEM}]+[{'role':m['role'],'content':m['content'][:700]} for m in history[-4:]]+[{'role':'user','content':prompt}]
        return {'messages':messages,'citations':citations,'retrieval_timings':timings,'calculation':calc,'refusal':refusal}
    def generate(self,question,mode='sft_rag',topic=None,year=None,history=None,calculator=True,retrieval_mode='full',guardrails=True,max_tokens=None):
        begin=time.perf_counter()
        yield 'stage',{'message':'正在检索公开资料' if mode in ['rag','sft_rag'] else '正在准备本地模型'}
        prepared=self.prepare(question,mode,topic,year,history,calculator,retrieval_mode,guardrails)
        yield 'evidence',{'citations':prepared['citations']}
        model=MODEL_NAMES[mode];calc=prepared['calculation'];draft='';usage={};model_digest=None
        if prepared['refusal']:
            answer=prepared['refusal'];yield 'token',{'text':answer};method='evidence_guard'
        else:
            yield 'stage',{'message':'正在核验计算并组织回答' if calc else '正在组织回答'}
            options={**GENERATION}
            if max_tokens:options['num_predict']=max_tokens
            with MODEL_LOCK,httpx.Client(timeout=httpx.Timeout(180,connect=5),trust_env=False) as client:
                # Fail explicitly if the requested SFT model has not been registered.
                tags=client.get(OLLAMA+'/api/tags').json().get('models',[])
                if not any(x['name'].split(':')[0]==model for x in tags):
                    raise RuntimeError(f'本地模型 {model} 尚未准备完成，请运行模型部署脚本。')
                model_digest=next(x.get('digest') for x in tags if x['name'].split(':')[0]==model)
                payload={'model':model,'messages':prepared['messages'],'stream':True,'think':False,'options':options,'keep_alive':'5m'}
                with client.stream('POST',OLLAMA+'/api/chat',json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:continue
                        item=json.loads(line)
                        if 'error' in item:raise RuntimeError(item['error'])
                        token=item.get('message',{}).get('content','')
                        draft+=token
                        if token and not calc:yield 'token',{'text':token}
                        if item.get('done'):usage={k:item.get(k) for k in ['prompt_eval_count','eval_count','total_duration','eval_duration','load_duration','done_reason']}
            answer=draft.strip();method='model'
            if calc:
                answer=render_calculation(calc,[c['id'] for c in prepared['citations']]);method='verified_calculator_with_model_draft'
                for offset in range(0,len(answer),24):yield 'token',{'text':answer[offset:offset+24]}
            elif prepared['citations']:
                allowed={c['id'] for c in prepared['citations']}
                answer=re.sub(r'\[\[(C\d+)\]\]',lambda m:m[0] if m[1] in allowed else '',answer)
                if not re.search(r'\[\[C\d+\]\]',answer) and not re.search('证据不足|无法|未提供|未提及',answer):
                    answer+='\n\n检索参考：'+''.join(f'[[{c["id"]}]]' for c in prepared['citations'][:2])
            else:
                answer=re.sub(r'\[\[C\d+\]\]','',answer)
        result={'answer':answer,'citations':prepared['citations'],'mode':mode,'model':model,
            'elapsed_ms':(time.perf_counter()-begin)*1000,'retrieval_timings':prepared['retrieval_timings'],
            'calculation':({k:v for k,v in calc.items() if k!='sources'} if calc else None),
            'generation_method':method,'model_draft':draft if calc else None,'usage':usage,'guardrails':guardrails,'model_digest':model_digest}
        yield 'done',result
    def answer(self,*args,**kwargs):
        for event,result in self.generate(*args,**kwargs):
            if event=='done':return result

_engine=None
def get_engine():
    global _engine
    if _engine is None:_engine=Engine()
    return _engine
