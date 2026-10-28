"""Frozen, deterministic reference-fact scoring. No external or self-judging LLM."""
from __future__ import annotations
from collections import defaultdict
from decimal import Decimal
import csv,json,re,time,statistics
import numpy as np
import psutil
from .paths import DATA,RESULTS,read_jsonl,write_jsonl,save_json,digest
from .engine import get_engine,GENERATION

def normalized(s):
    s=re.sub(r'\[\[C\d+\]\]','',s).lower()
    return re.sub(r'[\s，。；、：,;:”“"《》()]','',s).replace('％','%').replace('：',':')

def fact_match(key,answer):
    k=normalized(key);a=normalized(answer)
    if k in a:return True
    aliases={'中国共产党南京邮电大学委员会':['校党委','学校党委'],'李恒':['李恒'],'不委托任何中介机构或个人':['不委托中介','不委托任何中介'],
      '本人当天不能再借':['当天不能再借','当天不可再借','当天不能重新借'],'过期图书不接受续借':['过期不能续借','过期图书不能续借','逾期不能续借'],
      '校园一卡通和离校单':['校园卡和离校单','一卡通和离校单'],'辅导员开具钥匙丢失':['辅导员出具','辅导员开具'],
      '二级学院提出申请':['向二级学院申请','向所在学院申请'],'上岗学生培训':['岗前培训','上岗培训']}
    if k in aliases and any(normalized(x) in a for x in aliases[k]):return True
    nums=re.findall(r'\d+(?:\.\d+)?',k)
    if nums:
        available=re.findall(r'(?<![\d.])\d+(?:\.\d+)?(?![\d.])',a)
        if not all(any(Decimal(x)==Decimal(y) for y in available) for x in nums):return False
        if '%' in k and '%' not in a and '百分之' not in a:return False
        if '天' in k and '天' not in a and '日' not in a:return False
        if '册' in k and not any(u in a for u in ['册','本']):return False
        if '元' in k and '元' not in a:return False
        if '次' in k and '次' not in a:return False
        if '月' in k and '月' not in a:return False
        return True
    return False

def grade(case,result):
    answer=result['answer'];citations=result['citations']
    refused=bool(re.search('证据不足|资料不足|无法|不能确定|不能提供|无法确认|不提供|不确定|未提供|未公布|未找到|无.*资料|需.*登录|不足以',answer))
    if not case['answerable']:
        return {'correct':refused,'fact_fraction':None,'refused':refused,'citation_locatable':None,'gold_evidence_hit':None,'reference_facts_in_evidence':None}
    facts=[fact_match(k,answer) for k in case['required_facts']]
    actual_ids={c['id'] for c in citations};used=re.findall(r'\[\[(C\d+)\]\]',answer)
    evidence_text='\n'.join(c['content'] for c in citations)
    return {'correct':bool(facts) and all(facts),'fact_fraction':sum(facts)/len(facts),'fact_matches':facts,'refused':refused,
      'citation_locatable':bool(used) and all(x in actual_ids for x in used),
      'gold_evidence_hit':bool(set(case['gold_doc_ids'])&{c['doc_id'] for c in citations}),
      'reference_facts_in_evidence':all(fact_match(k,evidence_text) for k in case['required_facts']) if citations else False}

def csv_export(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['问题','答案','参考资料'])
        for row in rows:
            refs='\n'.join(f'[{c["id"]}] {c["title"]}'+(f' PDF第{c["page"]}页' if c.get('page') else '')+' '+c['url'] for c in row.get('citations',[]))
            writer.writerow([row['question'],row['answer'],refs])

def summarize(rows):
    valid=[r for r in rows if not r.get('error')];positive=[r for r in valid if r['case']['answerable']];negative=[r for r in valid if not r['case']['answerable']]
    avg=lambda seq:sum(seq)/len(seq) if seq else None
    summary={'cases':len(rows),'completed':len(valid),'errors':len(rows)-len(valid),'accuracy':avg([int(r['grade']['correct']) for r in valid]),
      'answerable_fact_accuracy':avg([int(r['grade']['correct']) for r in positive]),'unanswerable_refusal_accuracy':avg([int(r['grade']['correct']) for r in negative]),
      'citation_locatable_rate':avg([int(r['grade']['citation_locatable']) for r in positive]),
      'reference_fact_evidence_coverage':avg([int(r['grade']['reference_facts_in_evidence']) for r in positive]),
      'latency_mean_ms':avg([r['elapsed_ms'] for r in valid]),'latency_p95_ms':float(np.percentile([r['elapsed_ms'] for r in valid],95)) if valid else None,
      'by_topic':{t:{'n':len([r for r in valid if r['case']['topic']==t]),'accuracy':avg([int(r['grade']['correct']) for r in valid if r['case']['topic']==t])} for t in set(r['case']['topic'] for r in valid)}}
    return summary

def evaluate(split='test',modes=None,calculator=False,max_tokens=768):
    modes=modes or ['raw','sft','rag','sft_rag'];cases=read_jsonl(DATA/f'eval/{split}.jsonl')
    suffix='_calculator' if calculator else '';folder=RESULTS/f'{split}{suffix}';folder.mkdir(parents=True,exist_ok=True)
    protocol={'split':split,'calculator':calculator,'guardrails':False,'generation':GENERATION,'max_tokens':max_tokens,
       'scoring':'Deterministic reference-fact/number rules, not human grading or an LLM judge; citation location and reference-fact coverage are separate from entailment.',
       'dataset_sha256':__import__('hashlib').file_digest((DATA/f'eval/{split}.jsonl').open('rb'),'sha256').hexdigest()}
    protocol_path=folder/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        changed=[k for k in ['split','calculator','guardrails','generation','max_tokens','dataset_sha256'] if previous.get(k)!=protocol[k]]
        if changed:raise ValueError('已有结果的实验配置不同（'+','.join(changed)+'）。请先另存旧结果目录，再运行新实验。')
    save_json(protocol_path,protocol)
    engine=get_engine()
    all_summaries={}
    for mode in modes:
        out=folder/f'{mode}.jsonl';rows=read_jsonl(out);done={r['case']['id'] for r in rows if not r.get('error')}
        rows=[r for r in rows if not r.get('error')]
        if rows:
            import httpx
            from .engine import OLLAMA,MODEL_NAMES
            with httpx.Client(timeout=5,trust_env=False) as client:tags=client.get(OLLAMA+'/api/tags').json().get('models',[])
            live=next((t.get('digest') for t in tags if t['name'].split(':')[0]==MODEL_NAMES[mode]),None)
            if not live or any(r.get('model_digest') and r['model_digest']!=live for r in rows):
                raise ValueError('已保存结果的模型版本与当前部署不同，请另存旧结果后重新评测。')
        for case in cases:
            if case['id'] in done:continue
            try:
                result=engine.answer(case['question'],mode,topic=None,calculator=calculator,guardrails=False,max_tokens=max_tokens)
                row={'case':case,'question':case['question'],**result,'grade':grade(case,result),'python_rss_mb':psutil.Process().memory_info().rss/1e6}
                print(mode,case['id'],row['grade']['correct'],round(result['elapsed_ms']),flush=True)
            except Exception as e:
                row={'case':case,'question':case['question'],'answer':'','citations':[],'error':str(e)}
                print('ERROR',mode,case['id'],str(e),flush=True)
            rows.append(row);write_jsonl(out,rows)
        csv_export(folder/f'{mode}.csv',rows)
        all_summaries[mode]=summarize(rows);save_json(folder/'summary.json',all_summaries)
    save_json(folder/'protocol.json',protocol)
    return all_summaries

def retrieval_evaluation():
    engine=get_engine();retriever=engine.retrieval();cases=[c for c in read_jsonl(DATA/'eval/test.jsonl') if c['answerable']]
    summaries={};details=[]
    for mode in ['bm25','dense','hybrid','full']:
        for case in cases:
            refs,timings=retriever.retrieve(case['question'],mode=mode,k=5)
            relevant=[]
            for rank,c in enumerate(refs,1):
                hit=c['doc_id'] in case['gold_doc_ids'] and (not case['gold_pages'] or c.get('page') in case['gold_pages'])
                if hit:relevant.append(rank)
            details.append({'mode':mode,'case_id':case['id'],'recall_at_5':float(bool(relevant)),'mrr_at_5':1/min(relevant) if relevant else 0,
                'granularity':'gold document + PDF page where available; HTML document otherwise','refs':refs,'timings':timings})
        subset=[r for r in details if r['mode']==mode]
        summaries[mode]={'cases':len(subset),'recall_at_5':statistics.mean(r['recall_at_5'] for r in subset),
            'mrr_at_5':statistics.mean(r['mrr_at_5'] for r in subset),'mean_ms':statistics.mean(r['timings']['total_ms'] for r in subset),
            'degraded_cases':sum(bool(r['timings'].get('warnings')) for r in subset)}
        print('Retrieval',mode,summaries[mode],flush=True)
        write_jsonl(RESULTS/'retrieval_details.jsonl',details);save_json(RESULTS/'retrieval_summary.json',summaries)
    return summaries

def acceptance():
    engine=get_engine();rows=[]
    for case in read_jsonl(DATA/'eval/acceptance.jsonl'):
        result=engine.answer(case['question'],'sft_rag',calculator=True,max_tokens=256)
        ok=all(fact_match(x,result['answer']) for x in case['expected']) and bool(result['calculation'])
        rows.append({'question':case['question'],'id':case['id'],'expected':case['expected'],'passed':ok,**result})
        print('Acceptance',case['id'],ok,result['calculation'] and result['calculation']['result'],flush=True)
        write_jsonl(RESULTS/'acceptance.jsonl',rows)
    csv_export(RESULTS/'acceptance.csv',rows)
    save_json(RESULTS/'acceptance_summary.json',{'passed':sum(r['passed'] for r in rows),'total':len(rows),'all_passed':all(r['passed'] for r in rows)})
    return rows
