import json,re
import pytest
from collections import Counter
from njupt_qa.paths import DATA,read_jsonl
from njupt_qa.calculations import calculate
from njupt_qa.crawl import html_table
from njupt_qa.engine import production_refusal
from njupt_qa.evaluation import grade
from bs4 import BeautifulSoup

def docs():return read_jsonl(DATA/'processed/documents.jsonl')
def all_refs():return [{'doc_id':d['doc_id']} for d in docs()]

def test_corpus_is_official_and_covers_every_topic():
    data=docs();assert all(n>=8 for n in Counter(d['topic'] for d in data).values())
    assert len({d['doc_id'] for d in data})==len(data)
    assert len({d['content_sha256'] for d in data})==len(data)
    assert all(re.match(r'https://[^/]*\.njupt\.edu\.cn/',d['url']) for d in data)

def test_html_rowspan_restores_subjects():
    t=BeautifulSoup('<table><tr><th>科目</th><th>专业</th></tr><tr><td rowspan="2">物理</td><td>A</td></tr><tr><td>B</td></tr></table>','lxml').table
    assert html_table(t)==[['科目','专业'],['物理','A'],['物理','B']]

def test_four_acceptance_calculations_come_from_sources():
    for case in read_jsonl(DATA/'eval/acceptance.jsonl'):
        calc=calculate(case['question'],docs(),all_refs());assert calc
        assert all(x in calc['result'] or x in calc['formula'] for x in case['expected'])
        assert calc['sources'] and all(s['doc_id'] in {d['doc_id'] for d in docs()} for s in calc['sources'])

def test_fee_horizon_is_not_the_final_digit_of_2026():
    c=calculate('按2026年章程，工科类优势学科专业学费加住宿费，标准三年不变合计多少？',docs(),all_refs())
    assert c['operands']['years']==3 and c['result']=='23640元'

def test_calculator_does_not_have_hidden_access_to_unretrieved_sources():
    assert calculate('根据2023—2024学年报告反推外聘折算系数，再复算生师比。',docs(),[]) is None

def test_teacher_formula_generalizes_to_different_data():
    d={'doc_id':'example','title':'合成测试','topic':'本科教学','url':'https://example.njupt.edu.cn/','published_at':None,'years':[],
       'pages':[{'page':1,'text':'专任教师50人，聘请校外教师20人，折合教师总数为60，折合在校生人数600。'}], 'text':'','tables':[],'raw_file':'synthetic'}
    c=calculate('反推外聘折算系数，再复算生师比。',[d],[{'doc_id':'example'}])
    assert c['operands']['equivalent_teachers']=='60' and '10.00' in c['result'] and '0.5' in c['result']

def test_sft_sample_counts_lengths_and_fact_group_isolation():
    metadata=read_jsonl(DATA/'sft/provenance.jsonl');groups={}
    for split,n in [('train',1200),('valid',150),('test',150)]:
        assert len(read_jsonl(DATA/f'sft/{split}.jsonl'))==n
        groups[split]={g for x in metadata if x['split']==split for g in [x['fact_group_id']]+x.get('extra_fact_groups',[])}
    assert not groups['train']&groups['valid'];assert not groups['train']&groups['test'];assert not groups['valid']&groups['test']
    assert max(x['tokens'] for x in metadata)<=1536
    assert max(Counter(x['fact_group_id'] for x in metadata).values())<=3
    assert any(x.get('multi_turn') for x in metadata)
    assert any(x.get('multi_evidence') for x in metadata)

def test_acceptance_documents_are_not_in_sft():
    forbidden={'3c538cd612625bbe8b7e','c69898caec483a487960','ac45ca3a781584acd021','580982b06be633fd91c1','29429770992e86830a0a'}
    assert not forbidden&{x.get('source_doc_id') for x in read_jsonl(DATA/'sft/provenance.jsonl')}

def test_eval_split_has_five_topics_and_unanswerable_cases():
    dev=read_jsonl(DATA/'eval/dev.jsonl');test=read_jsonl(DATA/'eval/test.jsonl')
    assert len(dev)==20 and len(test)==80
    assert set(c['id'] for c in dev).isdisjoint(c['id'] for c in test)
    assert all(n==20 for n in Counter(c['topic'] for c in dev+test).values())
    assert sum(not c['answerable'] for c in dev+test)==10

def test_private_and_future_requests_are_not_invented():
    assert production_refusal('请查我的个人期末考试成绩。')
    assert production_refusal('南京邮电大学2028年综合评价招生计划是多少？')
    assert not production_refusal('2026年综合评价招生计划是多少？')

@pytest.mark.models
def test_token_chunks_preserve_source_chinese_without_inserted_spaces():
    from njupt_qa.paths import MODELS
    if not (MODELS/'bge-small-zh-v1.5/tokenizer_config.json').exists():
        pytest.skip('Download the pinned BGE model before running tokenizer integration checks')
    from transformers import AutoTokenizer
    from njupt_qa.retrieval import chunk_documents
    t=AutoTokenizer.from_pretrained(MODELS/'bge-small-zh-v1.5',local_files_only=True)
    text='南京邮电大学的校训是厚德、弘毅、求是、笃行。The acronym is NJUPT.'
    d={'doc_id':'x','title':'测试','topic':'学校概况','url':'https://www.njupt.edu.cn/','published_at':None,'years':[], 'pages':[],'text':text,'tables':[],'raw_file':'x'}
    c=chunk_documents([d],t)
    assert c[0]['text']==text

def test_grading_does_not_count_citation_number_as_a_numeric_answer():
    case={'answerable':True,'required_facts':['1册'],'gold_doc_ids':['a']}
    result={'answer':'参考[[C1]]。','citations':[{'id':'C1','doc_id':'a','content':'本科生每证限借1册'}]}
    assert not grade(case,result)['correct']

def test_partial_table_match_cannot_be_claimed_as_a_verified_three_center_sum():
    q='信息与通信、计算机和不存在的第三个中心，三个实验教学中心面积之和占总面积多少？'
    assert calculate(q,docs(),all_refs()) is None

def test_follow_up_uses_current_operation_not_previous_sum():
    from types import SimpleNamespace
    from njupt_qa.engine import Engine
    engine=Engine();d=next(x for x in docs() if '2023—2024学年本科教学质量报告' in x['title'])
    c={'doc_id':d['doc_id'],'id':'C1','title':d['title'],'page':9,'published_at':None,'content':'表内总面积69864平方米。'}
    engine.retrieval=lambda:SimpleNamespace(retrieve=lambda *args,**kwargs:([c],{}))
    prepared=engine.prepare('那报告表内总面积是多少？','rag',history=[{'role':'user','content':'信息与通信、电子与光学、计算机三个实验教学中心面积之和占报告表内总面积多少？'}])
    assert prepared['calculation'] is None
    assert '69864' in prepared['messages'][-1]['content']

def test_session_crud_does_not_touch_live_database(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import njupt_qa.app as web
    monkeypatch.setattr(web,'SESSION_DB',tmp_path/'sessions.sqlite')
    web.persist('test-session','测试名称',[{'role':'user','content':'测试问题'}])
    with TestClient(web.app) as client:
        assert client.get('/api/sessions/test-session').json()['title']=='测试名称'
        assert client.patch('/api/sessions/test-session',json={'title':'更新名称'}).status_code==200
        assert client.get('/api/sessions/test-session').json()['title']=='更新名称'
        assert client.patch('/api/sessions/test-session',json={'title':''}).status_code==422
        assert client.delete('/api/sessions/test-session').status_code==200
        assert client.get('/api/sessions/test-session').status_code==404

def test_regenerate_after_interrupted_user_message_does_not_duplicate_turn(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import njupt_qa.app as web
    monkeypatch.setattr(web,'SESSION_DB',tmp_path/'sessions.sqlite');monkeypatch.setattr(web,'RESULTS',tmp_path)
    monkeypatch.setattr(web,'ACTIVE_REQUESTS',{})
    class FakeEngine:
        def generate(self,*args,**kwargs):
            yield 'done',{'answer':'测试完成','citations':[],'elapsed_ms':1,'calculation':None}
    monkeypatch.setattr(web,'get_engine',lambda:FakeEngine())
    web.persist('cancelled','停止测试',[{'role':'user','content':'问题'}])
    with TestClient(web.app) as client:
        r=client.post('/api/chat/stream',json={'session_id':'cancelled','question':'问题','regenerate':True,'mode':'raw'})
        assert r.status_code==200
        messages=client.get('/api/sessions/cancelled').json()['messages']
        assert [m['role'] for m in messages]==['user','assistant']

def test_superseded_completion_cannot_overwrite_a_newer_turn(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import njupt_qa.app as web
    monkeypatch.setattr(web,'SESSION_DB',tmp_path/'sessions.sqlite');monkeypatch.setattr(web,'RESULTS',tmp_path)
    monkeypatch.setattr(web,'ACTIVE_REQUESTS',{})
    newer=[{'role':'user','content':'新问题'},{'role':'assistant','content':'新回答'}]
    class FakeEngine:
        def generate(self,*args,**kwargs):
            web.ACTIVE_REQUESTS['race']='newer-request'
            web.persist('race','新会话',newer)
            yield 'done',{'answer':'旧回答','citations':[],'elapsed_ms':1,'calculation':None}
    monkeypatch.setattr(web,'get_engine',lambda:FakeEngine())
    web.persist('race','旧会话',[])
    with TestClient(web.app) as client:
        client.post('/api/chat/stream',json={'session_id':'race','question':'旧问题','mode':'raw'})
        assert client.get('/api/sessions/race').json()['messages']==newer

def test_resume_rejects_changed_generation_protocol(tmp_path,monkeypatch):
    import pytest
    import njupt_qa.evaluation as ev
    p=tmp_path/'test';p.mkdir()
    from njupt_qa.paths import ROOT
    original=json.loads((ROOT/'experiments/2026-10-26/test/protocol.json').read_text());original['max_tokens']=128
    (p/'protocol.json').write_text(json.dumps(original))
    monkeypatch.setattr(ev,'RESULTS',tmp_path)
    with pytest.raises(ValueError,match='max_tokens'):ev.evaluate(max_tokens=768)
