"""Separate arithmetic challenge; gold values are independently transcribed from official sources.

These are same-source operation variations, excluded from training and from the main
campus score. They quantify calculator contribution without changing the frozen test.
"""
from pathlib import Path
from decimal import Decimal
import json
from njupt_qa.paths import DATA,RESULTS,read_jsonl,write_jsonl,save_json
from njupt_qa.engine import get_engine
from njupt_qa.evaluation import fact_match,csv_export

def dataset():
 docs=read_jsonl(DATA/'processed/documents.jsonl');cases=[]
 admission=next(d for d in docs if d['doc_id']=='c69898caec483a487960')
 charter=next(d for d in docs if d['doc_id']=='ac45ca3a781584acd021')
 teaching=next(d for d in docs if d['doc_id']=='3c538cd612625bbe8b7e')
 for first,pair,expected in [('物理','政治和地理',65),('物理','生物和地理',65),('物理','政治和生物',65),
                            ('物理','化学和地理',270),('物理','化学和政治',270),('物理','化学和生物',270),
                            ('历史','政治和地理',30),('历史','化学和生物',30)]:
  cases.append({'question':f'按2026年综合评价招生简章，首选{first}、再选{pair}，仅按选科条件筛选，符合专业计划数合计多少人？',
    'expected':[str(expected)],'kind':'conditional_sum','source_doc_id':admission['doc_id']})
 for kind,tuition,years in [('工科',5800,1),('工科',5800,2),('工科',5800,3),('工科',5800,5),
                             ('理科',5500,4),('文科',5200,4),('艺术',6800,4),('工科',5800,6)]:
  expected=(Decimal(tuition)*Decimal('1.1')+1500)*years
  cases.append({'question':f'按2026年招生章程，假设某{kind}类专业属于上浮10%的优势学科专业，学费加住宿费标准{years}年不变，预收额合计多少元？',
    'expected':[format(expected.normalize(),'f')],'kind':'fee_projection','source_doc_id':charter['doc_id']})
 centers={'信息与通信':9230,'电子与光学':9983,'计算机':8056,'自动化':5632,'材料与化学':7150,'物联网':2146}
 for names in [('信息与通信','电子与光学'),('信息与通信','计算机'),('电子与光学','计算机'),
               ('信息与通信','自动化'),('计算机','物联网'),('信息与通信','计算机','自动化')]:
  area=sum(centers[n] for n in names);pct=Decimal(area)/Decimal(69864)*100
  count='两个' if len(names)==2 else '三个'
  cases.append({'question':f'根据2023—2024学年教学质量报告，{"、".join(names)}{count}实验教学中心面积之和是多少平方米，占同表总面积的百分比是多少？',
    'expected':[str(area),f'{pct:.2f}'],'kind':'table_share','source_doc_id':teaching['doc_id']})
 for q in ['请依据2023—2024学年报告，利用专任教师、聘请校外教师和折合教师总数，求外聘教师折算系数，并复算生师比。',
           '按2023—2024学年报告中的折合在校生、专任教师和聘请校外教师等数据，计算生师比以及校外教师的折算系数。']:
  cases.append({'question':q,'expected':['0.5','16.38'],'kind':'teacher_ratio','source_doc_id':teaching['doc_id']})
 for i,c in enumerate(cases):c['id']=f'M{i+1:02}'
 assert len(cases)==24
 write_jsonl(DATA/'eval/calculation_24.jsonl',cases)
 save_json(DATA/'eval/calculation_protocol.json',{'description':'Same-source arithmetic variations; not included in the frozen 80-question main score. No SFT fitting or checkpoint selection uses this challenge.',
   'gold_derivation':'Transcribed official inputs and separate Decimal arithmetic; inference calculator is not called to generate gold answers.',
   'inputs':{'admissions_total':300,'history_unrestricted':30,'physics_unrestricted':65,'physics_all':270,'annual_tuition':{'工科':5800,'理科':5500,'文科':5200,'艺术':6800},'uplift':'10%','dormitory':1500,'areas':centers,'area_total':69864,'teachers':[2318,353,'2494.5'],'equivalent_students':'40863.2'}})
 return cases

def main():
 cases=dataset();engine=get_engine();folder=RESULTS/'calculation_24';folder.mkdir(exist_ok=True)
 summaries={}
 for mode,enabled in [('rag',False),('rag',True),('sft_rag',False),('sft_rag',True)]:
  label=mode+('_calculator' if enabled else '');out=folder/(label+'.jsonl');rows=read_jsonl(out);done={r['id'] for r in rows}
  for case in cases:
   if case['id'] in done:continue
   result=engine.answer(case['question'],mode,calculator=enabled,guardrails=False)
   passed=all(fact_match(x,result['answer']) for x in case['expected'])
   rows.append({'id':case['id'],'question':case['question'],'expected':case['expected'],'kind':case['kind'],'correct':passed,**result})
   write_jsonl(out,rows);print(label,case['id'],passed,flush=True)
  csv_export(folder/(label+'.csv'),rows)
  summaries[label]={'cases':len(rows),'correct':sum(r['correct'] for r in rows),'numeric_rule_accuracy':sum(r['correct'] for r in rows)/len(rows),
    'verified_calculator_cases':sum(bool(r['calculation']) for r in rows),'calculator_enabled':enabled}
  save_json(folder/'summary.json',summaries)
 print('ARITHMETIC CHALLENGE COMPLETE',flush=True)

if __name__=='__main__':main()
