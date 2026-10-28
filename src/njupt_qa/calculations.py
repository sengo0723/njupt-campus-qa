"""Deterministic operations over retrieved official source data, never question/answer lookup."""
from __future__ import annotations
from decimal import Decimal
import re
from .paths import digest

def compact(s):return re.sub(r'\s+','',s).replace('％','%')
def number(s):return Decimal(re.sub(r'[^\d.]','',s))
def fmt(v,places=None):
    x=Decimal(str(v))
    return f'{x:.{places}f}' if places is not None else format(x.normalize(),'f')

def evidence(doc,page,text):
    return {'id':'','doc_id':doc['doc_id'],'chunk_id':digest(doc['doc_id']+str(page)+text),'title':doc['title'],
       'topic':doc['topic'],'url':doc['url'],'published_at':doc['published_at'],'page':page,'content':text,
       'kind':'calculation_source','raw_file':doc['raw_file']}

def calculate(question,documents,citations):
    q=compact(question);available={c['doc_id'] for c in citations}
    docs=[d for d in documents if d['doc_id'] in available]
    years=re.findall(r'20\d{2}',q)
    if years:docs=[d for d in docs if not d['years'] or years[0] in d['years']]
    if re.search('选科|首选|再选',q) and re.search('计划|招生',q):
        first=re.search(r'首选(?:科目)?(?:为|是)?[“"：:]?(物理|历史)',q)
        if first:
            chosen={s for s in ['政治','地理','化学','生物'] if s in q}
            chosen={s for s in chosen if not re.search(r'(?:不选|未选|没有选|没选)(?:修)?'+s,q)}
            if not chosen and '不限' not in q:return None
            for d in docs:
                for t in d['tables']:
                    rows=t['rows']; header=next((i for i,r in enumerate(rows) if any('首选' in c for c in r) and any('计划' in c for c in r)),None)
                    if header is None:continue
                    labels=rows[header]; fi=next(i for i,x in enumerate(labels) if '首选' in x); si=next(i for i,x in enumerate(labels) if '再选' in x)
                    pi=next(i for i,x in enumerate(labels) if '专业' in x); ni=next(i for i,x in enumerate(labels) if '计划' in x)
                    selected=[]
                    for row in rows[header+1:]:
                        if len(row)<=max(fi,si,pi,ni) or not re.fullmatch(r'\d+',compact(row[ni])):continue
                        if compact(row[fi])!=first[1]:continue
                        req=compact(row[si]); subjects={s for s in ['政治','地理','化学','生物'] if s in req}
                        if req=='不限' or (subjects and subjects.issubset(chosen)):
                            selected.append({'专业':row[pi],'计划':int(row[ni]),'选科':row[fi]+'+'+row[si]})
                    if not selected:continue
                    total=sum(x['计划'] for x in selected);form='+'.join(str(x['计划']) for x in selected)+'='+str(total)
                    text='\n'.join(' | '.join(r) for r in rows)
                    return {'kind':'conditional_sum','result':str(total)+'人','formula':form,'operands':selected,
                      'explanation':'仅按选科条件筛选，不代表满足综合评价的全部报名条件；招生计划以主管部门最终核定为准。',
                      'sources':[evidence(d,t['page'],text)],'details':'\n'.join(f'- {x["专业"]}：{x["计划"]}人' for x in selected)}
    if '学费' in q and '住宿' in q:
        category=next((x for x in ['工科','理科','文科','艺术'] if x in q),None)
        if category:
            for d in docs:
                text=compact(d['text'])
                fee=re.search(category+r'类专业每生每年(\d+)元',text)
                dorm=re.search(r'住宿费(?:暂)?按(\d+)元',text)
                uplift=re.search(r'上浮(\d+(?:\.\d+)?)%',text)
                if not fee or not dorm:continue
                n=re.search(r'([一二三四五六七八九十\d]+)年(?:不变|合计|预收|费用|标准)?',q)
                year_match=re.search(r'(?<![0-9])([一二三四五六七八九十]|[1-9])年(?:不变|预收|费用|学制|合计)?',q)
                years_count={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}.get(year_match[1],None) if year_match else None
                if years_count is None and year_match:years_count=int(year_match[1])
                years_count=years_count or 1
                pct=Decimal(uplift[1])/100 if uplift and re.search('优势|上浮',q) else Decimal(0)
                base=Decimal(fee[1]);accommodation=Decimal(dorm[1]);raised=base*(1+pct);total=(raised+accommodation)*years_count
                form=f'({fmt(base)}×(1+{fmt(pct*100)}%)+{fmt(accommodation)})×{years_count}={fmt(total)}元'
                pos=text.find(category+'类专业');snippet=text[max(0,pos-90):text.find('第六章',pos) if '第六章' in text[pos:] else pos+900]
                return {'kind':'fee_projection','result':fmt(total)+'元','formula':form,
                   'operands':{'annual_tuition':fmt(base),'uplift':fmt(pct),'annual_dormitory':fmt(accommodation),'years':years_count},
                   'explanation':f'上浮后学费为每年{fmt(raised)}元。此结果按题设假定标准{years_count}年不变计算，为预收额；实际费用据实结算。',
                   'details':'','sources':[evidence(d,None,snippet)]}
    if '生师比' in q or '折算系数' in q:
        for d in docs:
            for p in d['pages'] or [{'page':None,'text':d['text']}]:
                text=compact(p['text'])
                match=re.search(r'专任教师(\d+)人[，,]聘请校外教师(\d+)人[，,]折合教师总数为(\d+(?:\.\d+)?)',text)
                students=re.search(r'折合在校生人数(?:为)?(\d+(?:\.\d+)?)',text)
                if not match or not students:continue
                regular,external,total=map(Decimal,match.groups());stu=Decimal(students[1])
                if not external or not total:continue
                coeff=(total-regular)/external;ratio=stu/total
                return {'kind':'teacher_ratio','result':f'外聘教师折算系数 {fmt(coeff)}；生师比 {fmt(ratio,2)}:1',
                   'formula':f'({fmt(total)}−{fmt(regular)})÷{fmt(external)}={fmt(coeff)}；{fmt(stu)}÷{fmt(total)}={fmt(ratio,2)}',
                   'operands':{'regular':fmt(regular),'external':fmt(external),'equivalent_teachers':fmt(total),'equivalent_students':fmt(stu)},
                   'explanation':'使用报告给出的折合在校生人数计算；生师比保留两位小数。',
                   'details':'','sources':[evidence(d,p['page'],p['text'])]}
    if '面积' in q and re.search('合计|之和|占|百分比|比例',q):
        for d in docs:
            for t in d['tables']:
                joined=' '.join(' '.join(r) for r in t['rows'])
                if not ('面积' in joined or '平方' in joined):continue
                rows=[];total=None
                for row in t['rows']:
                    cells=[compact(x) for x in row]
                    if any('合计' in x for x in cells):
                        nums=[x for x in cells if re.fullmatch(r'\d+(?:\.\d+)?',x)]
                        if nums:total=Decimal(nums[-1])
                        continue
                    name=next((x for x in cells if re.search('实验教学中心|实验室|工程实验教学部',x)),None)
                    if not name:continue
                    nums=[x for x in cells if re.fullmatch(r'\d+(?:\.\d+)?',x)]
                    if not nums:continue
                    stem=re.sub('工程|实验教学中心|实验室|实验教学部','',name)
                    if stem and stem in q:rows.append({'name':name,'area':Decimal(nums[-1])})
                if not rows or not total:continue
                mentioned_count=re.search(r'([一二三四五六七八九十]|[1-9])(?:个)?(?:实验教学)?中心',q)
                if mentioned_count:
                    count={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}.get(mentioned_count[1])
                    if count is None:count=int(mentioned_count[1])
                    if len(rows)!=count:continue
                if '之和' in q and len(rows)<2:continue
                area=sum(x['area'] for x in rows);pct=area/total*100
                form='+'.join(fmt(x['area']) for x in rows)+f'={fmt(area)}平方米；{fmt(area)}÷{fmt(total)}×100%={fmt(pct,2)}%'
                return {'kind':'table_share','result':f'面积合计 {fmt(area)}平方米，占表内总面积约 {fmt(pct,2)}%',
                  'formula':form,'operands':[{**x,'area':fmt(x['area'])} for x in rows]+[{'total':fmt(total)}],
                  'explanation':'分母采用同一张表中的合计面积，百分比保留两位小数。',
                  'details':'\n'.join(f'- {x["name"]}：{fmt(x["area"])}平方米' for x in rows),
                  'sources':[evidence(d,t['page'],'\n'.join(' | '.join(r) for r in t['rows']))]}
    return None

def render_calculation(calc,citation_ids):
    cite=''.join(f'[[{x}]]' for x in citation_ids)
    return f'**{calc["result"]}。** {cite}\n\n'+(calc['details']+'\n\n' if calc.get('details') else '')+f'计算：{calc["formula"]}。\n\n{calc["explanation"]}\n\n*数值由本地计算器依据引用资料核验。*'
