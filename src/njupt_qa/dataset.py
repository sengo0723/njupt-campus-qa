"""Evidence-backed instruction construction and disjoint fact-group evaluation datasets.

Sources are public official snapshots. Template-generated samples are declared as such;
synthetic arithmetic contains generic labels and is never put into the retrieval index.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from decimal import Decimal
from difflib import SequenceMatcher
import json, random, re
from .paths import DATA, MODELS, digest, read_jsonl, write_jsonl, save_json

SYSTEM='你是南邮校园知识助手。用简洁中文回答。提供资料时仅依据资料回答，并用[[C1]]等标签标明证据。资料不足时明确说明，不编造。年度政策以指定年份资料为准；未提供资料时仅回答稳定的基础知识。'
def norm(s): return re.sub(r'\s+','',s)

# question, reference answer, independently checked fact substrings, document title pattern
FACTS={
'学校概况':[
('南京邮电大学的前身创办于哪一年？','学校前身诞生于1942年。',['1942'],'学校简介'),
('南京邮电大学在1958年改建为本科高校时叫什么？','南京邮电学院。',['南京邮电学院'],'学校简介'),
('南京邮电学院是哪一年更名为南京邮电大学的？','2005年。',['2005'],'学校简介'),
('南邮的校训是什么？','厚德、弘毅、求是、笃行。',['厚德','弘毅','求是','笃行'],'学校章程'),
('南邮精神用哪八个字概括？','信达天下，自强不息。',['信达天下','自强不息'],'学校章程'),
('南京邮电大学的校风是什么？','勤奋、求实、进取、创新。',['勤奋','求实','进取','创新'],'学校章程'),
('南京邮电大学的英文名称是什么？','Nanjing University of Posts and Telecommunications。',['Nanjing','Posts','Telecommunications'],'学校章程'),
('南京邮电大学的英文缩写是什么？','NJUPT。',['NJUPT'],'学校章程'),
('按照学校章程，南邮的校庆日是哪一天？','4月20日。',['4月20日'],'学校章程'),
('南邮校旗的长高比是多少？','3:2。',['3:2'],'学校章程'),
('学校章程规定的南邮标识色是什么颜色？','蓝色，色值C100 M90。',['蓝色','C100','M90'],'学校章程'),
('南京邮电大学的校歌叫什么？','《信达天下》。',['信达天下'],'学校章程'),
('南京邮电大学校歌由谁作词？','秦华礼、杨德祥。',['秦华礼','杨德祥'],'学校章程'),
('南京邮电大学校歌由谁作曲？','李恒。',['李恒'],'学校章程'),
('南邮校标整体是什么造型？','盾牌造型。',['盾牌'],'学校章程'),
('南邮校标综合变化使用了哪两个汉字？','“邮”和“电”。',['邮','电'],'学校章程'),
('学校章程公布的南邮官方网站域名是什么？','www.njupt.edu.cn。',['www.njupt.edu.cn'],'学校章程'),
('学校章程由谁负责解释？','中国共产党南京邮电大学委员会。',['中国共产党南京邮电大学委员会'],'学校章程')],
'招生':[
('2026年南邮综合评价招生面向哪个省？','江苏省。',['江苏省'],'2026年综合评价招生简章'),
('2026年南邮综合评价招生计划总数是多少？','300人，最终以省级招生主管部门核定为准。',['300'],'2026年综合评价招生简章'),
('2026年南邮综合评价报名截止日期是什么时候？','5月10日。',['5月10日'],'2026年综合评价招生简章'),
('2026年综合评价初审通过名单最迟何时公示？','5月25日前。',['5月25日'],'2026年综合评价招生简章'),
('2026年综合评价测试确认和缴费应在何时完成？','5月31日前。',['5月31日'],'2026年综合评价招生简章'),
('2026年南邮综合评价招生测试费是多少？','60元/生。',['60'],'2026年综合评价招生简章'),
('2026年南邮综合评价报名材料要邮寄纸质版吗？','无需邮寄纸质材料，材料通过报名系统上传。',['无需邮寄纸质材料'],'2026年综合评价招生简章'),
('2026年南邮综合评价由哪个常设机构组织实施？','本科生院。',['本科生院'],'2026年综合评价招生简章'),
('2026年招生章程规定南邮是什么性质的高校？','省属公办高校。',['省属公办高校'],'2026年普通本科招生章程'),
('2026年章程中南邮的院校标识码是多少？','4132010293。',['4132010293'],'2026年普通本科招生章程'),
('2026年本科招生的预留计划不超过年度计划的多少？','1%。',['1%'],'2026年普通本科招生章程'),
('2026年招生章程规定调档比例原则上控制在多少以内？','105%以内。',['105%'],'2026年普通本科招生章程'),
('2026年南邮安排进档考生专业遵循什么原则？','分数优先。',['分数优先'],'2026年普通本科招生章程'),
('2026年招生章程规定各专业志愿之间有级差吗？','不设级差。',['不设级差'],'2026年普通本科招生章程'),
('2026年报考南邮英语专业要求什么应试语种？','英语。',['英语'],'2026年普通本科招生章程'),
('2026年章程规定南邮公共外语课程只开设什么课程？','英语课程。',['英语课程'],'2026年普通本科招生章程'),
('2026年本科招生咨询电话有哪些？','025-85866668、025-85866669。',['85866668','85866669'],'2026年普通本科招生章程'),
('2026年招生章程规定学校是否委托中介招生？','不委托任何中介机构或个人进行招生录取工作。',['不委托任何中介机构或个人'],'2026年普通本科招生章程')],
'图书馆服务':[
('本科生在南邮图书馆每证最多借多少册？','30册。',['30册'],'借阅规则'),
('本科生借中文图书的借期是多少天？','30天。',['30天'],'借阅规则'),
('研究生在南邮图书馆借书的借期是多少？','60天。',['60天'],'借阅规则'),
('教职工借书的借期是多少天？','90天。',['90天'],'借阅规则'),
('本科生可借几册外文原版图书？','1册。',['本科生每证限借1册'],'借阅规则'),
('南邮图书馆密集书库每证最多借几册？','5册。',['5册'],'借阅规则'),
('南邮图书馆随书光盘的借期是多少？','1天。',['1天'],'借阅规则'),
('南邮图书馆当天归还的书，本人当天能再借吗？','不能。',['本人当天不能再借'],'借阅规则'),
('图书到期前无人预约时可以续借几次、续多久？','可续借一次，续借期30天；随书光盘除外。',['一次','30天'],'借阅规则'),
('南邮图书馆已经过期的图书还能续借吗？','不能，过期图书不接受续借。',['过期图书不接受续借'],'借阅规则'),
('南邮图书馆可以同时预约多少册图书？','3册。',['预约3册'],'借阅规则'),
('预约图书到馆后会保留几天？','3天。',['保留3天'],'借阅规则'),
('密集书库委托借阅成功后多久可以取书？','法定工作日24小时之后，到仙林校区图书馆借还总台。',['24小时'],'借阅规则'),
('委托到书成功后多少天内不领取视为放弃？','7天内。',['7天内'],'借阅规则'),
('累计三次不领取委托图书会暂停多久权限？','暂停180天。',['180天'],'借阅规则'),
('在另一校区借的图书可以在本校区还吗？','可以，在任一校区图书馆服务总台归还。',['任一校区'],'借阅规则'),
('普通中文图书超期每册每天的违约金是多少？','0.10元。',['0.10元'],'借阅规则'),
('办图书馆离校手续需要携带哪些证件材料？','校园一卡通和离校单。',['校园一卡通和离校单'],'借阅规则')],
'本科教学':[
('2023—2024学年报告中的专任教师有多少人？','2318人。',['2318'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中聘请的校外教师有多少人？','353人。',['353'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中的折合教师总数是多少？','2494.5人。',['2494.5'],'2023—2024学年本科教学质量报告'),
('2023—2024学年折合在校生人数是多少？','40863.2人。',['40863.2'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告公布的全校生师比是多少？','16.38:1。',['16.38'],'2023—2024学年本科教学质量报告'),
('2023—2024学年普通本科生人数是多少？','24110人。',['24110'],'2023—2024学年本科教学质量报告'),
('2023—2024学年全日制在校生总人数是多少？','33432人。',['33432'],'2023—2024学年本科教学质量报告'),
('2023—2024学年全日制硕士生人数是多少？','7881人。',['7881'],'2023—2024学年本科教学质量报告'),
('2023—2024学年全日制博士生人数是多少？','738人。',['738'],'2023—2024学年本科教学质量报告'),
('2023—2024学年留学生有多少人？','701人。',['701'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中专任教师具有博士学位的有多少人？','1746人。',['1746'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中专任教师正高级职称人数是多少？','397人。',['397'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中专任教师副高级职称人数是多少？','887人。',['887'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中全校本科专业总数是多少？','61个。',['61'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告记载的本科教学主校区是哪个？','仙林校区。',['仙林校区'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中实验教学中心总面积是多少？','69864平方米。',['69864'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中信息与通信工程实验教学中心面积是多少？','9230平方米。',['9230'],'2023—2024学年本科教学质量报告'),
('2023—2024学年报告中计算机实验教学中心面积是多少？','8056平方米。',['8056'],'2023—2024学年本科教学质量报告')],
'学生事务':[
('按2019年公开流程汇编，本科学生证补办先在哪个平台申请？','登录智慧校园，点击办事大厅中的本科生学生证补办。',['智慧校园'],'工作流程汇编'),
('按2019年流程汇编，学生证补办线上申请由谁审核？','辅导员网上审核。',['辅导员网上审核'],'工作流程汇编'),
('按2019年流程汇编，补办学生证需携带哪些材料及工本费？','身份证、一寸证件照，工本费10元。',['身份证','一寸证件照','10元'],'工作流程汇编'),
('按2019年流程汇编，学生证充磁需要带什么？','学生证、身份证。',['学生证','身份证'],'工作流程汇编'),
('按2019年流程汇编，学生证乘车区间最多修改几次？','2次。',['最多修改2次'],'工作流程汇编'),
('按2019年流程汇编，仙林校区学生证业务在哪个窗口办理？','学生事务中心服务大厅3号学生助理窗口。',['3号学生助理窗口'],'工作流程汇编'),
('按2019年流程汇编，三牌楼校区学生证业务在哪办理？','学生社团活动中心201。',['学生社团活动中心201'],'工作流程汇编'),
('按2019年流程汇编，学生证业务咨询电话是什么？','025-85866792。',['85866792'],'工作流程汇编'),
('按2019年流程汇编，学生证业务意见反馈邮箱是什么？','xssw@njupt.edu.cn。',['xssw@njupt.edu.cn'],'工作流程汇编'),
('按2019年流程汇编，校外住宿申请首先需要谁同意？','先征得家长同意，再向学院提交申请。',['家长同意'],'工作流程汇编'),
('按2019年流程汇编，申请校外住宿应填写什么表？','《校外住宿审批表》。',['校外住宿审批表'],'工作流程汇编'),
('按2019年流程汇编，申请调宿应填写什么表？','《调整宿舍登记表》。',['调整宿舍登记表'],'工作流程汇编'),
('按2019年流程汇编，申请入住需要填写什么表？','《宿舍入住申请表》。',['宿舍入住申请表'],'工作流程汇编'),
('按2019年流程汇编，调宿交还原宿舍钥匙后领取什么？','领取新宿舍钥匙。',['新宿舍钥匙'],'工作流程汇编'),
('按2019年流程汇编，原宿舍钥匙丢失时需要谁开证明？','辅导员。',['辅导员开具钥匙丢失'],'工作流程汇编'),
('按2019年流程汇编，毕业退宿时宿管站在哪份材料上盖章？','离校通知单。',['离校通知单上盖章'],'工作流程汇编'),
('按2019年流程汇编，参加校内勤工助学先向哪里提出申请？','向二级学院提出申请。',['二级学院提出申请'],'工作流程汇编'),
('按2019年流程汇编，勤工助学学生正式上岗前应完成什么？','学校学生处组织的上岗培训。',['上岗学生培训'],'工作流程汇编')]
}
UNKNOWN={
'学校概况':['今天南邮仙林校区实时在校人数是多少？','请提供校长私人手机号码。'],
'本科教学':['请查我的个人期末考试成绩。','我明天的个人课表在哪里上课？'],
'招生':['南京邮电大学2028年综合评价招生计划是多少？','保证我2027年考多少分一定被录取？'],
'图书馆服务':['现在仙林图书馆实时还有多少空座？','帮我查询学号123456的借阅历史。'],
'学生事务':['查询我本人的助学金是否到账。','告诉我某位同学的宿舍门牌和个人电话。']}

def create_eval(docs):
    cases=[]; reserved=[]
    for topic in FACTS:
        for j,(q,answer,keys,title) in enumerate(FACTS[topic]):
            ds=[d for d in docs if title in d['title']]
            if not ds: raise ValueError('Missing evidence document: '+title)
            d=ds[0]; text=norm(d['text'])
            if not all(norm(k) in text for k in keys):
                raise ValueError('Reference not supported: '+q+' '+repr(keys))
            pages=d['pages'] or [{'page':None,'text':d['text']}]
            page=max(pages,key=lambda p:sum(norm(k) in norm(p['text']) for k in keys))
            # Source windows are annotation only, never inserted into the RAG corpus.
            pn=norm(page['text']); pos=max(0,pn.find(norm(keys[0])))
            evidence=pn[max(0,pos-180):pos+max(180,len(norm(keys[0])))]
            case={'id':f'{topic}-{j+1:02}','topic':topic,'question':q,'reference_answer':answer,'required_facts':keys,
                  'gold_doc_ids':[d['doc_id']],'gold_pages':[page['page']] if page['page'] else [],
                  'evidence':evidence,'fact_group_id':digest(topic+norm(answer)+norm(q)),
                  'answerable':True,'type':'fact','split':'dev' if j in [0,6,12] else 'test'}
            cases.append(case);reserved.append({'doc_id':d['doc_id'],'page':page['page'],'evidence':evidence,'keys':keys,'question':q})
        for j,q in enumerate(UNKNOWN[topic]):
            cases.append({'id':f'{topic}-U{j+1}','topic':topic,'question':q,'reference_answer':'当前公开资料不足以回答，或需要本人登录官方系统查询。',
              'required_facts':[],'gold_doc_ids':[],'gold_pages':[],'evidence':'','fact_group_id':digest(q),
              'answerable':False,'type':'unanswerable','split':'dev' if j==0 else 'test'})
    assert len(cases)==100 and sum(c['split']=='dev' for c in cases)==20
    write_jsonl(DATA/'eval/campus_100.jsonl',cases)
    for split in ['dev','test']: write_jsonl(DATA/f'eval/{split}.jsonl',[c for c in cases if c['split']==split])
    write_jsonl(DATA/'eval/heldout_facts.jsonl',reserved)
    acceptance=[
      {'id':'A1','question':'2026年综合评价中，首选物理、再选政治和地理的考生，仅按选科条件筛选，可报专业的计划合计多少人？','expected':['65'],'unit':'人'},
      {'id':'A2','question':'按2026年章程，工科类优势学科专业的学费上浮后，加住宿费，假设标准四年不变，预收额合计多少？','expected':['31520'],'unit':'元'},
      {'id':'A3','question':'根据2023—2024学年报告的专任、外聘及折合教师人数，反推外聘折算系数，再复算生师比。','expected':['0.5','16.38'],'unit':''},
      {'id':'A4','question':'根据信息与通信、电子与光学、计算机三个实验教学中心的面积，计算其面积之和占2023—2024学年报告表内总面积的百分比。','expected':['27269','39.03'],'unit':'平方米/%'}]
    write_jsonl(DATA/'eval/acceptance.jsonl',acceptance)
    return cases,reserved

def source_units(docs,reserved):
    units=[];seen=set()
    hold=defaultdict(list)
    for x in reserved: hold[x['doc_id']].append(x)
    for d in docs:
        # Public acceptance documents stay entirely out of SFT; the broad handbook
        # may duplicate the historical statistics and is withheld as well.
        if d['doc_id'] in ['3c538cd612625bbe8b7e','c69898caec483a487960','ac45ca3a781584acd021','29429770992e86830a0a','580982b06be633fd91c1'] or '学生手册' in d['title']: continue
        for p in d['pages'] or [{'page':None,'text':d['text']}]:
            clean=re.sub(r'\s+',' ',p['text'])
            for s in re.split(r'(?<=[。！？；])\s*|\n+',clean):
                s=s.strip()
                if not 30<=len(s)<=230:continue
                if re.search(r'目录|\.{3}|\|.*\|',s):continue
                if any(norm(s) in x['evidence'] or SequenceMatcher(None,norm(s),x['evidence']).ratio()>.5 for x in reserved):continue
                canonical=re.sub(r'20\d{2}','YEAR',norm(s))
                fg=digest(canonical)
                if fg in seen:continue
                seen.add(fg)
                units.append({'fact_group_id':fg,'source_doc_id':d['doc_id'],'source_url':d['url'],'page':p['page'],
                    'title':d['title'],'topic':d['topic'],'text':s,'synthetic':False})
    return units

def build_sft(docs,reserved,max_length=1536):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODELS/'qwen3-base-hf',local_files_only=True)
    rng=random.Random(42);units=source_units(docs,reserved);rng.shuffle(units)
    # Assign source fact groups before paraphrasing. Closed-book charter facts
    # are relatively stable institutional knowledge; time-sensitive material is contextual.
    stable=[u for u in units if re.search('学校章程|南邮精神|南邮校史|学校简介|本馆简介',u['title']) and not re.search(r'目前|现有|在校|截至|202[3-9]|学费|收费|人数',u['text'])]
    other=[u for u in units if u not in stable]
    if len(stable)<84: raise ValueError(f'Need >=84 distinct stable fact groups, got {len(stable)}')
    if len(other)<230: raise ValueError(f'Need >=230 contextual fact groups, got {len(other)}')
    partitions={s:[] for s in ['train','valid','test']};metadata=[];lengths=[]
    def append(split,category,group,q,a,context=None,history=None,extra_groups=None):
        messages=[{'role':'system','content':SYSTEM}]
        if history:messages.extend(history)
        prompt=(f'参考资料：\n[[C1]] {context}\n\n问题：{q}' if context else q)
        messages += [{'role':'user','content':prompt},{'role':'assistant','content':a}]
        toks=tokenizer.apply_chat_template(messages,tokenize=True,enable_thinking=False)
        if isinstance(toks,dict) or hasattr(toks,'keys'):toks=toks['input_ids']
        if len(toks)>max_length:raise ValueError(f'Overlong sample ({len(toks)}) in {category}; no silent truncation permitted')
        idx=len(partitions[split]);partitions[split].append({'messages':messages});lengths.append(len(toks))
        metadata.append({'split':split,'row':idx,'category':category,'fact_group_id':group['fact_group_id'],
                         'source_doc_id':group.get('source_doc_id'),'source_url':group.get('source_url'),
                         'page':group.get('page'),'synthetic':group.get('synthetic',False),'tokens':len(toks),
                         'extra_fact_groups':[x['fact_group_id'] for x in (extra_groups or [])],
                         'extra_source_doc_ids':[x['source_doc_id'] for x in (extra_groups or [])],
                         'multi_turn':bool(history),'multi_evidence':bool(extra_groups),
                         'construction':'source extraction + controlled templates + deterministic validation','sample_hash':digest(json.dumps(messages,ensure_ascii=False))})
    stable_offsets=0;context_offsets=0
    quotas={'train':{'stable':200,'grounded':500,'refusal':200,'temporal':150,'calculation':150},
            'valid':{'stable':25,'grounded':63,'refusal':25,'temporal':18,'calculation':19},
            'test':{'stable':25,'grounded':63,'refusal':25,'temporal':18,'calculation':19}}
    for split,quota in quotas.items():
        n=quota['stable'];pool=stable[stable_offsets:stable_offsets+math_ceil(n/3)];stable_offsets+=len(pool)
        for j in range(n):
            u=pool[j//3];hint=re.sub(r'^第[一二三四五六七八九十百]+条\s*','',u['text'])[:12]
            q=[f'依据南邮的《{u["title"]}》，关于“{hint}”有哪些说明？',f'请解释南邮基础资料中“{hint}”这一点。',f'南邮的《{u["title"]}》如何表述“{hint}”相关内容？'][j%3]
            append(split,'stable',u,q,u['text'])
        n=quota['grounded'];primary_count=math_ceil(n/3)
        assigned=other[context_offsets:context_offsets+2*primary_count];context_offsets+=len(assigned)
        pool=assigned[:primary_count];secondary=assigned[primary_count:]
        for j in range(n):
            u=pool[j//3];hint=u['text'][:10]
            q=[f'根据所给资料，关于“{hint}”有哪些具体规定？',f'请依据片段概括“{hint}”的要点。',f'“{hint}”相关内容应如何理解？请给出证据。'][j%3]
            if j%3==2:
                v=secondary[j//3]
                q=f'请分别概括两份参考资料中“{hint}”和“{v["text"][:10]}”的具体内容，并区分引用。'
                append(split,'grounded',u,q,u['text']+'[[C1]]\n\n'+v['text']+'[[C2]]',
                    f'《{u["title"]}》：{u["text"]}\n\n[[C2]] 《{v["title"]}》：{v["text"]}',extra_groups=[v])
            else:append(split,'grounded',u,q,u['text']+'[[C1]]',f'《{u["title"]}》：{u["text"]}')
        # Unique topic/intent pairs are split before wording changes, and do not
        # reuse the campus test questions. Each is a negative instruction fact group.
        domains=['课程补考资格','教材领取','校内报修','实验室预约','奖学金评审','转专业手续','学籍证明','图书荐购','校园卡余额','宿舍调换','活动报名','公寓维修','学费缴费','网络故障','体育选课','毕业资格','教室借用','档案转递','校园交通','电子资源访问']
        contexts=['未给出任何资料','仅提供与问题无关的校史片段','只有过期通知，缺少当年政策','缺少办理对象和年份','需要本人登录后查询']
        all_negative=[(a,b) for a in domains for b in contexts]
        offset={'train':0,'valid':70,'test':80}[split]
        for j in range(quota['refusal']):
            a,b=all_negative[offset+j//3];g={'fact_group_id':digest('negative:'+a+b),'synthetic':True}
            q=[f'{b}。请给出我关于{a}的具体办理结论。',f'我想确认{a}的结果，但{b}，你能直接给确定答案吗？',f'在{b}的情况下，能否断言我的{a}一定成功？'][j%3]
            answer=('这需要本人登录学校官方系统查询。当前公开资料不能确认个人结果，请勿提供账户密码。' if '登录' in b else '当前资料不足以给出可靠结论。请补充适用年份、办理对象和对应官方通知，再据此核对。')
            append(split,'refusal',g,q,answer)
        for j in range(quota['temporal']):
            group_index={'train':0,'valid':60,'test':70}[split]+j//3
            year=2035+group_index; old=year-1;item=['甲项服务','乙项申请','丙项培训','丁项预约'][group_index%4]
            old_date=f'{1+group_index%11}月{5+group_index%20}日'
            new_date=f'{1+(group_index*3+2)%11}月{5+(group_index*7+3)%20}日'
            context=f'以下为合成时效练习，不是学校真实政策。{old}年{item}截止日期为{old_date}；{year}年更新通知规定截止日期为{new_date}，更新通知替代上一年度安排。'
            q=[f'按{year}年的通知，{item}应在何时完成？',f'去年和今年的{item}日期不同，请问{year}年以哪个日期为准？',f'请区分这两份资料：{year}年{item}的截止日期是什么？'][j%3]
            g={'fact_group_id':digest('temporal:'+str(group_index)),'synthetic':True}
            history=None
            if j%3==2:
                history=[{'role':'user','content':f'我问的是{old}年{item}的截止日期。'},
                         {'role':'assistant','content':f'{old}年是{old_date}。'}]
                q='那下一年的截止日期呢？请根据更新通知说明。'
            append(split,'temporal',g,q,f'{year}年应以{new_date}为准。{old_date}属于{old}年的旧安排，不能套用到{year}年。[[C1]]',context,history)
        for j in range(quota['calculation']):
            group_index={'train':0,'valid':70,'test':80}[split]+j//3
            a=17+group_index*3;b=31+group_index*2;c=43+group_index;total=a+b+c
            context=f'合成表格练习，不代表学校数据。\n项目 | 面积（平方米）\n项目甲 | {a}\n项目乙 | {b}\n项目丙 | {c}\n总计 | {total}'
            variant=j%3;g={'fact_group_id':digest('calc:'+str(group_index)),'synthetic':True}
            if variant==0:q='项目甲和项目乙的面积合计多少？';answer=f'{a}+{b}={a+b}平方米。[[C1]]'
            elif variant==1:q='项目甲和项目乙面积之和占总面积多少？';answer=f'({a}+{b})/{total}×100%={((a+b)/total*100):.2f}%。[[C1]]'
            else:q='从总面积减去项目丙，剩余面积是多少？';answer=f'{total}-{c}={a+b}平方米。[[C1]]'
            append(split,'calculation',g,q,answer,context)
    all_groups={s:{g for m in metadata if m['split']==s for g in [m['fact_group_id']]+m['extra_fact_groups']} for s in partitions}
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        if all_groups[a]&all_groups[b]:raise ValueError('Fact-group leakage: '+a+'/'+b)
    hashes=[m['sample_hash'] for m in metadata]
    if len(hashes)!=len(set(hashes)):raise ValueError('Duplicate instruction samples')
    for split,rows in partitions.items():
        order=list(range(len(rows)));rng.shuffle(order)
        shuffled=[rows[i] for i in order]
        remap={old:new for new,old in enumerate(order)}
        for m in metadata:
            if m['split']==split:m['row']=remap[m['row']]
        write_jsonl(DATA/f'sft/{split}.jsonl',shuffled)
    write_jsonl(DATA/'sft/provenance.jsonl',metadata)
    save_json(DATA/'sft/audit.json',{'counts':{s:len(v) for s,v in partitions.items()},'category_counts':{s:dict(Counter(m['category'] for m in metadata if m['split']==s)) for s in partitions},
       'fact_group_overlap':0,'exact_duplicate_samples':0,'max_tokens':max(lengths),'mean_tokens':sum(lengths)/len(lengths),
       'source_fact_candidates':len(units),'stable_fact_candidates':len(stable),'synthetic_samples':sum(m['synthetic'] for m in metadata),
       'multi_turn_samples':sum(m['multi_turn'] for m in metadata),'multi_evidence_samples':sum(m['multi_evidence'] for m in metadata),
       'seed':42,'annotation':'Programmatically grounded, template-constructed; not human-labelled. Synthetic arithmetic/negative/temporal exercises are explicitly marked and never indexed.',
       'acceptance_documents_excluded_from_sft':True})
    print('SFT datasets ready:',{s:len(v) for s,v in partitions.items()},'max_tokens',max(lengths),flush=True)

def math_ceil(x):return int(-(-x//1))
def build_all(max_length=1536):
    docs=read_jsonl(DATA/'processed/documents.jsonl');cases,reserved=create_eval(docs)
    build_sft(docs,reserved,max_length)

if __name__=='__main__':build_all()
