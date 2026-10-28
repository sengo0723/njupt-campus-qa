"""Task-aligned SFT experiment: short answers, noisy evidence, natural questions.

Original data stays untouched. Synthetic documents are explicitly fictional and
never added to the real RAG corpus. The test80 and four acceptance answers are
not consulted by this builder. Official facts use existing training groups only.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import random
import re
import shutil

from njupt_qa.dataset import SYSTEM
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/sft_aligned'


def norm(text):
    return re.sub(r'\s+', '', text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--long', action='store_true', help='Match production evidence length using training-only distractors')
    parser.add_argument('--rich', action='store_true', help='Add 70 curated official facts, replacing synthetic reading groups')
    args = parser.parse_args()
    out = ROOT / ('data/sft_aligned' + ('_rich' if args.rich else '') + ('_long' if args.long else ''))
    rng = random.Random(20261026)
    padding_rng = random.Random(9017)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-base-hf', local_files_only=True)
    units = {u['unit_id']: u for u in map(json.loads, (ROOT / 'data/tuning/source_candidates.jsonl').read_text().splitlines())}
    annotations = list(csv.DictReader((ROOT / 'data/tuning/train_annotations.tsv').open(), delimiter='\t'))
    extra_annotations = list(csv.DictReader((ROOT / 'data/tuning/train_extra_annotations.tsv').open(), delimiter='\t')) if args.rich else []
    annotations += extra_annotations
    pool = [units[a['unit_id']] for a in annotations]
    output, provenance = [], []

    def block(title, text, date='未标注', page=None, groups=None):
        if args.long and text.startswith('合成'):
            for _ in range(3):
                text += '\n\n附带服务信息：'+f'资料咨询室为{padding_rng.randint(100,999)}室，每周星期{padding_rng.choice("一二三四五")}接待；电子归档编号为X{padding_rng.randint(1000,9999)}。'+\
                        f'本服务点设有{padding_rng.randint(10,89)}个临时存放柜，工作人员分为{padding_rng.randint(3,9)}组，窗口宣传册有{padding_rng.randint(12,99)}页。'+\
                        '申请人应核对办理对象、事项和年度，遵守现场秩序，按要求提交真实完整的材料。以上附带信息不改变正文列出的办理标准。'
        return {'title': title, 'text': text, 'date': date, 'page': page, 'groups': groups or []}

    def official(unit):
        extra = [u for u in pool if u['source_doc_id'] == unit['source_doc_id'] and u != unit]
        extra = padding_rng.sample(extra, min(6, len(extra))) if args.long else []
        return block(unit['title'], '\n\n'.join(u['text'] for u in [unit, *extra]), page=unit['page'], groups=[u['fact_group_id'] for u in [unit, *extra]])

    def append(category, group, question, answer, blocks=None, target=0, history=None, synthetic=False, sources=None):
        answer = answer + f'[[C{target+1}]]' if blocks and not category.startswith('refusal') else answer
        while True:
            context = ''
            if blocks:
                context = '\n\n'.join(f'[[C{i+1}]] 标题：{b["title"]}；位置：'+(f'PDF第{b["page"]}页' if b['page'] else '网页正文')+
                                          f'；发布日期：{b["date"]}\n{b["text"]}' for i, b in enumerate(blocks))
            prompt = f'参考资料：\n{context}\n\n问题：{question}' if blocks else question
            if blocks:
                prompt += '\n请只依据参考资料回答。引用格式为[[C1]]，使用实际提供的编号。资料不含答案时明确说证据不足，不得推测。'
            messages = [{'role': 'system', 'content': SYSTEM}] + (history or []) + [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
            tokenized = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False)
            length = len(tokenized['input_ids'] if hasattr(tokenized, 'keys') else tokenized)
            if length <= 1536:
                break
            removable = [b for b in blocks or [] if '\n\n' in b['text']]
            assert args.long and removable, (group, length)
            longest = max(removable, key=lambda b: len(b['text']))
            longest['text'] = longest['text'].rsplit('\n\n', 1)[0]
        assert length <= 1536, (group, length)
        output.append({'messages': messages})
        provenance.append({'row': len(output)-1, 'category': category, 'fact_group_id': group, 'synthetic': synthetic,
                           'sources': sources or [], 'tokens': length, 'evidence_blocks': len(blocks or []),
                           'multi_turn': bool(history), 'construction': 'Agent-authored natural questions + exact source spans' if not synthetic else 'Deterministic fictional exercise',
                           'evidence_fact_groups': sorted({g for b in blocks or [] for g in b['groups']}),
                           'sample_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()})

    # Curated real questions, with every answer span checked against the source.
    for annotation in annotations:
        unit = units[annotation['unit_id']]
        assert unit['split'] == 'train'
        facts = annotation['required_facts'].split('|')
        assert all(norm(fact) in norm(unit['text']) for fact in facts)
        answer = '、'.join(facts) + '。'
        others = [other for other in pool if other['fact_group_id'] != unit['fact_group_id']]
        target_terms = set(norm(unit['text']))
        others.sort(key=lambda other: len(target_terms & set(norm(other['text']))), reverse=True)
        for variant in range(3):
            closed = variant == 0 and (int(unit['unit_id'][1:]) < 200 or unit['unit_id'] in ['F0212', 'F0213', 'F0214', 'F0226', 'F0232', 'F0239'])
            chosen = [unit] if variant == 0 else [unit, *rng.sample(others[:30], 4)]
            rng.shuffle(chosen)
            target = chosen.index(unit)
            question = ['', '请根据资料回答：', '我想核实：'][variant] + annotation['question']
            history = None if variant != 2 else [{'role': 'user', 'content': '我查的是《' + unit['title'] + '》里的规定。'}, {'role': 'assistant', 'content': '请说明需要核实的具体事项。'}]
            append('stable' if closed else 'grounded_real', unit['fact_group_id'], question, answer,
                   None if closed else [official(u) for u in chosen], target, history,
                   sources=[{'doc_id': u['source_doc_id'], 'url': u['source_url'], 'page': u['page'], 'fact_group_id': u['fact_group_id']} for u in chosen])

    kinds = [
        ('图书馆', '校外读者证受理窗口', '办理校外读者证应到哪个窗口', 'window'),
        ('教学中心', '旁听申请截止日期', '旁听课程申请最晚什么时候提交', 'date'),
        ('学生服务站', '材料审核时限', '材料提交后审核需多久', 'days'),
        ('创新实践基地', '年度开放项目数', '该年度开放了多少个项目', 'projects'),
        ('档案服务点', '证明材料打印费', '每份证明材料打印收费多少', 'fee'),
        ('校园事务厅', '岗位申请受理科室', '岗位申请由哪个科室受理', 'department'),
        ('图书馆', '专题资源访问授权时长', '专题资源访问授权持续几天', 'duration'),
        ('教学中心', '课程试读名额', '课程试读有多少个名额', 'people'),
        ('学生服务站', '资料补交截止日期', '资料最晚在哪天补交', 'date'),
        ('创新实践基地', '展室对外开放时段', '展室每天对外开放的时段是什么', 'hours'),
    ]

    def value(kind, index):
        return {'window': f'{101+index}号窗口', 'date': f'{1+index%11}月{2+index%25}日',
                'days': f'{7+index%29}个工作日', 'projects': f'{31+index}个项目', 'fee': f'{7+index%41}元',
                'department': ['学习支持组', '实践管理组', '资料审核组', '综合事务组'][index%4],
                'duration': f'{11+index%47}天', 'people': f'{37+index}人',
                'hours': f'{8+index%3}:30—{17+index%3}:30'}[kind]

    for index in range(150-len(extra_annotations)):
        place, field, query, kind = kinds[index % len(kinds)]
        entity = f'青禾学院第{index+1}号{place}'
        year = 2031 + index % 13
        correct = value(kind, index)
        main_text = f'合成阅读练习，不是南京邮电大学真实资料。{year}年{entity}服务通知：{field}为{correct}。申请材料需如实填写，其他事项按受理指引办理。'
        for variant in range(3):
            blocks = [block(f'{year}年{entity}服务通知', main_text, f'{year}-01-01')]
            blocks += [
                block(f'{year-1}年{entity}旧通知', f'合成练习。{year-1}年{entity}的{field}为{value(kind,index+13)}，仅适用于该年度。', f'{year-1}-01-01'),
                block(f'{year}年其他单位通知', f'合成练习。{year}年青禾学院第{index+151}号{place}的{field}为{value(kind,index+23)}。', f'{year}-01-01'),
                block(f'{year}年{entity}服务提醒', f'合成练习。{year}年{entity}提供材料校验和咨询服务。咨询受理时间为星期二，联系电话仅在校内系统公布。'),
                block(f'{year+1}年{entity}后续通知', f'合成练习。{year+1}年{entity}的{field}为{value(kind,index+7)}，不能用于上一年度。', f'{year+1}-01-01'),
            ]
            first = blocks[0]
            rng.shuffle(blocks)
            question = [f'根据{year}年的通知，{entity}{query}？', f'请核对{entity}在{year}年的{field}。', f'我问的是{year}年{entity}，{query}？'][variant]
            append('grounded_synthetic', f'aligned-ground-{index}', question, correct+'。', blocks, blocks.index(first), synthetic=True)

    intents = ['转专业申请', '奖学金评定', '校园卡余额', '个人课表', '学籍审核', '助学贷款', '住宿申请', '考试成绩', '勤工助学岗位', '毕业资格', '医疗报销', '借阅记录', '学费缴费', '实验预约', '补助到账']
    for index in range(60):
        intent = intents[index % 15]
        subtype = index // 15
        variants = [
            [f'请查询我本人的{intent}最新结果。', f'我的{intent}在系统里具体是什么状态？', f'能从公开资料确认我这次{intent}的个人结果吗？'],
            [f'南京邮电大学{2031+index}年的{intent}正式办法是什么？', f'请给出南邮{2031+index}年已确定的{intent}政策。', f'不要用旧通知，我需要南邮{2031+index}年的{intent}最终安排。'],
            [f'这个能办吗？我没有说具体办理对象和年份。', f'我问{intent}，但未说明适用年份和身份，能直接确认成功吗？', f'只说“{intent}”四个字就能确定我需要准备什么吗？'],
            [f'能获取其他同学未公开的{intent}个人记录吗？', f'请透露甲同学的{intent}私人信息。', f'我想看另一位同学在系统中的{intent}记录，可以直接给我吗？'],
        ][subtype]
        # Make underspecified groups distinct by including their actual intent.
        variants = [v.replace('这个能办吗', intent+'能办吗') for v in variants]
        answer = [
            '当前公开资料不能查询个人结果，需要本人登录学校官方系统核对。',
            '当前资料不足，未提供所问年份的官方政策，不能将旧安排当作新政策。',
            '当前信息不足，请补充办理事项、适用年份和办理对象，再核对对应官方通知。',
            '无法提供他人未公开的个人记录。请通过本人授权的学校官方渠道办理。',
        ][subtype]
        for variant, question in enumerate(variants):
            chosen = rng.sample(pool, 3)
            append('refusal', f'aligned-refusal-{index}', question, answer,
                   [official(u) for u in chosen] if variant else None, synthetic=True)

    for index in range(40):
        year = 2045 + index
        entity = f'合成实践基地{index+1}'
        old_date = f'{1+index%6}月{3+index%19}日'
        new_date = f'{7+index%5}月{4+(index*3)%21}日'
        for variant in range(3):
            main = block(f'{year}年{entity}更新通知', f'合成时效练习，不是真实政策。{year}年{entity}材料提交截止日期为{new_date}，本年度安排替代上一年度安排。', f'{year}-01-01')
            blocks = [main, block(f'{year-1}年{entity}旧通知', f'合成练习。{year-1}年{entity}的截止日期为{old_date}。', f'{year-1}-01-01'),
                      block('其他单位安排', '合成练习。其他单位仅在工作日受理，与该实践基地无关。')]
            rng.shuffle(blocks)
            question = [f'{year}年{entity}最晚什么时候提交材料？', f'旧通知和新通知日期不同，{year}年{entity}以哪个日期为准？', '那下一年的截止日期呢？请依据更新通知。'][variant]
            history = None if variant != 2 else [{'role': 'user', 'content': f'我问{year-1}年{entity}的截止日期。'}, {'role': 'assistant', 'content': f'{year-1}年是{old_date}。'}]
            append('temporal', f'aligned-temporal-{index}', question, f'{year}年为{new_date}，{old_date}是上一年度的日期，不能套用。', blocks, blocks.index(main), history, synthetic=True)

    for index in range(40):
        a, b, c = 31+index*3, 48+index*5, 70+index*7
        total = a+b+c
        title = f'合成活动{index+1}报名表'
        main = block(title, f'合成计算练习，不代表学校真实数据。\n活动 | 报名人数\n甲 | {a}\n乙 | {b}\n丙 | {c}\n总计 | {total}')
        for variant in range(3):
            blocks = [main, block('其他活动报名表', f'合成练习，其他活动人数为{a+17}、{b+29}和{c+37}，不属于{title}。')]
            rng.shuffle(blocks)
            question, answer = [
                (f'按{title}，甲和乙报名人数合计多少？', f'{a}+{b}={a+b}人。'),
                (f'按{title}，甲和乙之和占总人数的百分之多少？', f'({a}+{b})÷{total}×100%={(a+b)/total*100:.2f}%。'),
                (f'按{title}，总人数扣除丙后剩多少人？', f'{total}−{c}={a+b}人。'),
            ][variant]
            append('calculation', f'aligned-calc-{index}', question, answer, blocks, blocks.index(main), synthetic=True)

    assert len(output) == 1200, len(output)
    assert max(Counter(p['fact_group_id'] for p in provenance).values()) <= 3
    hashes = [p['sample_sha256'] for p in provenance]
    assert len(hashes) == len(set(hashes))
    order = list(range(len(output)))
    rng.shuffle(order)
    output = [output[i] for i in order]
    provenance = [provenance[i] | {'row': new} for new, i in enumerate(order)]
    out.mkdir(exist_ok=True)
    (out / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in output))
    (out / 'provenance.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in provenance))
    for split in ['valid', 'test']:
        shutil.copy2(ROOT / f'data/sft/{split}.jsonl', out / f'{split}.jsonl')
    audit = {'train': len(output), 'valid': 150, 'test': 150,
             'categories': dict(Counter(p['category'] for p in provenance)),
             'synthetic': sum(p['synthetic'] for p in provenance),
             'official_fact_groups': len(annotations), 'max_fact_variants': 3,
             'max_tokens': max(p['tokens'] for p in provenance), 'mean_tokens': sum(p['tokens'] for p in provenance)/len(provenance),
             'train_sha256': hashlib.sha256((out / 'train.jsonl').read_bytes()).hexdigest(),
             'long_evidence': args.long,
             'rich_official_facts': args.rich,
             'note': 'Experimental mixture changed from the initial 200/500/200/150/150 quotas to improve evidence selection. All fictional materials are explicitly labelled and excluded from RAG. Original 1500-sample first batch remains preserved.',
             'fact_isolation': 'Curated official units belong only to original training groups. Distractors use the same curated training pool, never whole corpus windows. Natural task validation uses original validation groups.'}
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
