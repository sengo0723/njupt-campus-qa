"""Freeze a new task test before model selection; this set must not tune weights."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    units = {u['unit_id']: u for u in map(json.loads, (ROOT / 'data/tuning/source_candidates.jsonl').read_text().splitlines())}
    output = []
    norm = lambda text: re.sub(r'\s+', '', text)
    with (ROOT / 'data/tuning/fresh_test_annotations.tsv').open() as stream:
        for annotation in csv.DictReader(stream, delimiter='\t'):
            unit = units[annotation['unit_id']]
            facts = annotation['required_facts'].split('|')
            assert unit['split'] == 'test'
            assert all(norm(fact) in norm(unit['text']) for fact in facts), (unit['unit_id'], facts)
            output.append({'id': 'FT-' + unit['unit_id'], 'question': annotation['question'], 'required_facts': facts,
                           'reference_answer': '；'.join(facts) + '。', 'topic': unit['topic'], 'answerable': True,
                           'fact_group_id': unit['fact_group_id'], 'gold_doc_ids': [unit['source_doc_id']],
                           'gold_pages': [unit['page']] if unit['page'] else [], 'evidence': unit['text'],
                           'type': 'fresh_natural_test', 'split': 'fresh_test'})
    negatives = [
        '请确认我这次缓考申请是否已经审批通过。',
        '我本人上一学期的具体学分绩点是多少？',
        '能直接告诉我个人学费账户现在还有多少欠费吗？',
        '系统里我的本年度学位授予审核是否通过了？',
        '请列出一名未授权同学的个人贷款还款明细。',
        '校园网此刻在线的设备数量精确是多少？',
        '我的个人门禁记录显示昨天几点离开图书馆？',
        '请查询另一名学生的身份证号码。',
        '南邮2038年的综合评价考试日期最终定在哪天？',
        '南邮2039年普通本科生完整收费表已经确定为多少？',
        '现在指定教室里具体还有几个空座位？',
        '能保证我下次转专业申请一定成功吗？',
        '没有给出任何个人材料，能直接确定我符合学费减免条件吗？',
        '请给我某同学没有公开的家庭住址。',
        '仅凭学生管理制度能确认我本人现在具有学籍吗？',
    ]
    for index, question in enumerate(negatives):
        output.append({'id': f'FT-U{index:02d}', 'question': question, 'required_facts': [],
                       'reference_answer': '当前公开资料不足以确认，需要本人登录核对或等待官方公布。',
                       'topic': ['学生事务', '本科教学', '图书馆服务', '学校概况', '招生'][index % 5],
                       'answerable': False, 'fact_group_id': hashlib.sha256(question.encode()).hexdigest()[:20],
                       'gold_doc_ids': [], 'gold_pages': [], 'evidence': '', 'type': 'unanswerable', 'split': 'fresh_test'})
    assert len(output) == 60
    path = ROOT / 'data/tuning/fresh_test_60.jsonl'
    text = ''.join(json.dumps(case, ensure_ascii=False)+'\n' for case in output)
    if path.exists():
        assert path.read_text() == text, 'Frozen test cannot change during tuning'
    else:
        path.write_text(text)
        (ROOT / 'data/tuning/fresh_test_protocol.json').write_text(json.dumps({
            'frozen_at': datetime.now().astimezone().isoformat(), 'cases': 60, 'positive': 45, 'negative': 15,
            'purpose': 'Final-only test. No model generation on this set until hyperparameters and adapter are locked.',
            'source': 'Official literal spans from the original SFT test fact partition; natural questions authored independently of candidate outputs.',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }, ensure_ascii=False, indent=2))
    print('Fresh final-only test frozen: 45 positive + 15 negative')


if __name__ == '__main__':
    main()
