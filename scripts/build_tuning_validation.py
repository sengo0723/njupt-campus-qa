"""Natural task validation over existing validation facts; no training/test80 facts."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def norm(text):
    return re.sub(r'\s+', '', text)


def main():
    units = {u['unit_id']: u for u in map(json.loads, (ROOT / 'data/tuning/source_candidates.jsonl').read_text().splitlines())}
    output = []
    with (ROOT / 'data/tuning/validation_annotations.tsv').open() as stream:
        for annotation in csv.DictReader(stream, delimiter='\t'):
            unit = units[annotation['unit_id']]
            assert unit['split'] == 'valid'
            facts = annotation['required_facts'].split('|')
            assert all(norm(fact) in norm(unit['text']) for fact in facts), (unit['unit_id'], facts)
            output.append({'id': 'TV-' + unit['unit_id'], 'question': annotation['question'], 'required_facts': facts,
                           'reference_answer': '；'.join(facts) + '。', 'topic': unit['topic'], 'answerable': True,
                           'fact_group_id': unit['fact_group_id'], 'gold_doc_ids': [unit['source_doc_id']],
                           'gold_pages': [unit['page']] if unit['page'] else [], 'evidence': unit['text'],
                           'type': 'natural_validation', 'split': 'tuning_valid'})
    negatives = [
        '我今天刚提交了转专业申请，系统是否已经批准？',
        '请查一下我上学期英语期末考试的具体分数。',
        '能根据公开网页告诉我本人校园卡当前余额吗？',
        '我这次申请的国家助学贷款今天是否已经到账？',
        '请提供图书馆某位读者最近一个月的个人借书清单。',
        '现在实验楼里一共有多少个人？请给实时数字。',
        '请确认我明天个人选修课程的上课教室。',
        '能否提供学院老师未公开的私人联系方式？',
        '南邮2033年的正式本科招生专业计划各有多少人？',
        '请给出南邮2034年本科新生已经确定的收费标准。',
        '下一分钟图书馆会空出多少个座位？',
        '能保证我未来高考达到某个分数就一定录取吗？',
        '不看我的具体条件，直接确认我今年一定能拿到困难补助。',
        '请告诉我另一位同学的宿舍门牌和个人手机号。',
        '仅凭学校简介能确定我个人的毕业资格审核已经通过吗？',
    ]
    for index, question in enumerate(negatives):
        output.append({'id': f'TV-U{index:02d}', 'question': question, 'required_facts': [],
                       'reference_answer': '当前公开资料不足以回答，需要本人登录查询或等待相应官方资料。',
                       'topic': ['学生事务', '本科教学', '图书馆服务', '学校概况', '招生'][index % 5],
                       'answerable': False, 'fact_group_id': hashlib.sha256(question.encode()).hexdigest()[:20],
                       'gold_doc_ids': [], 'gold_pages': [], 'evidence': '', 'type': 'unanswerable', 'split': 'tuning_valid'})
    assert len(output) == 60
    path = ROOT / 'data/tuning/validation_60.jsonl'
    path.write_text(''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in output))
    (ROOT / 'data/tuning/validation_protocol.json').write_text(json.dumps({
        'cases': 60, 'positive': 45, 'negative': 15,
        'annotation': 'Manually formulated natural questions; answer facts are literal spans in official source snapshots. Positive facts come exclusively from the pre-existing SFT validation partition. No test80 questions/answers are used for selection.',
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2))
    print('Task validation ready: 45 positive + 15 negative')


if __name__ == '__main__':
    main()
