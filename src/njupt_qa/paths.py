from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
TOPICS = ['学校概况', '本科教学', '招生', '图书馆服务', '学生事务']

def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]

def read_jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))

def save_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2))
