"""Reproduce the curated official corpus from its frozen source manifest."""
import json
from .paths import ROOT,DATA,read_jsonl,write_jsonl

def collect():
    import njupt_qa.crawl as c
    manifest=json.loads((ROOT/'config/sources.json').read_text())
    c.SEEDS={t:[s['url'] for s in manifest if s['topic']==t] for t in c.TOPICS}
    c.TERMS={t:'(?!)' for t in c.TOPICS}
    c.crawl(target=12,max_attempts=60)
    labels={s['url']:s for s in manifest}
    docs=read_jsonl(DATA/'processed/documents.jsonl')
    for d in docs:
        if d['url'] in labels:
            source=labels[d['url']]
            d['title']=source['title'];d['years']=source['years']
            if source.get('parent_url'):d['parent_url']=source['parent_url']
    write_jsonl(DATA/'processed/documents.jsonl',docs)

if __name__=='__main__':collect()
