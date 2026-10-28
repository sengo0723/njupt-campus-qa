"""Bounded, resumable official-site crawler with source snapshots and table preservation."""
from __future__ import annotations
from collections import deque, Counter
from datetime import datetime
from io import BytesIO
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
import hashlib, re, time
import httpx
from bs4 import BeautifulSoup
import pdfplumber
from .paths import DATA, TOPICS, digest, read_jsonl, write_jsonl, save_json

REPORT_URL = 'https://jwc.njupt.edu.cn/_upload/article/files/fd/e1/90d98cee4fc6a6b007f6a40eaa28/97aa85c1-0df8-444f-88fb-6286ea1bf7e1.pdf'
SEEDS = {
 '学校概况': ['https://www.njupt.edu.cn/17220/list.htm', 'https://www.njupt.edu.cn/17221/list.htm',
             'https://www.njupt.edu.cn/17222/list.htm', 'https://dag.njupt.edu.cn/_t75/jj1/list.htm',
             'https://www.njupt.edu.cn/'],
 '本科教学': [REPORT_URL, 'https://jwc.njupt.edu.cn/',
             'https://xxgk.njupt.edu.cn/jxzlxx/list.htm',
             'https://jwc.njupt.edu.cn/_upload/article/files/1e/2c/af7879b84b10b39a320d210e6fb4/bd5ebfd5-563f-4f71-bb4f-dc12aa83471b.pdf'],
 '招生': ['https://zs.njupt.edu.cn/2026/0428/c12721a301108/pagem.htm',
          'https://zs.njupt.edu.cn/2026/0521/c2562a302356/page.htm',
          'https://zs.njupt.edu.cn/2562/list.htm', 'https://zs.njupt.edu.cn/2578/list.htm',
          'https://zs.njupt.edu.cn/'],
 '图书馆服务': ['https://lib.njupt.edu.cn/1385/list.htm', 'https://lib.njupt.edu.cn/1384/list.htm',
               'https://lib.njupt.edu.cn/1383/list.htm', 'https://lib.njupt.edu.cn/1400/list.htm',
               'https://lib.njupt.edu.cn/1409/list.htm', 'https://lib.njupt.edu.cn/zzfw/list.htm',
               'https://lib.njupt.edu.cn/'],
 '学生事务': ['https://xsc.njupt.edu.cn/1169/list.htm', 'https://xsc.njupt.edu.cn/',
             'https://cs.njupt.edu.cn/_upload/article/files/da/24/5c86871a4ae38f69d0940d28fbe2/4bb62b0e-59d5-41ba-b9e0-b95b08fe7d04.pdf'],
}
TERMS = {
 '学校概况': '简介|校史|校训|校标|校歌|校风|精神|章程|校区|历史|沿革|学院介绍|学校概况|院系设置|机构设置',
 '本科教学': '教学|学籍|本科|课程|考试|培养|质量|报告|学分|毕业|学位|管理办法|规章',
 '招生': '2026|2025|2024|2023|章程|计划|报考|问答|收费|专业|简章|评价|资助',
 '图书馆服务': '借阅|开放|入馆|服务|指南|读者|续借|图书馆|预约|资源|馆藏|证明|提交|通知|制度|须知',
 '学生事务': '学生|资助|奖|贷|助|办理|指南|手册|管理|规定|宿舍|证|学费|流程|事务|勤工',
}
UA = 'NJUPT-CourseQA/0.1 (public academic documents; max 1 request/second/host)'

def canonical(url):
    u = urlsplit(url.strip().replace(' ', ''))
    path = re.sub(r'/_t\d+/', '/', u.path)
    return urlunsplit(('https', u.netloc.lower(), path, u.query if 'pdf' in path else '', ''))

def html_table(table):
    grid, spans = [], {}
    for ri, tr in enumerate(table.find_all('tr')):
        row, ci = [], 0
        for cell in tr.find_all(['td', 'th'], recursive=False):
            while (ri, ci) in spans:
                row.append(spans[(ri, ci)]); ci += 1
            text = re.sub(r'\s+', ' ', cell.get_text(' ', strip=True)).strip()
            rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
            for dc in range(cs):
                row.append(text)
                for dr in range(1, rs): spans[(ri+dr, ci+dc)] = text
            ci += cs
        while (ri, ci) in spans:
            row.append(spans[(ri, ci)]); ci += 1
        if any(row): grid.append(row)
    return grid

def parse_html(body, url):
    soup = BeautifulSoup(body, 'lxml')
    links = [(urljoin(url, a.get('href')), a.get_text(' ', strip=True)) for a in soup.select('a[href]')]
    title_node = soup.select_one('h1.arti_title, .Article_Title, .arti_title, h1, h2.arti_title')
    title = title_node.get_text(' ', strip=True) if title_node else (soup.title.get_text(' ', strip=True) if soup.title else '')
    selected = None
    for selector in ['.wp_articlecontent', '.wp_articlecontent.article', '#articlecontent', '.Article_Content', '.arti_content', '.article_content', '.entry-content']:
        nodes = soup.select(selector)
        if nodes:
            selected = max(nodes, key=lambda n: len(n.get_text())); break
    is_content = selected is not None
    if selected is None: selected = soup.body or soup
    all_text = soup.get_text(' ', strip=True)
    d = re.search(r'(?:发布时间|发布日期|时间)\s*[:：]?\s*(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})', all_text)
    date = f'{int(d[1]):04}-{int(d[2]):02}-{int(d[3]):02}' if d else None
    tables = []
    for i, t in enumerate(selected.select('table')):
        rows = html_table(t)
        if rows:
            tables.append({'table_id': f't{i+1}', 'page': None, 'rows': rows})
            t.replace_with(soup.new_string('\n[表格 '+str(i+1)+']\n'+'\n'.join(' | '.join(row) for row in rows)+'\n'))
    for n in selected.select('script, style, nav, header, footer, .wp_paging'): n.decompose()
    lines = [re.sub(r'[\t \u3000]+', ' ', x).strip() for x in selected.get_text('\n', strip=True).splitlines()]
    text = '\n'.join(x for x in lines if x)
    return title, text, date, tables, links, is_content

def parse_pdf(body, url):
    pages, tables = [], []
    with pdfplumber.open(BytesIO(body)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ''
            pages.append({'page': i+1, 'text': text})
            for j, table in enumerate(page.extract_tables() or []):
                rows = [[re.sub(r'\s+', ' ', c or '').strip() for c in row] for row in table]
                if len(rows) > 1 and sum(bool(c) for row in rows for c in row) > 3:
                    tables.append({'table_id': f'p{i+1}t{j+1}', 'page': i+1, 'rows': rows})
    text = '\n\n'.join(f'[PDF第{p["page"]}页]\n{p["text"]}' for p in pages)
    if url == REPORT_URL: title = '南京邮电大学2023—2024学年本科教学质量报告'
    else:
        first = pages[0]['text'].splitlines() if pages else []
        title = ' '.join(first[:3])[:120] or urlsplit(url).path.split('/')[-1]
    return title, text, pages, tables

def crawl(target=12, max_attempts=110):
    out = DATA / 'processed/documents.jsonl'
    docs = read_jsonl(out); log = read_jsonl(DATA / 'crawl_log.jsonl')
    seen = {d['url'] for d in docs}; hashes = {d['content_sha256'] for d in docs}
    counts = Counter(d['topic'] for d in docs)
    robots, last = {}, {}
    client = httpx.Client(timeout=25, follow_redirects=True, headers={'User-Agent': UA})
    def allowed(url):
        u = urlsplit(url)
        if not (u.hostname == 'njupt.edu.cn' or (u.hostname or '').endswith('.njupt.edu.cn')): return False
        base = f'https://{u.netloc}'
        if base not in robots:
            rp = RobotFileParser(); rp.set_url(base + '/robots.txt')
            try:
                r = client.get(rp.url, timeout=8)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception: rp.parse([])
            robots[base] = rp
        return robots[base].can_fetch(UA, url)
    for topic in TOPICS:
        queue = deque((u, 0) for u in SEEDS[topic]); attempted = set()
        while queue and counts[topic] < target and len(attempted) < max_attempts:
            url, depth = queue.popleft(); url = canonical(url)
            if url in attempted: continue
            attempted.add(url)
            if url in seen: continue
            if not allowed(url):
                log.append({'url': url, 'status': 'robots_or_domain_rejected'}); continue
            host = urlsplit(url).netloc
            time.sleep(max(0, 1-(time.monotonic()-last.get(host, 0))))
            try:
                r = client.get(url); last[host] = time.monotonic(); r.raise_for_status()
                body = r.content
                if len(body) > 35_000_000: raise ValueError('File exceeds 35MB limit')
                pdf = body[:4] == b'%PDF'
                if pdf:
                    title, text, pages, tables = parse_pdf(body, url); date = None; links = []; content = True
                else:
                    title, text, date, tables, links, content = parse_html(body, url); pages = []
                # Navigation lists discover sources but are not counted as documents.
                children = []
                for link, label in links:
                    clean = canonical(link)
                    if not re.search(TERMS[topic], label) and not (depth > 0 and urlsplit(clean).hostname == host): continue
                    if re.search(r'\.(jpg|png|gif|mp4|zip|xls|xlsx|doc|docx)$', clean, re.I): continue
                    if topic == '学校概况' and urlsplit(clean).hostname not in ['www.njupt.edu.cn','dag.njupt.edu.cn']: continue
                    if re.search(r'/20(?:0\d|1\d|2[0-2])/', clean) and topic not in ['学校概况','本科教学']: continue
                    children.append((clean, depth+1))
                children.sort(key=lambda x: (not x[0].endswith('.pdf'), 'list.' in x[0]))
                if depth < 3: queue.extend(children)
                if not content or len(text) < 180:
                    log.append({'url': url, 'status': 'navigation_or_short', 'discovered': len(children)}); continue
                sha = hashlib.sha256(text.encode()).hexdigest()
                if sha in hashes:
                    log.append({'url': url, 'status': 'duplicate_content'}); continue
                doc_id = digest(url)
                raw_path = DATA / 'raw' / (doc_id + ('.pdf' if pdf else '.html'))
                raw_path.write_bytes(body)
                years = sorted(set(re.findall(r'20\d{2}', title)))
                doc = {'doc_id': doc_id, 'topic': topic, 'title': title, 'url': url,
                       'published_at': date, 'crawled_at': datetime.now().astimezone().isoformat(),
                       'years': years, 'text': text, 'pages': pages, 'tables': tables,
                       'raw_file': str(raw_path.relative_to(DATA.parent)), 'content_sha256': sha,
                       'raw_sha256': hashlib.sha256(body).hexdigest(), 'format': 'pdf' if pdf else 'html'}
                docs.append(doc); seen.add(url); hashes.add(sha); counts[topic] += 1
                print(f'[{topic} {counts[topic]}/{target}] {title[:90]} ({len(text)} chars)', flush=True)
                write_jsonl(out, docs)
                log.append({'url': url, 'status': 'saved', 'doc_id': doc_id})
            except Exception as e:
                last[host] = time.monotonic()
                print('Fetch failed', url, str(e)[:140], flush=True)
                log.append({'url': url, 'status': 'failed', 'error': str(e)[:300]})
            write_jsonl(DATA / 'crawl_log.jsonl', log)
        print('Topic complete', topic, counts[topic], flush=True)
    save_json(DATA / 'corpus_manifest.json', {'counts': dict(counts), 'documents': len(docs),
         'captured_at': datetime.now().astimezone().isoformat(), 'target_per_topic': target,
         'eligible': all(counts[t] >= 8 for t in TOPICS), 'sources': [{k:d[k] for k in ['doc_id','topic','title','url','published_at','content_sha256']} for d in docs]})
    return docs

if __name__ == '__main__': crawl()
