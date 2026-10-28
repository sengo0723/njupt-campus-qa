"""Restore only byte-identical raw snapshots, without changing frozen cleaned data."""
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser
import hashlib,json,time
import httpx
ROOT=Path(__file__).resolve().parents[1]
AGENT='NJUPTCampusQA/0.1 (educational source restoration)'
MAX_BYTES=40*1024*1024

def official(url):
    part=urlsplit(url);host=(part.hostname or '').lower()
    return part.scheme=='https' and (host=='njupt.edu.cn' or host.endswith('.njupt.edu.cn')) and not part.username

class Fetcher:
    def __init__(self,client):self.client=client;self.last={};self.robots={}
    def get(self,url):
        for _ in range(9):
            if not official(url):raise ValueError('Source or redirect is outside the official HTTPS domain')
            origin='https://'+urlsplit(url).netloc
            delay=1-(time.monotonic()-self.last.get(origin,0))
            if delay>0:time.sleep(delay)
            self.last[origin]=time.monotonic()
            with self.client.stream('GET',url) as response:
                if response.is_redirect:
                    url=str(response.url.join(response.headers['location']));continue
                response.raise_for_status();chunks=[];size=0
                for chunk in response.iter_bytes():
                    size+=len(chunk)
                    if size>MAX_BYTES:raise ValueError('Source exceeds the download size limit')
                    chunks.append(chunk)
                return b''.join(chunks)
        raise ValueError('Too many redirects')
    def permitted(self,url):
        origin='https://'+urlsplit(url).netloc
        if origin not in self.robots:
            robot=RobotFileParser(origin+'/robots.txt')
            try:robot.parse(self.get(origin+'/robots.txt').decode('utf-8','replace').splitlines())
            except httpx.HTTPStatusError as error:
                if error.response.status_code==404:robot.parse([])
                else:raise
            self.robots[origin]=robot
        return self.robots[origin].can_fetch(AGENT,url)

def main():
    sources=json.loads((ROOT/'data/corpus_manifest.json').read_text())['sources'];rows=[]
    raw=(ROOT/'data/raw').resolve();raw.mkdir(parents=True,exist_ok=True)
    with httpx.Client(timeout=60,trust_env=False,headers={'User-Agent':AGENT}) as client:
        fetch=Fetcher(client)
        for source in sources:
            row={'doc_id':source['doc_id'],'url':source['url']}
            try:
                target=(ROOT/source['raw_file']).resolve()
                if not target.is_relative_to(raw):raise ValueError('Source path escapes data/raw')
                expected=source['raw_sha256']
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
                    row['status']='already_verified'
                else:
                    if not fetch.permitted(source['url']):raise ValueError('Blocked by robots.txt')
                    data=fetch.get(source['url']);actual=hashlib.sha256(data).hexdigest()
                    if actual!=expected:
                        row.update(status='snapshot_changed',expected_sha256=expected,actual_sha256=actual)
                    else:
                        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);row['status']='restored'
            except Exception as error:row.update(status='error',error=str(error))
            rows.append(row);print(source['doc_id'],row['status'],flush=True)
    report={'sources':len(rows),'verified':sum(r['status'] in ['restored','already_verified'] for r in rows),'checks':rows}
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/source_restore.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if report['verified']!=len(rows):raise SystemExit('Some original snapshots could not be restored. See results/source_restore.json.')

if __name__=='__main__':main()
