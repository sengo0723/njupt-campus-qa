"""Check the explicit Git file set before publishing. Print locations, never secrets."""
from pathlib import Path
import re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS={'.venv','node_modules','models','raw','__pycache__','.git'}
FORBIDDEN_SUFFIXES={'.sqlite','.safetensors','.gguf','.onnx','.pt','.pth','.pem','.key','.docx','.pdf','.zip'}
PATTERNS=[
    ('GitHub credential',r'gh[pousr]_[A-Za-z0-9]{30,}'),
    ('Hugging Face credential',r'hf_[A-Za-z0-9]{30,}'),
    ('API credential',r'sk-[A-Za-z0-9_-]{30,}'),
    ('private key',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    ('personal absolute path',r'/(?:Users|home)/[A-Za-z0-9_.-]+/'),
]

def main():
    selected=subprocess.check_output(['git','ls-files','--cached','-z'],cwd=ROOT).decode().split('\0')
    issues=[];count=0
    for name in filter(None,selected):
        path=ROOT/name;count+=1
        if path.is_symlink():issues.append((name,'symlink'));continue
        if not path.is_file():issues.append((name,'missing tracked file'));continue
        if set(path.parts[len(ROOT.parts):])&FORBIDDEN_DIRS or path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith('.env'):
            issues.append((name,'private/runtime file type'))
        if path.stat().st_size>10*1024*1024:issues.append((name,'file exceeds 10 MiB publication limit'))
        try:text=path.read_text()
        except UnicodeDecodeError:continue
        for label,pattern in PATTERNS:
            match=re.search(pattern,text)
            if match:issues.append((name,f'{label} near line {text[:match.start()].count(chr(10))+1}'))
    if issues:
        for name,reason in issues:print(name+': '+reason)
        raise SystemExit(1)
    if count==0:raise SystemExit('No tracked files were checked; stage the intended public files first.')
    print(f'PASS: {count} tracked files; no blocked artifacts, personal paths or recognized credentials')

if __name__=='__main__':main()
