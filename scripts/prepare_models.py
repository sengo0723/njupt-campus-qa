"""Download frozen official model revisions and verify expected checksums."""
from pathlib import Path
import hashlib,json
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    pins=json.loads((ROOT/'config/model_sources.json').read_text())
    models=ROOT/'models';models.mkdir(exist_ok=True);manifest={}
    for name,patterns in [
        ('qwen3-base-hf',['*.safetensors','*.json','*.txt','README.md','LICENSE']),
        ('bge-small-zh-v1.5',['*.safetensors','*.json','*.txt','README.md','LICENSE']),
        ('bge-reranker-base',['onnx/model.onnx','*.json','*.txt','README.md','LICENSE'])]:
        pin=pins[name];out=models/name
        snapshot_download(pin['repo'],revision=pin['revision'],local_dir=out,allow_patterns=patterns,max_workers=3)
        expected=pin.get('files',{}) if name!='bge-reranker-base' else {'onnx/model.onnx':pin['sha256']}
        for file,digest in expected.items():
            path=out/file
            if not path.is_file() or sha(path)!=digest:
                raise RuntimeError(f'Checksum mismatch: {name}/{file}. Remove the bad file and download again.')
        manifest[name]={'repo':pin['repo'],'revision':pin['revision'],'files':{
            str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and '.cache' not in p.parts}}
        print('Verified',pin['repo'],pin['revision'],flush=True)
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2
    ort.InferenceSession(str(models/'bge-reranker-base/onnx/model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    (models/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
