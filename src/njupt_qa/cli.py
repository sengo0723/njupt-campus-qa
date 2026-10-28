import argparse,json
from .paths import ROOT

def main():
 p=argparse.ArgumentParser(description='南邮知问本地实验工具');sub=p.add_subparsers(dest='command',required=True)
 for n in ['crawl','dataset','index','retrieval-eval','acceptance','doctor']:sub.add_parser(n)
 s=sub.add_parser('serve');s.add_argument('--port',type=int,default=8765)
 c=sub.add_parser('chat');c.add_argument('question');c.add_argument('--mode',choices=['raw','rag','sft','sft_rag'],default='sft_rag')
 e=sub.add_parser('eval');e.add_argument('--split',choices=['dev','test'],default='test');e.add_argument('--modes',nargs='+',choices=['raw','sft','rag','sft_rag']);e.add_argument('--calculator',action='store_true');e.add_argument('--max-tokens',type=int,default=768)
 a=p.parse_args()
 if a.command=='crawl':from .collect import collect;return collect()
 if a.command=='dataset':from .dataset import build_all;return build_all()
 if a.command=='index':from .retrieval import build_index;return build_index()
 if a.command=='serve':
  import uvicorn;uvicorn.run('njupt_qa.app:app',host='127.0.0.1',port=a.port);return
 if a.command=='doctor':from .app import health;print(json.dumps(health(),ensure_ascii=False,indent=2));return
 if a.command=='chat':
  from .engine import get_engine
  r=get_engine().answer(a.question,a.mode);print(json.dumps(r,ensure_ascii=False,indent=2));return
 from .evaluation import evaluate,retrieval_evaluation,acceptance
 if a.command=='eval':return evaluate(a.split,a.modes,a.calculator,a.max_tokens)
 if a.command=='retrieval-eval':return retrieval_evaluation()
 if a.command=='acceptance':return acceptance()

if __name__=='__main__':main()
