from __future__ import annotations
from dataclasses import dataclass
import json, math, re, sqlite3, time, os
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from .paths import DATA, MODELS, RESULTS, digest, read_jsonl, write_jsonl, save_json

STOP=set('的了是在和与及或等为对中按于将请问如何什么多少根据目前学校南京邮电大学'.split())
def words(text):
    return [w.lower() for w in jieba.lcut(text) if w.strip() and re.search(r'[\w\u4e00-\u9fff]',w) and w not in STOP]

class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        torch.set_num_threads(3)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(MODELS/'bge-small-zh-v1.5',local_files_only=True)
        self.model=AutoModel.from_pretrained(MODELS/'bge-small-zh-v1.5',local_files_only=True).eval()
    def encode(self,texts,query=False):
        if query: texts=['为这个句子生成表示以用于检索相关文章：'+x for x in texts]
        rows=[]
        for i in range(0,len(texts),24):
            inputs=self.tokenizer(texts[i:i+24],padding=True,truncation=True,max_length=512,return_tensors='pt')
            with self.torch.inference_mode():
                v=self.model(**inputs).last_hidden_state[:,0]
                v=self.torch.nn.functional.normalize(v,p=2,dim=1)
            rows.append(v.numpy())
        return np.concatenate(rows).astype(np.float32)

def chunk_documents(docs,tokenizer):
    chunks=[]
    for doc in docs:
        units=doc['pages'] or [{'page':None,'text':doc['text']}]
        for unit in units:
            # Split first at paragraphs/sentences, then enforce a real token limit.
            text=unit['text']
            encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            ids=encoded['input_ids'];offsets=encoded['offset_mapping']
            for start in range(0,len(ids),300):
                end=min(start+350,len(ids))
                # Slice the original source by tokenizer offsets. WordPiece.decode
                # inserts spaces between Han characters and loses punctuation case.
                passage=text[offsets[start][0]:offsets[end-1][1]]
                if len(passage.strip())<20: continue
                chunk={'chunk_id':digest(doc['doc_id']+':'+str(unit['page'])+':'+str(start)+':'+passage),
                  'doc_id':doc['doc_id'],'title':doc['title'],'topic':doc['topic'],'url':doc['url'],
                  'published_at':doc['published_at'],'years':doc['years'],'page':unit['page'],
                  'text':passage,'kind':'text','raw_file':doc['raw_file']}
                chunks.append(chunk)
                if start+350>=len(ids): break
        for table in doc['tables']:
            rows=table['rows']
            headers=' | '.join(rows[0])
            for start in range(1,len(rows),6):
                selected=[rows[0]]+rows[start:start+6]
                passage='[表格 '+table['table_id']+']\n'+'\n'.join(' | '.join(r) for r in selected)
                if len(passage.strip())<30: continue
                chunks.append({'chunk_id':digest(doc['doc_id']+table['table_id']+str(start)),
                  'doc_id':doc['doc_id'],'title':doc['title'],'topic':doc['topic'],'url':doc['url'],
                  'published_at':doc['published_at'],'years':doc['years'],'page':table['page'],
                  'text':passage,'kind':'table','table_id':table['table_id'],'raw_file':doc['raw_file']})
    return chunks

def build_index():
    docs=read_jsonl(DATA/'processed/documents.jsonl')
    embed=Embedder(); chunks=chunk_documents(docs,embed.tokenizer)
    print('Encoding',len(chunks),'chunks',flush=True)
    vectors=embed.encode([c['title']+'\n'+c['text'] for c in chunks])
    db=sqlite3.connect(DATA/'knowledge.sqlite');db.execute('CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    db.execute('DELETE FROM chunks')
    db.executemany('INSERT INTO chunks VALUES(?,?)',[(c['chunk_id'],json.dumps(c,ensure_ascii=False)) for c in chunks]);db.commit();db.close()
    np.save(DATA/'vectors.npy',vectors)
    write_jsonl(DATA/'processed/chunks.jsonl',chunks)
    save_json(DATA/'index_manifest.json',{'documents':len(docs),'chunks':len(chunks),'embedding_dimension':int(vectors.shape[1]),
        'embedding_model':'bge-small-zh-v1.5','chunk_tokens':350,'overlap_tokens':50,'fusion':'RRF k=60','dense_candidates':20,'sparse_candidates':20,'final_k':5})
    print('Index ready',vectors.shape,flush=True)

class Retriever:
    def __init__(self):
        self.chunks=read_jsonl(DATA/'processed/chunks.jsonl')
        self.vectors=np.load(DATA/'vectors.npy',mmap_mode='r')
        self.bm25=BM25Okapi([words(c['title']+' '+c['text']) for c in self.chunks])
        self.embed=Embedder();self.reranker=None;self.rerank_tokenizer=None
    def _load_reranker(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        opts=ort.SessionOptions();opts.intra_op_num_threads=3;opts.inter_op_num_threads=1
        model_path=MODELS/'bge-reranker-base/onnx/model.onnx'
        if not model_path.exists():model_path=MODELS/'bge-reranker-base/model.onnx'
        self.reranker=ort.InferenceSession(str(model_path),sess_options=opts,providers=['CPUExecutionProvider'])
        self.rerank_tokenizer=AutoTokenizer.from_pretrained(MODELS/'bge-reranker-base',local_files_only=True)
    def retrieve(self,question,topic=None,year=None,mode='full',k=5):
        begin=time.perf_counter();timings={}
        asked_years=re.findall(r'20\d{2}',question)
        wanted=str(year) if year else (asked_years[0] if asked_years else None)
        eligible=[]
        for i,c in enumerate(self.chunks):
            if topic and c['topic']!=topic: continue
            if wanted and c['years'] and wanted not in c['years']: continue
            eligible.append(i)
        if not eligible: return [],{'total_ms':0,'reason':'empty_filter'}
        eligible=np.array(eligible,dtype=int)
        t=time.perf_counter();sparse=self.bm25.get_scores(words(question));timings['sparse_ms']=(time.perf_counter()-t)*1000
        sparse_ids=eligible[np.argsort(sparse[eligible])[::-1][:20]].tolist()
        dense=np.zeros(len(self.chunks),dtype=np.float32);dense_ids=[]
        if mode!='bm25':
            t=time.perf_counter();q=self.embed.encode([question],query=True)[0];dense=self.vectors@q;timings['embed_dense_ms']=(time.perf_counter()-t)*1000
            dense_ids=eligible[np.argsort(dense[eligible])[::-1][:20]].tolist()
        if mode=='bm25': ranking=sparse_ids
        elif mode=='dense': ranking=dense_ids
        else:
            scores={}
            for ids in [sparse_ids,dense_ids]:
                for rank,i in enumerate(ids): scores[i]=scores.get(i,0)+1/(60+rank+1)
            ranking=sorted(scores,key=scores.get,reverse=True)[:20]
        rerank_scores={};warnings=[]
        if mode=='full' and ranking:
            try:
                if self.reranker is None: self._load_reranker()
                t=time.perf_counter()
                pairs=[[question,self.chunks[i]['title']+'\n'+self.chunks[i]['text']] for i in ranking]
                names={x.name for x in self.reranker.get_inputs()}
                values=[]
                for off in range(0,len(pairs),4):
                    batch=self.rerank_tokenizer(pairs[off:off+4],padding=True,truncation=True,max_length=512,return_tensors='np')
                    inputs={name:np.asarray(batch[name],dtype=np.int64) for name in names if name in batch}
                    vals=self.reranker.run(None,inputs)[0].reshape(-1)
                    values.extend(float(v) for v in vals)
                rerank_scores=dict(zip(ranking,values));ranking=sorted(ranking,key=lambda i:rerank_scores[i],reverse=True)
                timings['rerank_ms']=(time.perf_counter()-t)*1000
            except Exception as e:
                warnings.append('本地精排降级为融合排序: '+str(e)[:100])
        result=[];seen=set()
        for i in ranking:
            c=self.chunks[i]
            # Preserve independent evidence, rather than near-identical overlapping windows.
            fingerprint=(c['doc_id'],c['page'],c['text'][:80])
            if fingerprint in seen: continue
            seen.add(fingerprint)
            item=dict(c);item['content']=item.pop('text');item['id']='C'+str(len(result)+1)
            item['dense_score']=float(dense[i]);item['sparse_score']=float(sparse[i]);item['rerank_score']=rerank_scores.get(i)
            result.append(item)
            if len(result)>=k: break
        timings['total_ms']=(time.perf_counter()-begin)*1000;timings['warnings']=warnings
        return result,timings

if __name__=='__main__': build_index()
