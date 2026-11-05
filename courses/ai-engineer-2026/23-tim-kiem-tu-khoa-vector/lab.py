"""TF-IDF/BM25/LSA exact search + RRF + graded evaluation. CPU NumPy, no downloads."""
import argparse
from collections import Counter,defaultdict
from dataclasses import dataclass,asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import unicodedata
import numpy as np
from data import documents,queries

ANALYZER_VERSION='nfc-casefold-word-compound-v1'

def analyze(text):
    return re.findall(r'\w+(?:[-./]\w+)*',unicodedata.normalize('NFC',text).casefold())


def normalize_rows(x):
    x=np.asarray(x,dtype=np.float64)
    norm=np.linalg.norm(x,axis=-1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=norm>1e-12)

@dataclass(frozen=True)
class Config:
    lsa_dim:int=6
    k1:float=1.2
    b:float=.75
    rrf_constant:int=60
    candidate_depth:int=6


def rrf(rankings,constant=60):
    if constant<1:raise ValueError('RRF constant must be >= 1')
    scores=defaultdict(float)
    for ranking in rankings:
        if len(ranking)!=len(set(ranking)):raise ValueError('Duplicate ID within a ranking')
        for rank,doc_id in enumerate(ranking,1):scores[doc_id]+=1/(constant+rank)
    return sorted(scores.items(),key=lambda pair:(-pair[1],pair[0]))

class SearchIndex:
    def __init__(self,docs,cfg=Config(),*,_fit_dense=True):
        if not docs or len({d['id'] for d in docs})!=len(docs):raise ValueError('Nonempty corpus and unique IDs required')
        if cfg.k1<=0 or not 0<=cfg.b<=1 or cfg.lsa_dim<1 or cfg.candidate_depth<1 or cfg.rrf_constant<1:raise ValueError('Invalid search config')
        self.docs=[dict(d) for d in docs];self.cfg=cfg
        self.tokens=[analyze(d['title']+' '+d['text']) for d in self.docs]
        self.vocab=sorted(set(t for seq in self.tokens for t in seq));self.term_to_col={t:i for i,t in enumerate(self.vocab)}
        if not self.vocab:raise ValueError('Corpus has no analyzed terms')
        self.counts=np.zeros((len(docs),len(self.vocab)),dtype=np.float64)
        self.postings=defaultdict(list)
        for row,seq in enumerate(self.tokens):
            for term,tf in Counter(seq).items():
                self.counts[row,self.term_to_col[term]]=tf;self.postings[term].append((row,tf))
        self.lengths=self.counts.sum(1);self.avgdl=float(self.lengths.mean());self.df=(self.counts>0).sum(0)
        self.idf=np.log((1+len(docs))/(1+self.df))+1
        self.tfidf=normalize_rows(self.counts*self.idf)
        # Dense SVD is intentional for tiny corpus, not a scalable inverted/vector database.
        if cfg.lsa_dim>min(self.tfidf.shape):raise ValueError('lsa_dim exceeds matrix size')
        if _fit_dense:
            _,s,vt=np.linalg.svd(self.tfidf,full_matrices=False)
            self.components=vt[:cfg.lsa_dim]
            self.singular_values=s
            self.dense=normalize_rows(self.tfidf@self.components.T)
        self.corpus_hash=hashlib.sha256(json.dumps(self.docs,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def query_tfidf(self,query):
        counts=np.zeros(len(self.vocab))
        for term,tf in Counter(analyze(query)).items():
            if term in self.term_to_col:counts[self.term_to_col[term]]=tf
        return normalize_rows(counts*self.idf)

    def bm25(self,query):
        scores=np.zeros(len(self.docs));n=len(self.docs)
        for term in set(analyze(query)):  # Unique query terms; no query-tf multiplier.
            if term not in self.postings:continue
            col=self.term_to_col[term];idf=math.log(1+(n-self.df[col]+.5)/(self.df[col]+.5))
            for row,tf in self.postings[term]:
                factor=tf*(self.cfg.k1+1)/(tf+self.cfg.k1*(1-self.cfg.b+self.cfg.b*self.lengths[row]/self.avgdl))
                scores[row]+=idf*factor
        return scores

    def scores(self,query,method):
        if method=='bm25':return self.bm25(query)
        q=self.query_tfidf(query)
        if method=='tfidf':return self.tfidf@q
        if method=='lsa':return self.dense@normalize_rows(q@self.components.T)
        raise ValueError('Unknown retrieval method')

    def search(self,query,method='bm25',k=3,allowed_scopes=None):
        if type(k) is not int or k<1:raise ValueError('k must be a positive integer')
        scopes={'public'} if allowed_scopes is None else set(allowed_scopes)
        if method=='hybrid':
            ranked=[self.search(query,m,self.cfg.candidate_depth,scopes) for m in ['bm25','lsa']]
            combined=rrf([[r['id'] for r in result] for result in ranked],self.cfg.rrf_constant)
            byid={d['id']:d for d in self.docs}
            return [dict(id=i,score=float(s),title=byid[i]['title']) for i,s in combined[:k]]
        scores=self.scores(query,method)
        # Filter BEFORE top-k. Zero/negative scores are not padded with arbitrary docs.
        candidates=[i for i,d in enumerate(self.docs) if d['scope'] in scopes and scores[i]>1e-12]
        candidates.sort(key=lambda i:(-scores[i],self.docs[i]['id']))
        return [dict(id=self.docs[i]['id'],score=float(scores[i]),title=self.docs[i]['title']) for i in candidates[:k]]

    def save(self,out):
        out.mkdir(parents=True,exist_ok=False)
        metadata=dict(docs=self.docs,config=asdict(self.cfg),vocab=self.vocab,idf=self.idf.tolist(),
                      corpus_hash=self.corpus_hash,analyzer=ANALYZER_VERSION)
        (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        np.savez(out/'arrays.npz',counts=self.counts,tfidf=self.tfidf,components=self.components,dense=self.dense,singular_values=self.singular_values)

    @classmethod
    def load(cls,path):
        obj=json.loads((path/'metadata.json').read_text())
        if obj['analyzer']!=ANALYZER_VERSION:raise ValueError('Analyzer version mismatch')
        # Rebuild lexical statistics from saved corpus; use persisted learned components, never refit from queries.
        index=cls(obj['docs'],Config(**obj['config']),_fit_dense=False)
        if index.corpus_hash!=obj['corpus_hash'] or index.vocab!=obj['vocab']:raise ValueError('Metadata mismatch')
        with np.load(path/'arrays.npz',allow_pickle=False) as arrays:
            if not np.array_equal(index.counts,arrays['counts']):raise ValueError('Count mismatch')
            if not np.allclose(index.idf,obj['idf']):raise ValueError('IDF mismatch')
            if arrays['components'].shape!=(index.cfg.lsa_dim,len(index.vocab)):raise ValueError('Component dimension mismatch')
            np.testing.assert_allclose(index.tfidf,arrays['tfidf'])
            index.tfidf=arrays['tfidf'].copy();index.components=arrays['components'].copy();index.dense=arrays['dense'].copy();index.singular_values=arrays['singular_values'].copy()
        np.testing.assert_allclose(index.dense,normalize_rows(index.tfidf@index.components.T))
        return index


def retrieval_metrics(ranked_ids,relevance,k=3):
    if len(ranked_ids)!=len(set(ranked_ids)):raise ValueError('Duplicate retrieved ID')
    relevant={i for i,g in relevance.items() if g>0}
    if not relevant:raise ValueError('No-answer cases must use separate evaluation')
    ids=ranked_ids[:k];hits=sum(i in relevant for i in ids)
    rr=next((1/r for r,i in enumerate(ids,1) if i in relevant),0.)
    dcg=sum((2**relevance.get(i,0)-1)/math.log2(r+1) for r,i in enumerate(ids,1))
    ideal=sorted(relevance.values(),reverse=True)[:k]
    idcg=sum((2**g-1)/math.log2(r+1) for r,g in enumerate(ideal,1))
    return dict(hit=float(hits>0),precision=hits/k,recall=hits/len(relevant),mrr=rr,ndcg=dcg/idcg)


def evaluate(index):
    methods=['tfidf','bm25','lsa','hybrid'];rows=[]
    for q in queries():
        for method in methods:
            result=index.search(q['text'],method,k=3);ids=[d['id'] for d in result]
            row=dict(query=q['id'],split=q['split'],slice=q['slice'],method=method,ranked_ids=ids,
                     relevance=q['relevance'],metrics=retrieval_metrics(ids,q['relevance']) if q['relevance'] else None,
                     returned_any=bool(ids))
            rows.append(row)
    aggregates={}
    for split in ['dev','test']:
        aggregates[split]={}
        for method in methods:
            selected=[r for r in rows if r['split']==split and r['method']==method and r['metrics'] is not None]
            aggregates[split][method]={m:sum(r['metrics'][m] for r in selected)/len(selected) for m in ['hit','precision','recall','mrr','ndcg']}
            aggregates[split][method]['queries']=len(selected)
    negative={m:dict(queries=sum(r['split']=='test' and r['metrics'] is None and r['method']==m for r in rows),
                      returned_any=sum(r['split']=='test' and r['metrics'] is None and r['method']==m and r['returned_any'] for r in rows)) for m in methods}
    report=dict(kind='tiny original corpus; exact NumPy search; LSA is NOT a pretrained neural encoder',
                corpus_docs=len(index.docs),visible_docs=sum(d['scope']=='public' for d in index.docs),vocab=len(index.vocab),
                config=asdict(index.cfg),corpus_hash=index.corpus_hash,queries_hash=hashlib.sha256(json.dumps(queries(),sort_keys=True,ensure_ascii=False).encode()).hexdigest(),aggregates=aggregates,no_answer=negative,rows=rows,
                metadata=dict(python=platform.python_version(),numpy=np.__version__,device='CPU',analyzer=ANALYZER_VERSION))
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


def hand_demo():
    # Three docs, vocab=(api,timeout,token); score all with public scope.
    docs=[dict(id='A',title='',text='api timeout timeout',scope='public'),dict(id='B',title='',text='api token',scope='public'),dict(id='C',title='',text='token',scope='public')]
    index=SearchIndex(docs,Config(lsa_dim=2))
    a=math.log(4/3)+1;b=math.log(2)+1
    np.testing.assert_allclose(index.idf,[a,b,a])
    q=index.query_tfidf('api timeout')
    tfidf=index.tfidf@q
    np.testing.assert_allclose(tfidf,[(a*a+2*b*b)/(math.sqrt(a*a+4*b*b)*math.sqrt(a*a+b*b)),a/(math.sqrt(2)*math.sqrt(a*a+b*b)),0])
    bm=index.bm25('api timeout')
    expected=[math.log(1.6)*(2.2/2.65)+math.log(8/3)*(4.4/3.65),math.log(1.6),0]
    np.testing.assert_allclose(bm,expected)
    vector=np.array([1.,1.]);docs_v=np.array([[1.,0.],[0.,1.],[2.,2.]])
    cos=normalize_rows(docs_v)@normalize_rows(vector)
    np.testing.assert_allclose(cos,[1/math.sqrt(2),1/math.sqrt(2),1.])
    fusion=rrf([['A','B','C'],['B','C','A']],constant=10)
    metrics=retrieval_metrics(['B','X','A'],{'A':3,'B':1},3)
    report=dict(tfidf_idf=dict(zip(index.vocab,index.idf.tolist())),tfidf_cosine=tfidf.tolist(),bm25=bm.tolist(),cosine=cos.tolist(),rrf=fusion,metrics=metrics)
    print(json.dumps(report,indent=2));return report


def search_demo(index,query):
    result={m:index.search(query,m) for m in ['tfidf','bm25','lsa','hybrid']}
    print(json.dumps(dict(query=query,results=result),ensure_ascii=False,indent=2));return result


def filter_demo(index):
    query='ERR_BILL_042 refund payment reference secret'
    public=index.search(query,'bm25',3)
    expanded=index.search(query,'bm25',3,{'public','team-blue'})
    assert all(r['id']!='D18' for r in public)
    assert any(r['id']=='D18' for r in expanded)
    assert index.search(query,'hybrid',3,set())==[]
    result=dict(public=public,with_team_blue=expanded)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','search','filter','eval','all'],default='hand')
    parser.add_argument('--query',default='ERR_BILL_042')
    parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a new output directory')
    report={}
    if args.mode in ('hand','all'):report['hand']=hand_demo()
    if args.mode in ('search','filter','eval','all'):
        index=SearchIndex(documents())
        if args.mode in ('search','all'):report['search']=search_demo(index,args.query)
        if args.mode in ('filter','all'):report['filter']=filter_demo(index)
        if args.mode in ('eval','all'):report['evaluation']=evaluate(index)
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        if args.mode in ('search','filter','eval','all'):index.save(args.out/'index')
        print('Output:',args.out)
if __name__=='__main__':main()
