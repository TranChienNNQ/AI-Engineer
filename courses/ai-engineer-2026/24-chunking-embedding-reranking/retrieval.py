"""Readable reference components: span-preserving chunks, neural retrieval and reranking."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PATTERN = re.compile(r'\w+(?:[-./]\w+)*')
ANALYZER = 'nfc-casefold-word-compound-v1'


def terms(text):
    return PATTERN.findall(unicodedata.normalize('NFC', text).casefold())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def windows(length, size, overlap):
    if type(length) is not int or length < 0 or type(size) is not int or type(overlap) is not int or size < 1 or not 0 <= overlap < size:
        raise ValueError('Require integer size > 0 and 0 <= overlap < size')
    result = []
    start = 0
    while start < length:
        end = min(start + size, length)
        result.append((start, end))
        if end == length:
            break  # Do not generate redundant short tails after already covering the end.
        start += size - overlap
    return result


def chunk_sources(sources, strategy='section', size=48, overlap=0, scopes=None):
    if strategy not in {'fixed', 'section'}:
        raise ValueError('Unknown chunk strategy')
    windows(0, size, overlap)  # Validate even when source collection is empty.
    scopes = {'public'} if scopes is None else set(scopes)
    chunks = []
    for source in sources:
        if source['scope'] not in scopes:
            continue  # Filter before vocabulary/statistics/representation fitting.
        text = source['text']
        if unicodedata.normalize('NFC', text) != text:
            raise ValueError('Canonicalize source to NFC before annotating offsets')
        spans = [(m.start(), m.end()) for m in PATTERN.finditer(text)]
        regions = source['sections'] if strategy == 'section' else [dict(start=0, end=len(text), title='Toàn văn')]
        for region in regions:
            selected = [(a,b) for a,b in spans if a >= region['start'] and b <= region['end']]
            for left, right in windows(len(selected), size, overlap):
                a, b = selected[left][0], selected[right-1][1]
                # Extend only to region ends when all boundary tokens are present.
                if left == 0:
                    a = region['start']
                if right == len(selected):
                    b = region['end']
                identity = [source['id'], source['version'], strategy, size, overlap, a, b]
                chunks.append(dict(id='C-'+digest(identity)[:12], source_id=source['id'],
                                   version=source['version'], scope=source['scope'], title=source['title'],
                                   heading=region['title'], start=a, end=b, text=text[a:b]))
    return chunks


def model_text(chunk):
    return chunk['title'] + ' / ' + chunk['heading'] + ' / ' + chunk['text']


def relevant_ids(query, chunks):
    e = query['evidence']
    if e is None:
        return set()
    return {c['id'] for c in chunks if c['source_id'] == e['source_id']
            and c['start'] <= e['start'] and c['end'] >= e['end']}


def evidence_metrics(query, chunks, ranked_ids, k=3):
    if query['evidence'] is None:
        raise ValueError('Evaluate no-answer separately')
    rel = relevant_ids(query, chunks)
    byid = {c['id']:c for c in chunks}
    hit = any(i in rel for i in ranked_ids[:k])
    rr = next((1/r for r,i in enumerate(ranked_ids[:k],1) if i in rel), 0.)
    source_hit = any(byid[i]['source_id'] == query['evidence']['source_id'] for i in ranked_ids[:k])
    return dict(coverage=float(bool(rel)), hit=float(hit), rr=rr, source_hit=float(source_hit))


class Vocabulary:
    def __init__(self, texts=None, words=None):
        self.words = list(words) if words is not None else ['<PAD>','<UNK>','<CLS>','<SEP>'] + sorted({t for text in texts for t in terms(text)})
        if self.words[:4] != ['<PAD>','<UNK>','<CLS>','<SEP>'] or len(set(self.words)) != len(self.words):
            raise ValueError('Invalid vocabulary contract')
        self.ids = {t:i for i,t in enumerate(self.words)}

    def encode(self, text, limit):
        ids = [self.ids.get(t,1) for t in terms(text)] or [1]
        if len(ids) > limit:
            raise ValueError(f'{len(ids)} tokens exceed budget {limit}; rechunk or shorten explicitly')
        return ids

    def known(self, text):
        return any(t in self.ids for t in terms(text))


def padded(sequences):
    size = max(map(len,sequences))
    ids = torch.zeros((len(sequences),size),dtype=torch.long)
    for i,seq in enumerate(sequences):
        ids[i,:len(seq)] = torch.tensor(seq)
    return ids


def text_batch(vocab, texts, limit):
    return padded([vocab.encode(text,limit) for text in texts])


def pair_batch(vocab, pairs, query_limit=32, doc_limit=64, pair_limit=112):
    sequences, types = [], []
    for query, doc in pairs:
        q, d = vocab.encode(query,query_limit), vocab.encode(doc,doc_limit)
        seq = [2] + q + [3] + d + [3]
        if len(seq) > pair_limit:
            raise ValueError('Query + document + special tokens exceed pair budget')
        sequences.append(seq)
        types.append([0]*(len(q)+2) + [1]*(len(d)+1))
    return padded(sequences), padded(types)


class BiEncoder(nn.Module):
    """Shared non-contextual word embeddings + masked mean + learned projection."""
    def __init__(self, vocab_size, dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,dim,padding_idx=0)
        self.projection = nn.Linear(dim,dim)

    def forward(self, ids):
        mask = ids.ne(0).unsqueeze(-1)
        pooled = (self.embedding(ids)*mask).sum(1)/mask.sum(1).clamp_min(1)
        return F.normalize(torch.tanh(self.projection(pooled)), dim=-1)


class CrossEncoder(nn.Module):
    """One bidirectional Transformer layer sees the joint query/document sequence."""
    def __init__(self, vocab_size, dim=32, pair_limit=112):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,dim,padding_idx=0)
        self.position = nn.Embedding(pair_limit,dim)
        self.segment = nn.Embedding(2,dim)
        layer = nn.TransformerEncoderLayer(dim,4,dim_feedforward=64,dropout=.1,batch_first=True)
        self.encoder = nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.head = nn.Linear(dim,1)

    def forward(self, ids, types):
        valid = ids.ne(0)
        pos = torch.arange(ids.shape[1]).unsqueeze(0)
        x = self.embedding(ids)+self.position(pos)+self.segment(types)
        x = self.encoder(x,src_key_padding_mask=~valid)
        # Pool contextual outputs; padded queries are explicitly excluded from pooling.
        pooled = (x*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True)
        return self.head(pooled).squeeze(-1)  # A logit, not a calibrated probability.


@dataclass(frozen=True)
class Config:
    dim:int=32
    query_limit:int=32
    doc_limit:int=64
    pair_limit:int=112
    candidates:int=6
    top_k:int=3
    rrf_constant:int=60
    temperature:float=.2
    bi_epochs:int=80
    cross_epochs:int=30
    seed:int=24

    def __post_init__(self):
        positive = [self.dim,self.query_limit,self.doc_limit,self.pair_limit,self.candidates,
                    self.top_k,self.rrf_constant,self.bi_epochs,self.cross_epochs]
        if any(type(v) is not int or v < 1 for v in positive):
            raise ValueError('Positive integer configuration required')
        if self.dim % 4 or self.top_k > self.candidates or self.temperature <= 0:
            raise ValueError('Invalid dimension, k/M or temperature')
        if self.query_limit + self.doc_limit + 3 > self.pair_limit:
            raise ValueError('Pair budget cannot hold configured query/document budgets')


class BM25:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokens = [Counter(terms(model_text(c))) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.tokens]
        self.avg = sum(self.lengths)/len(chunks)
        self.df = Counter(t for c in self.tokens for t in c)

    def scores(self, query):
        n = len(self.chunks)
        scores = []
        for c,length in zip(self.tokens,self.lengths):
            score = 0.
            for term in set(terms(query)):
                if c[term]:
                    idf = math.log(1+(n-self.df[term]+.5)/(self.df[term]+.5))
                    score += idf*c[term]*2.2/(c[term]+1.2*(.25+.75*length/self.avg))
            scores.append(score)
        return np.array(scores)


def fuse(rankings, constant=60):
    scores = defaultdict(float)
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError('Duplicate ID within ranking')
        for rank,doc_id in enumerate(ranking,1):
            scores[doc_id] += 1/(constant+rank)
    return sorted(scores,key=lambda i:(-scores[i],i))


class Pipeline:
    def __init__(self, chunks, vocab, bi, cross, cfg=Config()):
        if not chunks or len({c['id'] for c in chunks}) != len(chunks):
            raise ValueError('Need nonempty chunks with unique IDs')
        self.chunks,self.vocab,self.bi,self.cross,self.cfg = chunks,vocab,bi,cross,cfg
        self.byid = {c['id']:c for c in chunks}
        self.lexical = BM25(chunks)
        self.doc_ids = text_batch(vocab,[model_text(c) for c in chunks],cfg.doc_limit)
        self.refresh()

    def refresh(self):
        self.bi.eval();self.cross.eval()
        with torch.inference_mode():
            self.vectors = self.bi(self.doc_ids).numpy().copy()

    def retrieve(self, query, method='hybrid', m=None):
        m = self.cfg.candidates if m is None else m
        if type(m) is not int or m < 1:
            raise ValueError('Positive integer candidate count required')
        if not self.vocab.known(query):
            return []  # Explicit OOV gate for this word-level toy model.
        if method == 'hybrid':
            return fuse([self.retrieve(query,'bm25',m),self.retrieve(query,'dense',m)],self.cfg.rrf_constant)[:m]
        if method == 'bm25':
            scores = self.lexical.scores(query)
        elif method == 'dense':
            self.bi.eval()
            with torch.inference_mode():
                q = self.bi(text_batch(self.vocab,[query],self.cfg.query_limit)).numpy()[0]
            scores = self.vectors@q
        else:
            raise ValueError('Unknown method')
        order = sorted(range(len(scores)),key=lambda i:(-float(scores[i]),self.chunks[i]['id']))
        return [self.chunks[i]['id'] for i in order if scores[i]>1e-12][:m]

    def pair_scores(self, query, ids):
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate candidate')
        if not ids:
            return np.array([])
        pairs = [(query,model_text(self.byid[i])) for i in ids]
        x,types = pair_batch(self.vocab,pairs,self.cfg.query_limit,self.cfg.doc_limit,self.cfg.pair_limit)
        self.cross.eval()
        with torch.inference_mode():
            return self.cross(x,types).numpy()

    def rerank(self, query, ids):
        scores = self.pair_scores(query,ids)
        return [ids[i] for i in sorted(range(len(ids)),key=lambda j:(-float(scores[j]),ids[j]))]

    def save(self, path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        torch.save(dict(bi=self.bi.state_dict(),cross=self.cross.state_dict()),path/'models.pt')
        np.save(path/'vectors.npy',self.vectors,allow_pickle=False)
        obj=dict(config=asdict(self.cfg),chunks=self.chunks,vocabulary=self.vocab.words,
                 analyzer=ANALYZER,chunk_hash=digest(self.chunks),
                 weights_sha256=hashlib.sha256((path/'models.pt').read_bytes()).hexdigest())
        (path/'manifest.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))

    @classmethod
    def load(cls,path):
        path=Path(path);obj=json.loads((path/'manifest.json').read_text())
        if obj['analyzer']!=ANALYZER or obj['chunk_hash']!=digest(obj['chunks']):
            raise ValueError('Manifest mismatch')
        if obj['weights_sha256']!=hashlib.sha256((path/'models.pt').read_bytes()).hexdigest():
            raise ValueError('Model revision mismatch')
        cfg=Config(**obj['config']);vocab=Vocabulary(words=obj['vocabulary'])
        bi=BiEncoder(len(vocab.words),cfg.dim)
        cross=CrossEncoder(len(vocab.words),cfg.dim,cfg.pair_limit)
        states=torch.load(path/'models.pt',map_location='cpu',weights_only=True)
        bi.load_state_dict(states['bi']);cross.load_state_dict(states['cross'])
        result=cls(obj['chunks'],vocab,bi,cross,cfg)
        cached=np.load(path/'vectors.npy',allow_pickle=False)
        if cached.shape!=result.vectors.shape:
            raise ValueError('Cached vector shape mismatch')
        np.testing.assert_allclose(cached,result.vectors,rtol=1e-5,atol=1e-6)
        result.vectors=cached
        return result
