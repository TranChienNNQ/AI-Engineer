"""Train two tiny neural models, diagnose chunk boundaries, and evaluate two-stage search."""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F
from data import dataset
from retrieval import (BiEncoder, CrossEncoder, Config, Pipeline, Vocabulary, BM25,
                       chunk_sources, digest, evidence_metrics, model_text, pair_batch,
                       relevant_ids, terms, text_batch, windows)


def mean_metrics(pipeline, queries, method, rerank=False):
    rows=[]
    for q in queries:
        candidates=pipeline.retrieve(q['text'],method)
        ranked=pipeline.rerank(q['text'],candidates) if rerank else candidates
        metric=evidence_metrics(q,pipeline.chunks,ranked,pipeline.cfg.top_k)
        metric['candidate_hit']=float(any(i in relevant_ids(q,pipeline.chunks) for i in candidates))
        rows.append(dict(query_id=q['id'],section_id=q['section_id'],query=q['text'],
                         candidates=candidates,ranked=ranked,metrics=metric))
    return {name:sum(r['metrics'][name] for r in rows)/len(rows)
            for name in ['coverage','hit','rr','source_hit','candidate_hit']},rows


def train_bi(pipeline, train, dev):
    cfg=pipeline.cfg;model=pipeline.bi
    qids=text_batch(pipeline.vocab,[q['text'] for q in train],cfg.query_limit)
    chunk_ids=[c['id'] for c in pipeline.chunks]
    targets=[]
    for q in train:
        rel=relevant_ids(q,pipeline.chunks)
        if len(rel)!=1:raise ValueError('Training demo expects one positive section chunk per query')
        targets.append(chunk_ids.index(next(iter(rel))))
    targets=torch.tensor(targets)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.015,weight_decay=.01)
    history=[];best=-1.;best_state=None;best_epoch=None
    baseline=mean_metrics(pipeline,dev,'dense')[0]
    for epoch in range(1,cfg.bi_epochs+1):
        model.train();optimizer.zero_grad()
        scores=model(qids)@model(pipeline.doc_ids).T/cfg.temperature
        loss=F.cross_entropy(scores,targets)
        loss.backward();optimizer.step()
        row=dict(epoch=epoch,train_loss=float(loss.detach()))
        if epoch%10==0 or epoch==cfg.bi_epochs:
            pipeline.refresh();metric=mean_metrics(pipeline,dev,'dense')[0]['rr']
            row['dev_mrr3']=metric
            if metric>best+1e-12:
                best,best_state,best_epoch=metric,copy.deepcopy(model.state_dict()),epoch
            print(f'bi epoch {epoch}: loss={row["train_loss"]:.4f}, dev MRR@3={metric:.4f}',flush=True)
        history.append(row)
    model.load_state_dict(best_state);pipeline.refresh()
    return dict(history=history,best_epoch=best_epoch,best_dev_mrr3=best,untrained_dev=baseline)


def train_cross(pipeline, train, dev):
    cfg=pipeline.cfg;model=pipeline.cross
    pairs=[];labels=[];forced=0
    # Hard negatives from a frozen hybrid retriever, mined only on TRAIN queries.
    for q in train:
        ids=pipeline.retrieve(q['text'],'hybrid')
        rel=relevant_ids(q,pipeline.chunks)
        for i in sorted(rel):
            if i not in ids:
                ids.append(i);forced+=1  # Train only. Evaluation never injects gold evidence.
        for i in ids:
            pairs.append((q['text'],model_text(pipeline.byid[i])))
            labels.append(float(i in rel))
    x,segments=pair_batch(pipeline.vocab,pairs,cfg.query_limit,cfg.doc_limit,cfg.pair_limit)
    y=torch.tensor(labels);positive=int(y.sum());negative=len(y)-positive
    pos_weight=torch.tensor(negative/positive)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    rng=torch.Generator().manual_seed(cfg.seed+1)
    history=[];best=-1.;best_state=None;best_epoch=None
    for epoch in range(1,cfg.cross_epochs+1):
        model.train();total=0.
        for batch in torch.randperm(len(y),generator=rng).split(36):
            optimizer.zero_grad();logits=model(x[batch],segments[batch])
            loss=F.binary_cross_entropy_with_logits(logits,y[batch],pos_weight=pos_weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();total+=float(loss.detach())*len(batch)
        row=dict(epoch=epoch,train_loss=total/len(y))
        if epoch%5==0 or epoch==cfg.cross_epochs:
            metric=mean_metrics(pipeline,dev,'hybrid',rerank=True)[0]['rr']
            row['dev_mrr3']=metric
            if metric>best+1e-12:
                best,best_state,best_epoch=metric,copy.deepcopy(model.state_dict()),epoch
            print(f'cross epoch {epoch}: loss={row["train_loss"]:.4f}, dev MRR@3={metric:.4f}',flush=True)
        history.append(row)
    model.load_state_dict(best_state);model.eval()
    return dict(history=history,best_epoch=best_epoch,best_dev_mrr3=best,
                pairs=len(y),positive=positive,negative=negative,forced_train_positives=forced,
                negative_miner='frozen best-bi + BM25 hybrid, train queries only')


def chunk_ablation(sources, queries):
    rows=[]
    for name,strategy,size,overlap in [('fixed24','fixed',24,0),('overlap24_8','fixed',24,8),('section48','section',48,0)]:
        chunks=chunk_sources(sources,strategy,size,overlap);bm=BM25(chunks)
        obj=dict(name=name,chunks=len(chunks),body_tokens=sum(len(terms(c['text'])) for c in chunks),
                 model_tokens=sum(len(terms(model_text(c))) for c in chunks),splits={})
        for split in ['dev','test']:
            selected=[q for q in queries if q['split']==split and q['evidence']]
            metrics=[]
            for q in selected:
                scores=bm.scores(q['text'])
                order=sorted(range(len(chunks)),key=lambda i:(-float(scores[i]),chunks[i]['id']))
                ranked=[chunks[i]['id'] for i in order if scores[i]>1e-12][:6]
                metric=evidence_metrics(q,chunks,ranked,3)
                metric['candidate_hit']=float(any(i in relevant_ids(q,chunks) for i in ranked))
                metrics.append(metric)
            obj['splits'][split]={m:sum(r[m] for r in metrics)/len(metrics) for m in metrics[0]}
        rows.append(obj)
    return rows


def hand_demo():
    boundaries=windows(13,6,2)
    assert boundaries==[(0,6),(4,10),(8,13)]
    q=np.array([1.,0.]);docs=np.array([[.8,.6],[0.,1.],[-.6,.8]])
    cosine=docs@q;logits=cosine/.2
    probs=np.exp(logits-logits.max());probs/=probs.sum()
    loss=-math.log(probs[0]);grad=(probs-np.array([1.,0.,0.]))/.2
    result=dict(windows=boundaries,total_body_tokens=sum(b-a for a,b in boundaries),
                cosine=cosine.tolist(),contrastive_probabilities=probs.tolist(),contrastive_loss=loss,
                dloss_dcosine=grad.tolist(),triplet_loss=max(0.,.2+.65-.8),
                rerank_before_rr=.5,rerank_after_rr=1.,pair_body_budget=128-17-3-9)
    print(json.dumps(result,indent=2));return result


def build_run(cfg):
    torch.set_num_threads(1);torch.manual_seed(cfg.seed)
    sources,queries=dataset()
    chunks=chunk_sources(sources,'section',48,0)
    train=[q for q in queries if q['split']=='train'];dev=[q for q in queries if q['split']=='dev']
    vocab=Vocabulary([model_text(c) for c in chunks]+[q['text'] for q in train])
    pipeline=Pipeline(chunks,vocab,BiEncoder(len(vocab.words),cfg.dim),
                      CrossEncoder(len(vocab.words),cfg.dim,cfg.pair_limit),cfg)
    started=time.perf_counter()
    bi=train_bi(pipeline,train,dev);cross=train_cross(pipeline,train,dev)
    training_seconds=time.perf_counter()-started
    report=dict(config=asdict(cfg),sources=len(sources),public_sources=sum(s['scope']=='public' for s in sources),
                chunks=len(chunks),vocab=len(vocab.words),split_counts={s:sum(q['split']==s for q in queries) for s in ['train','dev','test']},
                corpus_hash=digest(sources),queries_hash=digest(queries),chunk_hash=digest(chunks),
                metadata=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='CPU',
                              bi_parameters=sum(p.numel() for p in pipeline.bi.parameters()),
                              cross_parameters=sum(p.numel() for p in pipeline.cross.parameters()),
                              training_seconds_local=training_seconds),
                bi_training=bi,cross_training=cross,chunk_ablation=chunk_ablation(sources,queries),evaluation={})
    for split in ['dev','test']:
        selected=[q for q in queries if q['split']==split and q['evidence']]
        report['evaluation'][split]={}
        for name,method,rerank in [('bm25','bm25',False),('dense','dense',False),('hybrid','hybrid',False),('rerank','hybrid',True)]:
            aggregate,rows=mean_metrics(pipeline,selected,method,rerank)
            report['evaluation'][split][name]=dict(aggregate=aggregate,queries=len(selected),rows=rows)
    no_answer=[]
    for q in [q for q in queries if q['evidence'] is None]:
        ids=pipeline.retrieve(q['text'],'hybrid');reranked=pipeline.rerank(q['text'],ids)
        no_answer.append(dict(query_id=q['id'],query=q['text'],candidates=ids,reranked=reranked,
                              logits=pipeline.pair_scores(q['text'],ids).tolist()))
    report['no_answer']=no_answer
    return pipeline,report


def show_query(pipeline,query):
    candidates=pipeline.retrieve(query,'hybrid');reranked=pipeline.rerank(query,candidates)
    return dict(query=query,candidates=candidates,reranked=reranked,
                results=[dict(id=i,source_id=pipeline.byid[i]['source_id'],heading=pipeline.byid[i]['heading'],
                              start=pipeline.byid[i]['start'],end=pipeline.byid[i]['end'],text=pipeline.byid[i]['text'])
                         for i in reranked[:pipeline.cfg.top_k]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['hand','chunks','train','search'],default='hand')
    p.add_argument('--out',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--query',default='webhook cần raw body hay JSON serialize lại')
    args=p.parse_args();cfg=Config();torch.set_num_threads(1)
    if args.mode=='hand':
        hand_demo();return
    if args.mode=='chunks':
        sources,queries=dataset();print(json.dumps(chunk_ablation(sources,queries),ensure_ascii=False,indent=2));return
    if args.mode=='search':
        if args.bundle is None:p.error('--bundle required for search')
        pipeline=Pipeline.load(args.bundle)
        print(json.dumps(show_query(pipeline,args.query),ensure_ascii=False,indent=2));return
    if args.out is None:p.error('--out required for train')
    if args.out.exists():raise FileExistsError('Use a new output directory')
    pipeline,report=build_run(cfg);report['hand']=hand_demo()
    report['example']=show_query(pipeline,args.query)
    args.out.mkdir(parents=True,exist_ok=False)
    pipeline.save(args.out/'bundle')
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({s:{m:v['aggregate'] for m,v in report['evaluation'][s].items()} for s in ['dev','test']},indent=2))
    print('Output:',args.out)

if __name__=='__main__':main()
