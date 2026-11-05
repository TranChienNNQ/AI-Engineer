"""Offline RAG engineering lab: retrieve, pack, extract, cite, validate and evaluate."""
import argparse
from dataclasses import asdict,replace
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re

from data import dataset
from rag import Config,Index,SCHEMA,PROMPT_VERSION,digest,run,source_key
from evaluation import aggregate,coverage,evaluate_candidate
from fixtures import evaluate_fixtures


def evaluate_queries(cfg=Config()):
    sources,facts,queries=dataset();rows=[]
    for q in queries:
        result=run(q['text'],sources,q['as_of'],cfg=cfg)
        index=Index(sources,q['as_of'])
        row=dict(query=q,validation=result['validation'],semantic=evaluate_candidate(q,result['validation'],facts),
                 pool=coverage(q,index.chunks,facts),candidate=coverage(q,result['candidates'],facts),
                 packed=coverage(q,result['context'],facts),packing=result['packing'],
                 candidate_ids=[c['chunk_id'] for c in result['candidates']],
                 context_ids=[c['chunk_id'] for c in result['context']],
                 provider='extractive lexical baseline, not an LLM')
        rows.append(row)
    splits={}
    for split in ['dev','test']:
        selected=[r for r in rows if r['query']['split']==split]
        metrics=aggregate(selected)
        answerable=[r for r in selected if r['query']['required']]
        metrics['answerable_queries']=len(answerable)
        for stage in ['pool','candidate','packed']:
            metrics[stage+'_required_recall']=sum(r[stage]['required_recall'] for r in answerable)/len(answerable)
            metrics[stage+'_complete']=sum(r[stage]['complete'] for r in answerable)/len(answerable)
        no_answer=[r for r in selected if not r['query']['required']]
        metrics['no_answer']=dict(cases=len(no_answer),answered=sum(bool(r['validation']['candidate']['claims']) for r in no_answer))
        splits[split]=metrics
    return dict(config=asdict(cfg),splits=splits,rows=rows)


def hand_demo(fixtures=None):
    fixture_report=fixtures or evaluate_fixtures()
    row=next(r for r in fixture_report['rows'] if r['name']=='hand_three_claims_four_edges')
    s=row['semantic']
    result=dict(rag_sequence_mixture=.6*.7+.3*.2+.1*.1,
                context_budget=512-96-24-64-96,
                greedy_costs=[150,120,112],greedy_selected=[0],alternative_selected=[1,2],
                citation_attachment=s['attached_claims']/s['claim_count'],
                grounded_claim_rate=s['supported_claims']/s['claim_count'],
                citation_precision=s['supporting_citations']/s['citation_count'],
                required_fact_recall=s['required_fact_recall'],mechanically_accepted=row['validation']['accepted'],
                grounded_task_success=s['task_success'])
    assert result['context_budget']==232
    assert abs(result['rag_sequence_mixture']-.49)<1e-12
    assert result['citation_precision']==.5 and result['grounded_claim_rate']==2/3
    return result


def build_report():
    sources,facts,queries=dataset();cfg=Config();fixtures=evaluate_fixtures()
    primary=evaluate_queries(cfg)
    ablation=[]
    for budget in [40,110,180]:
        result=primary if budget==cfg.context_tokens else evaluate_queries(replace(cfg,context_tokens=budget))
        ablation.append(dict(context_budget=budget,metrics=result['splits']['test']))
    return dict(kind='offline retrieval + extractive baseline + closed annotated citation checks; no LLM/API sampled',
                source_snapshots=len(sources),logical_sources=len({s['source_id'] for s in sources}),
                current_public_sources=len(Index(sources,'2026-11-04').sources),
                current_public_chunks=len(Index(sources,'2026-11-04').chunks),facts=len(facts),queries=len(queries),
                corpus_hash=digest(sources),queries_hash=digest(queries),
                metadata=dict(python=platform.python_version(),jsonschema=version('jsonschema'),device='CPU',
                              prompt_version=PROMPT_VERSION,schema_hash=digest(SCHEMA)),
                primary=primary,packing_ablation=ablation,fixtures=fixtures,hand=hand_demo(fixtures))


def write_request(path,result,sources):
    path.mkdir(parents=True,exist_ok=False)
    (path/'sources').mkdir()
    bykey={source_key(s):s for s in sources}
    for e in result['context']:
        s=bykey[source_key(e)]
        file=path/'sources'/f'{s["source_id"]}-{s["version"]}.md'
        # Local snapshots contain source data; display as a literal block.
        fence='`'*max(4,max((len(m.group()) for m in re.finditer(r'`+',s['text'])),default=0)+1)
        text=f'# {s["source_id"]}@{s["version"]}\n\n'+fence+'text\n'+s['text']+'\n'+fence+'\n'
        file.write_text(text)
    (path/'answer.md').write_text(result['answer_markdown'])
    (path/'prompt.txt').write_text(result['prompt'])
    (path/'candidate.txt').write_text(result['raw'])
    (path/'trace.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','ask','eval','fixtures','all'],default='hand')
    parser.add_argument('--question',default='idempotency key lưu bao nhiêu giờ và cùng key khác payload xử lý thế nào')
    parser.add_argument('--as-of',default='2026-11-04');parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a new output directory')
    if args.mode=='hand':print(json.dumps(hand_demo(),ensure_ascii=False,indent=2));return
    if args.mode=='fixtures':print(json.dumps(evaluate_fixtures(),ensure_ascii=False,indent=2));return
    if args.mode in ['eval','all']:
        report=build_report()
        print(json.dumps(dict(primary=report['primary']['splits'],fixtures=report['fixtures']['aggregate'],hand=report['hand']),ensure_ascii=False,indent=2))
        if args.out:
            args.out.mkdir(parents=True,exist_ok=False)
            (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            (args.out/'answer-schema.json').write_text(json.dumps(SCHEMA,ensure_ascii=False,indent=2))
        if args.mode=='all' and args.out:
            sources,_,queries=dataset()
            for q in queries:
                result=run(q['text'],sources,q['as_of'])
                write_request(args.out/'requests'/q['id'],result,sources)
        return
    if args.out is None:parser.error('--out required for ask so citation snapshots can be opened')
    sources,_,_=dataset();result=run(args.question,sources,args.as_of)
    write_request(args.out,result,sources)
    print(result['answer_markdown']);print('Output:',args.out)

if __name__=='__main__':main()
