import hashlib,json
from pathlib import Path
from bridge46 import digest,source_hash,documents
from dataset import cases,validate_split,CONFIGS,REPEATS,VERSION,RUBRIC,RUBRIC_SPEC,GATE
from capture import capture,author_rating,Record
from scoring import content,join_ratings,summarize,gate,select_dev,grouped_success,paired_cluster_bootstrap,agreement,swap_result

def write_jsonl(path,rows):Path(path).write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows))
def implementation_hash():
    root=Path(__file__).parent
    return digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py')) if p.name!='make_figures.py'})

def validate_records(rows):
    out=[Record.model_validate(r).model_dump() for r in rows];index={c['id']:c for c in cases()};keys=set();ids=set()
    for r in out:
        if r['item_id'] in ids:raise ValueError('duplicate record ID')
        ids.add(r['item_id']);key=(r['config'],r['case_id'],r['repeat'])
        if key in keys:raise ValueError('duplicate sample')
        keys.add(key)
        c=index.get(r['case_id'])
        if c is None:raise ValueError('unknown case')
        for field in ['group','split','action','expected','expected_source','slice']:
            if r[field]!=c[field]:raise ValueError('case metadata mismatch')
        if r['query']!=c['query']:raise ValueError('case query mismatch')
    required={(cfg,c['id'],rep) for cfg in CONFIGS for c in cases() for rep in range(REPEATS)}
    if keys!=required:raise ValueError('complete paired panel required; do not drop failures/repeats')
    return out

def analyze(records,ratings):
    records=validate_records(records);joined=join_ratings(records,ratings);summary={};slices={};groups={}
    for cfg in CONFIGS:
        summary[cfg]={};slices[cfg]={};groups[cfg]={}
        for split in ['dev','test']:
            rows=[r for r in records if r['config']==cfg and r['split']==split]
            summary[cfg][split]=summarize(rows,joined)
            groups[cfg][split]=grouped_success(rows,joined)
            slices[cfg][split]={tag:summarize([r for r in rows if r['slice']==tag],joined) for tag in sorted({r['slice'] for r in rows})}
    chosen=select_dev({cfg:s['dev'] for cfg,s in summary.items()})
    left,right=list(CONFIGS);paired=paired_cluster_bootstrap(groups[left]['test'],groups[right]['test'])
    served=[r for r in records if r['served'] and r['item_id'] in joined]
    truth=[not joined[r['item_id']]['critical'] and all(joined[r['item_id']][k]==2 for k in ['support','completeness','relevance']) and (r['action']=='ask' or joined[r['item_id']]['draft']==2) for r in served]
    # Deliberately weak judge: citation exists => pass. Not an LLM or factuality check.
    proxy=[bool(r['citations']) for r in served]
    retrieval={};repeat_counts={};codes={}
    from collections import Counter
    for cfg in CONFIGS:
        selected=[r for r in records if r['config']==cfg]
        positive=[r for r in selected if r['expected_source'] is not None]
        retrieval[cfg]=dict(eligible_rows=len(positive),expected_source_hits=sum(any(e['doc_id']==r['expected_source'] for e in r['evidence']) for r in positive),scope='known-source fixture hit, not a ranking benchmark')
        repeat_counts[cfg]={c['id']:sum(summarize([r],joined)['success_count'] for r in selected if r['case_id']==c['id']) for c in cases()}
        codes[cfg]=dict(sorted(Counter(r['code'] for r in selected).items()))
    ledger={}
    for cfg in CONFIGS:
        selected=[r for r in records if r['config']==cfg];meta=[r['usage'] for r in selected if r['usage']]
        ledger[cfg]=dict(rows=len(selected),transport_calls=sum(r['transport_calls'] for r in selected),
            known_input_tokens=sum(m['known_input_tokens'] for m in meta),known_output_tokens=sum(m['known_output_tokens'] for m in meta),
            unknown_usage_attempts=sum(m['unknown_usage_attempts'] for m in meta),usage_source='scripted_fixture_not_bill',
            recorded_latency_scope='fake clock; no latency benchmark')
    return dict(dataset=validate_split(cases()),rubric=RUBRIC,gate_rule=GATE,summary=summary,slices=slices,
        dev_gate={cfg:gate(s['dev']) for cfg,s in summary.items()},selected_on_dev=chosen,
        selected_test=None if chosen is None else summary[chosen]['test'],
        test_pair_diagnostic=paired,groups=groups,
        judge_audit=dict(proxy='citation-present programmed judge, not LLM',vs_author=agreement(truth,proxy),
            author_provenance='scripted labels, not independent human study',swap_examples=[swap_result('A','B'),swap_result('A','A'),swap_result('tie','tie')]),
        retrieval=retrieval,repeat_success_counts=repeat_counts,outcome_codes=codes,ledger=ledger,records=len(records),ratings=len(joined),live_model_calls=0,
        interpretation='offline evaluation harness, scripted generation and authored ratings; no measured model quality, judge quality, billing or production approval')

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);data=cases();validate_split(data)
    records=[capture(out/'db',case,cfg,rep) for cfg in CONFIGS for case in data for rep in range(REPEATS)]
    ratings=[a for row in records if (a:=author_rating(row)) is not None]
    report=analyze(records,ratings)
    manifest=dict(dataset=VERSION,dataset_hash=digest(data),rubric=RUBRIC,rubric_spec=RUBRIC_SPEC,rubric_hash=digest(dict(spec=RUBRIC_SPEC,gate=GATE)),
        configs=CONFIGS,repeats=REPEATS,base_source=source_hash(),implementation=implementation_hash(),
        corpus_hash=digest(documents()),records_hash=digest(records),ratings_hash=digest(ratings),
        annotation_provenance='author-scripted',generator='HTTPX MockTransport through lesson46 SupportEngine',live_model_calls=0)
    report['manifest']=manifest;report['experiment_id']=digest(manifest)[:16]
    # Blind queue hides config/repeat/variant/expected outcome; ordering uses a stable hash.
    queue=[dict(item_id=r['item_id'],content_hash=r['content_hash'],rubric=RUBRIC,packet=content(r),
        rating=dict(support=None,completeness=None,relevance=None,draft=None,critical=None,reason='',provenance='')) for r in records if r['served']]
    queue.sort(key=lambda x:digest(['blind-order-47',x['item_id']]))
    write_jsonl(out/'records.jsonl',records);write_jsonl(out/'author-ratings.jsonl',ratings);write_jsonl(out/'review-queue.jsonl',queue)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return report
