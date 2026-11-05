"""Scripted candidates to isolate citation errors, NOT measured outputs of an LLM."""
import copy
import json
from data import dataset
from rag import Index,source_key,validate_candidate
from evaluation import evaluate_candidate,aggregate


def setup(fids,as_of='2026-11-04',include_restricted=False):
    sources,facts,_=dataset()
    index=Index(sources,as_of,{'public','team-blue'} if include_restricted else None)
    context=[];seen=set()
    for fid in fids:
        for c in index.chunks:
            if any(s['source_id']==c['source_id'] and s['version']==c['version'] and c['start']<=s['start'] and s['end']<=c['end'] for s in facts[fid]['spans']):
                if c['chunk_id'] not in seen:
                    context.append(dict(c,evidence_id=f'E{len(context)+1}'));seen.add(c['chunk_id'])
                break
        else:raise ValueError('Fixture fact unavailable: '+fid)
    return context,index.sources,facts


def citation(fid,context,facts):
    for e in context:
        for s in facts[fid]['spans']:
            if s['source_id']==e['source_id'] and s['version']==e['version'] and e['start']<=s['start'] and s['end']<=e['end']:
                a,b=s['start']-e['start'],s['end']-e['start']
                return dict(evidence_id=e['evidence_id'],start=a,end=b,quote=e['text'][a:b])
    raise ValueError('No fixture evidence for '+fid)


def fixture_cases():
    sources,facts,_=dataset();cases=[]
    def add(name,context,claims,required=('key_72',),status='answer',expected='answer',as_of='2026-11-04',raw=None,reason=''):
        query=dict(text='Fixture '+name,required=list(required),expected_status=expected,as_of=as_of)
        obj=dict(status=status,reason=reason,claims=claims)
        cases.append(dict(name=name,context=copy.deepcopy(context),query=query,
                          raw=raw if raw is not None else json.dumps(obj,ensure_ascii=False)))
    def claim(fid,ctx,cite_fid=None):
        return dict(text=facts[fid]['text'],citations=[citation(cite_fid or fid,ctx,facts)])
    ctx,_,_=setup(['key_72','no_permission','cache_ttl','deadline'])
    key=claim('key_72',ctx)
    add('valid',ctx,[key])
    add('derived_72h_to_3days',ctx,[dict(text='Idempotency key được lưu 3 ngày theo policy phiên bản 2.',citations=[citation('key_72',ctx,facts)])])
    add('real_quote_wrong_claim',ctx,[claim('key_72',ctx,'deadline')])
    obj=copy.deepcopy(key);obj['citations'][0]['evidence_id']='E999'
    add('unknown_context_id',ctx,[obj])
    obj=copy.deepcopy(key);obj['citations'][0]['quote']='Một câu trích không tồn tại.'
    add('fabricated_quote',ctx,[obj])
    obj=copy.deepcopy(key);obj['citations'][0]['end']=99999
    add('invalid_span',ctx,[obj])
    add('missing_citation',ctx,[dict(text=facts['key_72']['text'],citations=[])])
    obj=dict(status='answer',reason='',claims=[key],url='https://untrusted.invalid/')
    add('extra_url_field',ctx,[],raw=json.dumps(obj))
    obj=copy.deepcopy(key);obj['citations']*=2
    add('duplicate_citation',ctx,[obj])
    obj=claim('no_permission',ctx);obj['citations'].append(citation('cache_ttl',ctx,facts))
    hallucinated=dict(text='Mọi lỗi đều có thể retry vô hạn.',citations=[citation('key_72',ctx,facts)])
    add('hand_three_claims_four_edges',ctx,[key,obj,hallucinated],required=('key_72','no_permission'))
    composite='Key được lưu 72 giờ và mọi attempt retry chia sẻ cùng tổng deadline.'
    add('joint_support_complete',ctx,[dict(text=composite,citations=[citation('key_72',ctx,facts),citation('deadline',ctx,facts)])],required=('key_72','deadline'))
    add('joint_support_incomplete',ctx,[dict(text=composite,citations=[citation('key_72',ctx,facts)])],required=('key_72','deadline'))
    add('honest_partial',ctx,[key],required=('key_72','deadline'),status='partial')
    add('correct_abstain',[],[],required=(),status='abstain',expected='abstain',reason='insufficient_evidence')
    add('unnecessary_abstain',ctx,[],status='abstain',reason='insufficient_evidence')
    conflict,_,_=setup(['upload_20','upload_25'])
    add('conflict_disclosed',conflict,[claim('upload_20',conflict),claim('upload_25',conflict)],required=('upload_20','upload_25'),status='conflict',expected='conflict')
    add('conflict_hidden',conflict,[claim('upload_20',conflict)],required=('upload_20','upload_25'),expected='conflict')
    add('faithful_but_irrelevant',ctx,[claim('cache_ttl',ctx)],required=(),expected='abstain')
    old,_,_=setup(['key_24'],as_of='2026-06-01')
    add('stale_for_current_request',old,[claim('key_24',old)])
    private,_,_=setup(['secret'],include_restricted=True)
    add('restricted_context',private,[claim('secret',private)])
    add('invalid_json',ctx,[],raw='{not-json')
    obj=copy.deepcopy(key);obj['citations'][0]['start']=True
    add('boolean_offset',ctx,[obj])
    return cases


def evaluate_fixtures():
    sources,facts,_=dataset();rows=[]
    for case in fixture_cases():
        public=Index(sources,case['query']['as_of']).sources
        validation=validate_candidate(case['raw'],case['context'],public)
        semantic=evaluate_candidate(case['query'],validation,facts)
        rows.append(dict(name=case['name'],validation=validation,semantic=semantic))
    return dict(kind='22 scripted candidates, no LLM sampled',aggregate=aggregate(rows),rows=rows)
