"""Ratings are observations joined by content hash; missing ratings remain unknown."""
import json,math,random
from collections import Counter,defaultdict
from pydantic import BaseModel,ConfigDict,Field,model_validator
from dataset import RUBRIC,GATE
from bridge46 import digest

class Strict(BaseModel):model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
class Rating(Strict):
    item_id:str=Field(min_length=1)
    content_hash:str=Field(min_length=64,max_length=64)
    rubric:str
    support:int=Field(ge=0,le=2)
    completeness:int=Field(ge=0,le=2)
    relevance:int=Field(ge=0,le=2)
    draft:int|None=Field(ge=0,le=2)
    critical:bool
    reason:str=Field(min_length=1,max_length=2000)
    provenance:str=Field(min_length=1,max_length=100)
    @model_validator(mode='after')
    def rubric_version(self):
        if self.rubric!=RUBRIC:raise ValueError('rubric mismatch')
        return self

def content(record):
    # Principal/case labels/config are deliberately omitted from the blind content packet.
    return dict(query=record['query'],action=record['action'],evidence=record['evidence'],
        answer=record['answer'],citations=record['citations'],draft=record['draft'])

def join_ratings(records,raw):
    index={r['item_id']:r for r in records}
    if len(index)!=len(records):raise ValueError('duplicate record ID')
    joined={}
    for obj in raw:
        rating=Rating.model_validate(obj).model_dump();id_=rating['item_id']
        if id_ in joined or id_ not in index:raise ValueError('duplicate or unknown rating ID')
        record=index[id_]
        if not record['served']:raise ValueError('rate served candidates only')
        if rating['content_hash']!=digest(content(record)):raise ValueError('stale content rating')
        if (rating['draft'] is None)!=(record['action']=='ask'):raise ValueError('draft score applicability')
        joined[id_]=rating
    return joined

def ratio(a,b):return a/b if b else None

def summarize(records,ratings):
    n=len(records);served=[r for r in records if r['served']]
    rated=[r for r in served if r['item_id'] in ratings]
    success=critical=answered_good=correct_abstain=0;labels=Counter();mean_dims={}
    for r in records:
        if not r['served']:
            good=r['expected']=='abstain' and r['state']=='abstained' and r['code']=='insufficient_evidence'
            correct_abstain+=good;success+=good
            labels['correct_abstain' if good else 'missed_answer_or_integration_failure']+=1
            continue
        a=ratings.get(r['item_id'])
        if a is None:labels['unrated']+=1;continue
        bad=a['critical'] or r['expected']!='serve'
        critical+=bad
        ok=not bad and all(a[k]==2 for k in ['support','completeness','relevance']) and (r['action']=='ask' or a['draft']==2)
        success+=ok;answered_good+=ok
        for k in ['support','completeness','relevance']:
            if a[k]<2:labels[k]+=1
        if r['action']!='ask' and a['draft']<2:labels['draft']+=1
        if bad:labels['critical']+=1
    for k in ['support','completeness','relevance','draft']:
        values=[ratings[r['item_id']][k] for r in rated if ratings[r['item_id']][k] is not None]
        mean_dims[k]=ratio(sum(values),len(values))
    return dict(rows=n,served=len(served),rated=len(rated),unrated=len(served)-len(rated),
        coverage=ratio(len(served),n),rating_coverage=ratio(len(rated),len(served)),
        success_count=success,task_success=ratio(success,n),quality_conditional=ratio(answered_good,len(rated)),
        correct_abstain=correct_abstain,critical=critical,critical_rate_rated=ratio(critical,len(rated)),
        dimensions=mean_dims,errors=dict(sorted(labels.items())),
        all_eligible_rated=len(rated)==len(served),score_scope='task_success is a conservative observed-success fraction when any served ratings are missing; author annotations on scripted fixtures; missing served ratings never count as success')

def gate(summary):
    reasons=[]
    if not summary['all_eligible_rated']:reasons.append('missing_ratings')
    if summary['served']<GATE['min_served']:reasons.append('small_served_sample')
    if summary['task_success'] is None or summary['task_success']<GATE['min_task_success']:reasons.append('task_success')
    if summary['coverage'] is None or summary['coverage']<GATE['min_coverage']:reasons.append('coverage')
    if summary['critical']>GATE['max_critical']:reasons.append('critical')
    return dict(eligible=not reasons,reasons=reasons,scope='toy offline gate, not production approval')

def select_dev(summaries):
    eligible=[(name,s) for name,s in summaries.items() if gate(s)['eligible']]
    # Fixed deterministic tie-break, selection sees dev summaries only.
    eligible.sort(key=lambda item:(-item[1]['task_success'],-item[1]['coverage'],item[0]))
    return eligible[0][0] if eligible else None

def grouped_success(records,ratings):
    groups=defaultdict(list)
    for r in records:groups[r['group']].append(r)
    return {g:summarize(rows,ratings)['task_success'] for g,rows in sorted(groups.items())}

def paired_cluster_bootstrap(left,right,draws=2000,seed=47):
    if set(left)!=set(right) or not left:raise ValueError('paired groups required')
    if type(draws) is not int or draws<100:raise ValueError('at least 100 draws')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for d in [left,right] for v in d.values()):raise ValueError('finite scores required')
    keys=sorted(left);diffs=[right[k]-left[k] for k in keys];rng=random.Random(seed);samples=[]
    for _ in range(draws):samples.append(sum(rng.choice(diffs) for _ in keys)/len(keys))
    samples.sort();return dict(delta=sum(diffs)/len(diffs),low=samples[int(.025*(draws-1))],high=samples[int(.975*(draws-1))],groups=len(keys),draws=draws,seed=seed,
        weighting='equal group mean; right minus left',scope='descriptive toy bootstrap; four authored test groups cannot establish generalization')

def wilson(k,n,z=1.96):
    if type(k) is not int or type(n) is not int or not 0<=k<=n or n<=0:raise ValueError('valid binomial counts required')
    if not math.isfinite(z) or z<=0:raise ValueError('positive finite z')
    p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def agreement(truth,predicted):
    if len(truth)!=len(predicted) or not truth:raise ValueError('paired nonempty labels required')
    if any(type(x) is not bool for x in truth+predicted):raise ValueError('boolean labels required')
    tp=sum(t and p for t,p in zip(truth,predicted));fp=sum(not t and p for t,p in zip(truth,predicted));fn=sum(t and not p for t,p in zip(truth,predicted));tn=len(truth)-tp-fp-fn
    n=len(truth);po=(tp+tn)/n;pe=((tp+fn)*(tp+fp)+(tn+fp)*(tn+fn))/(n*n)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,agreement=po,kappa=(po-pe)/(1-pe) if pe<1 else None,bad_pass_rate=ratio(fp,fp+tn))

def swap_result(first,swapped):
    if first not in ['A','B','tie'] or swapped not in ['A','B','tie']:raise ValueError('A/B/tie required')
    mapped={'A':'B','B':'A','tie':'tie'}[swapped]
    return dict(consistent=first==mapped,verdict=first if first==mapped else 'needs_review')

def load_jsonl(path):
    from pathlib import Path
    p=Path(path)
    if p.stat().st_size>5_000_000:raise ValueError('input file too large for teaching lab')
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    return [json.loads(line,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON'))) for line in p.read_text().splitlines() if line.strip()]
