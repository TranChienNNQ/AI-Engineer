"""Point-in-time feedback join, mature-cohort metrics, slices and univariate drift proxy."""
from collections import Counter
from dataclasses import asdict
from statistics import mean
import math
from data import dataset


def joined_labels(predictions,feedback,as_of):
    if type(as_of) is not int or as_of<0:raise ValueError('nonnegative integer as_of required')
    by_key={}
    for p in predictions:
        key=(p.prediction_id,p.model_version)
        if key in by_key:raise ValueError('duplicate prediction identity/version')
        by_key[key]=p
    seen={};labels={};bad=set();reasons=Counter();duplicate=0
    for f in feedback:
        if f.available_at>as_of:continue
        key=(f.prediction_id,f.model_version)
        p=by_key.get(key)
        if p is None:reasons['unknown_prediction_or_version']+=1;continue
        if p.predicted_at>as_of or f.label_event_at<p.predicted_at:
            reasons['invalid_temporal_join']+=1;continue
        if f.feedback_id in seen:
            if f==seen[f.feedback_id]:duplicate+=1;continue
            previous=seen[f.feedback_id];bad.add((previous.prediction_id,previous.model_version));bad.add(key)
            reasons['feedback_id_conflict']+=1;continue
        seen[f.feedback_id]=f
        if key in labels:
            if labels[key]==f.label:duplicate+=1
            else:bad.add(key);reasons['label_conflict']+=1
        else:labels[key]=f.label
    for key in bad:labels.pop(key,None)
    return labels,dict(duplicates=duplicate,quarantined=dict(reasons),conflicted_predictions=len(bad))


def score_metrics(pairs):
    if not pairs:return None
    tp=fp=fn=tn=0
    for p,y in pairs:
        pred=p.score>=.5
        if pred and y==1:tp+=1
        elif pred and y==0:fp+=1
        elif not pred and y==1:fn+=1
        else:tn+=1
    n=len(pairs)
    return dict(n=n,tp=tp,fp=fp,fn=fn,tn=tn,accuracy=(tp+tn)/n,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                brier=mean((p.score-y)**2 for p,y in pairs))


def snapshot(predictions,feedback,start,end,as_of,model_version='toy-v1',lag=100,min_labels=8,min_coverage=.8):
    if not all(type(x) is int for x in [start,end,as_of,lag,min_labels]) or not 0<=start<end or lag<0 or min_labels<1:
        raise ValueError('invalid window/maturity policy')
    if not 0<=min_coverage<=1:raise ValueError('invalid coverage policy')
    labels,join=joined_labels(predictions,feedback,as_of)
    cohort=[p for p in predictions if start<=p.predicted_at<end and p.predicted_at<=as_of and p.model_version==model_version]
    pairs=[(p,labels[(p.prediction_id,p.model_version)]) for p in cohort if (p.prediction_id,p.model_version) in labels]
    n=len(cohort);coverage=len(pairs)/n if n else None;mature=as_of>=end+lag
    eligible=mature and len(pairs)>=min_labels and coverage is not None and coverage>=min_coverage
    state='READY' if eligible else ('PROVISIONAL' if not mature else 'INSUFFICIENT_LABELS')
    slices={}
    for s in ['mobile','web']:
        total=sum(p.segment==s for p in cohort);selected=[(p,y) for p,y in pairs if p.segment==s]
        slices[s]=dict(predictions=total,labeled=len(selected),label_coverage=len(selected)/total if total else None,
                       metrics=score_metrics(selected))
    alerts=[];metrics=score_metrics(pairs)
    if eligible:
        if metrics['accuracy']<.75:alerts.append(dict(kind='quality',scope='overall',code='accuracy_below_075'))
        for name,row in slices.items():
            if row['metrics'] and row['metrics']['n']>=4 and row['label_coverage']>=min_coverage and row['metrics']['accuracy']<.75:
                alerts.append(dict(kind='quality',scope=name,code='slice_accuracy_below_075'))
    if join['quarantined']:alerts.append(dict(kind='data',scope='join',code='feedback_quarantined'))
    return dict(window=[start,end],as_of=as_of,model_version=model_version,maturity_lag=lag,mature=mature,
                status=state,predictions=n,labeled=len(pairs),label_coverage=coverage,observed_metrics=metrics,
                predicted_positive_rate=mean(p.score>=.5 for p in cohort) if n else None,
                slices=slices,join=join,alerts=alerts,
                note='Observed labeled subset; maturity/coverage gate is policy, not a proof of unbiased labels')


def histogram(values):
    if not values:return None
    if any(type(v) not in {int,float} or not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('invalid histogram input')
    low=sum(v<.5 for v in values);return [low/len(values),1-low/len(values)]


def distribution_compare(reference,current):
    a=histogram([p.feature_value for p in reference]);b=histogram([p.feature_value for p in current])
    if a is None or b is None:return dict(tv=None,status='INSUFFICIENT_DATA')
    tv=.5*sum(abs(x-y) for x,y in zip(a,b))
    return dict(reference=a,current=b,bins=['[0,.5)','[.5,1]'],tv=tv,
                status='REVIEW_SHIFT' if min(len(reference),len(current))>=8 and tv>.2 else 'NO_PROXY_ALERT',
                note='Univariate 2-bin total variation proxy; not a hypothesis test or model-quality verdict')


def quality_demo():
    p,f=dataset();reference=[x for x in p if x.predicted_at<10]
    current=[x for x in p if 100<=x.predicted_at<110];shift=[x for x in p if 200<=x.predicted_at<210]
    return dict(reference=snapshot(p,f,0,10,120),early=snapshot(p,f,100,110,120),
                mature=snapshot(p,f,100,110,220),shift=snapshot(p,f,200,210,320),
                current_distribution=distribution_compare(reference,current),shift_distribution=distribution_compare(reference,shift))
