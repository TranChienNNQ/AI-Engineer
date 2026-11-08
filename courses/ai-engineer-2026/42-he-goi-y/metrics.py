import math
import numpy as np


def ranking_metrics(ranking,relevant,k=5):
    ranking=list(ranking);relevant=set(relevant)
    if type(k) is not int or k<1 or not relevant or len(set(ranking))!=len(ranking):raise ValueError('k/relevance/duplicate ranking')
    top=ranking[:k];hits=[int(i in relevant) for i in top]
    dcg=sum(hit/math.log2(pos+2) for pos,hit in enumerate(hits))
    ideal=sum(1/math.log2(pos+2) for pos in range(min(k,len(relevant))))
    return dict(precision=sum(hits)/k,recall=sum(hits)/len(relevant),ndcg=dcg/ideal,
                mrr=next((1/(pos+1) for pos,hit in enumerate(hits) if hit),0.),hits=sum(hits))


def order(ids,scores):
    ids=list(ids);scores=np.asarray(scores)
    if scores.shape!=(len(ids),) or not np.isfinite(scores).all() or len(set(ids))!=len(ids):raise ValueError('finite one score per unique item required')
    return sorted(ids,key=lambda i:(-float(scores[ids.index(i)]),i))


def reciprocal_fusion(rankings,constant=60):
    if type(constant) is not int or constant<1:raise ValueError('positive RRF constant')
    scores={}
    for ranking in rankings:
        if len(set(ranking))!=len(ranking):raise ValueError('duplicate source ranking')
        for position,item in enumerate(ranking,1):scores[item]=scores.get(item,0.)+1/(constant+position)
    return sorted(scores,key=lambda i:(-scores[i],i)),scores


def diversify(ranking,catalog,k=5,per_topic=2):
    if type(k) is not int or k<1 or type(per_topic) is not int or per_topic<1:raise ValueError('diversity budget')
    if len(set(ranking))!=len(ranking):raise ValueError('duplicate ranking')
    topics={i['id']:i['topic'] for i in catalog};counts={};result=[]
    for item in ranking:
        if item not in topics:raise ValueError('unknown item')
        topic=topics[item]
        if counts.get(topic,0)<per_topic:result.append(item);counts[topic]=counts.get(topic,0)+1
        if len(result)==k:break
    return result
