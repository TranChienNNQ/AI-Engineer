import numpy as np
import torch
from data import train_events,seen_at,eligible
from metrics import order,reciprocal_fusion,diversify

NAMES=['popular','content','content_fresh','item_cosine','mf_bpr','hybrid_rrf','hybrid_diverse']


class Policies:
    def __init__(self,data,model=None):
        self.data=data;self.model=model;self.catalog={i['id']:i for i in data['catalog']}
        self.x=np.zeros((48,72))
        for e in train_events(data):self.x[e['user'],e['item']]=1
        self.pop=self.x.sum(axis=0)
        norms=np.sqrt(self.pop);self.sim=self.x.T@self.x/np.maximum(norms[:,None]*norms[None,:],1)
        np.fill_diagonal(self.sim,0.)
    def raw(self,request,ids,kind):
        history=seen_at(self.data,request['user'],request['time'])
        popularity=np.array([self.pop[i] if i<72 else 0. for i in ids])
        if kind=='popular':return popularity
        if kind.startswith('content'):
            if not history:return popularity
            profile=np.bincount([self.catalog[i]['topic'] for i in history],minlength=6)/len(history)
            scores=np.array([profile[self.catalog[i]['topic']] for i in ids])
            if kind=='content_fresh':scores+=np.array([.001*self.catalog[i]['released'] for i in ids])
            return scores
        if kind=='item_cosine':
            if not history:return popularity
            return np.array([sum(self.sim[h,i] for h in history if h<72) if i<72 else 0. for i in ids])
        if kind=='mf_bpr':
            if self.model is None:raise ValueError('model required')
            if request['user']>=48:return popularity
            if any(i>=72 for i in ids):raise ValueError('ID-only MF cannot score unseen item IDs')
            with torch.no_grad():return self.model(torch.full((len(ids),),request['user'],dtype=torch.long),torch.tensor(ids,dtype=torch.long)).numpy()
        raise ValueError('policy kind')
    def recommend(self,request,kind,k=5,blocked=()):
        if kind not in NAMES or type(k) is not int or k<1:raise ValueError('policy/k')
        ids=eligible(self.data,request,blocked)
        if kind in ['hybrid_rrf','hybrid_diverse']:
            # Both sources scan their whole eligible subset. No ANN or learned reranker.
            a=self.recommend(request,'mf_bpr',len(ids) or 1,blocked)
            b=self.recommend(request,'content_fresh',len(ids) or 1,blocked)
            result,_=reciprocal_fusion([a,b])
            return diversify(result,self.data['catalog'],k,2) if kind=='hybrid_diverse' else result[:k]
        if kind=='mf_bpr':ids=[i for i in ids if i<72]
        return order(ids,self.raw(request,ids,kind))[:k]


def evaluate(policies,requests,kind,k=5):
    from metrics import ranking_metrics
    rows=[];all_displayed=set();all_available=set();topic_map=policies.catalog
    for req in requests:
        ids=eligible(policies.data,req)
        if req['target'] not in ids:raise ValueError('evaluation target is not eligible at request time')
        ranking=policies.recommend(req,kind,k);m=ranking_metrics(ranking,[req['target']],k)
        topics=len({topic_map[i]['topic'] for i in ranking});all_displayed.update(ranking)
        all_available.update(i for i,d in topic_map.items() if d['released']<=req['time'] and (d['retired'] is None or req['time']<d['retired']))
        rows.append(dict(user=req['user'],cohort=req['cohort'],target=req['target'],ranking=ranking,eligible_count=len(ids),topics=topics,**m))
    def summary(sub):
        if not sub:return dict(queries=0)
        return dict(queries=len(sub),precision=float(np.mean([r['precision'] for r in sub])),recall=float(np.mean([r['recall'] for r in sub])),
                    ndcg=float(np.mean([r['ndcg'] for r in sub])),mrr=float(np.mean([r['mrr'] for r in sub])),mean_topics=float(np.mean([r['topics'] for r in sub])))
    return dict(overall=summary(rows),cohorts={c:summary([r for r in rows if r['cohort']==c]) for c in ['warm','new_item','new_user']},
                catalog_coverage=len(all_displayed)/len(all_available),unique_displayed=len(all_displayed),available_denominator=len(all_available),rows=rows)


def candidate_demo(policies,requests,k=5,budget=10):
    from metrics import ranking_metrics
    rows=[]
    for req in requests:
        candidate=policies.recommend(req,'popular',budget)
        reranked=order(candidate,policies.raw(req,candidate,'content_fresh'))[:k]
        rows.append(dict(user=req['user'],candidate_hit=int(req['target'] in candidate),
                         **ranking_metrics(reranked,[req['target']],k)))
    return dict(candidate_budget=budget,queries=len(rows),candidate_recall=float(np.mean([r['candidate_hit'] for r in rows])),
                final_recall=float(np.mean([r['recall'] for r in rows])),final_ndcg=float(np.mean([r['ndcg'] for r in rows])),rows=rows,
                scope='popularity top10 then content scoring; no forced target insertion')
