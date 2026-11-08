import copy
import json
from pathlib import Path
import numpy as np
from data import tabular,temporal,calendar,validate_temporal,fingerprint
from detectors import KINDS,fit,score,robust,threshold,cusum,fit_temporal,temporal_scores
from metrics import confusion,average_precision,segments,events
from alerts import apply
from experiment import save_bundle,load_bundle,digest


def run(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);names=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        names.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError):check(name,True)
        else:raise AssertionError(name+' accepted')
    d=tabular();t=temporal();x=d['x'][:300]
    check('tabular split normal references',d['label'][:500].sum()==0 and d['label'][500:620].sum()==12 and d['label'][620:].sum()==18)
    check('tabular anomaly cohorts',all(np.sum(d['kind'][620:]==g)==9 for g in ['off','along']))
    check('temporal clean references',not t['label'][:336].any() and t['label'][336:].sum()==31)
    check('four true temporal events',segments(t['label'])==[(354,355),(384,392),(426,432),(468,484)])
    b=copy.deepcopy(t);b['t'][3]=2;reject('duplicate time rejected',lambda:validate_temporal(b))
    b=copy.deepcopy(t);b['y'][2]=np.nan;reject('missing target rejected',lambda:validate_temporal(b))
    reject('nonfinite table rejected',lambda:fit([[1,2],[np.nan,1]],'knn'))
    reject('shape rejected',lambda:fit(np.ones((10,3)),'knn'))
    reject('constant scale rejected',lambda:fit(np.ones((10,2)),'zmax'))
    reject('zero MAD rejected',lambda:robust([1,1,1]))
    med,s=robust([10,11,12,13,14]);check('MAD hand',med==12 and abs(s-1/.6745)<1e-12)
    m=fit(x,'zmax');check('z score manual',np.allclose(score(m,[[120,55]]),np.max(abs((np.array([[120,55]])-x.mean(0))/x.std(0)),axis=1)))
    m=fit(x,'madmax');check('MAD score manual',np.allclose(score(m,[[120,55]]),np.max(abs((np.array([[120,55]])-np.median(x,axis=0))/(np.median(abs(x-np.median(x,axis=0)),axis=0)/.6745)),axis=1)))
    m=fit(x,'knn');z=(d['x'][500:504]-m['mean'])/m['std'];manual=[]
    for row in z:manual.append(np.mean(sorted(np.linalg.norm(other-row) for other in m['reference'])[:5]))
    check('kNN loop oracle',np.allclose(score(m,d['x'][500:504]),manual))
    m=fit(x,'pca_residual');p=m['projection']@m['projection'].T
    check('PCA idempotent projector',np.allclose(p@p,p) and np.allclose(p,p.T))
    synthetic=m['mean']+10*m['projection'][:,0]*m['std'];check('PCA blind along subspace',score(m,[synthetic])[0]<1e-20)
    off=m['mean']+np.array([-m['projection'][1,0],m['projection'][0,0]])*m['std'];check('PCA residual hand',abs(score(m,[off])[0]-1)<1e-12)
    m=fit(x,'mahalanobis');z=(d['x'][500:504]-m['mean'])/m['std'];cov=(m['reference'].T@m['reference'])/300+.001*np.eye(2)
    check('Mahalanobis solve oracle',np.allclose(score(m,d['x'][500:504]),[v@np.linalg.solve(cov,v) for v in z]))
    for kind in KINDS:
        m=fit(x,kind);mut=copy.deepcopy(d);mut['x'][300:]+=10000;mut['label'][:]=1
        check(kind+' train independent of future/labels',np.array_equal(score(m,d['x'][500:505]),score(fit(mut['x'][:300],kind),d['x'][500:505])))
        q,r=threshold(score(m,d['x'][300:500]));check(kind+' held-out rank',r==191 and np.sum(score(m,d['x'][300:500])>q)<=9)
    q,r=threshold(np.arange(1,20),.1);check('quantile hand and strict tie',q==18 and r==18 and not 18>q)
    reject('insufficient reference rank',lambda:threshold([1],.05))
    reject('bad alpha rejected',lambda:threshold([1,2],1))
    reject('NaN score rejected',lambda:threshold([1,np.nan]))
    c=confusion([0,1,0,1,0],[0,1,1,0,0]);check('confusion hand',c==dict(tp=1,fp=1,fn=1,tn=2,precision=.5,recall=.5,f1=.5,fpr=1/3))
    check('no predictions precision undefined',confusion([0,1],[0,0])['precision'] is None)
    check('no positives recall/AP undefined',confusion([0,0],[1,0])['recall'] is None and average_precision([0,0],[1,2]) is None)
    check('AP hand',abs(average_precision([0,1,0,1,0],[.1,.9,.8,.7,.2])-5/6)<1e-12)
    check('AP ties grouped/permutation invariant',average_precision([1,0],[.5,.5])==average_precision([0,1],[.5,.5])==.5)
    reject('nonbinary rejected',lambda:confusion([2],[1]))
    reject('metric shapes rejected',lambda:average_precision([0,1],[0]))
    check('segments exclusive boundary',segments([1,1,0,1])==[(0,2),(3,4)])
    ev=events([0,1,1,0,1],[0,2],start=0);check('event delay and misses',ev['detected']==1 and ev['missed']==1 and ev['detected_delays']==[1] and ev['false_firings']==1)
    check('empty firing list valid',events([1,0],[])['missed']==1)
    reject('duplicate firings rejected',lambda:events([1,0],[0,0]))
    cs,st=cusum([1,1,1,1]);check('CUSUM hand',np.allclose(cs,[.5,1,1.5,2]) and st==[2.,0.])
    cs,st=cusum([-2,0,2]);check('two-sided CUSUM',np.allclose(cs,[1.5,1,1.5]))
    reject('CUSUM invalid state rejected',lambda:cusum([1],[-1,0]))
    model=fit_temporal(t);mut=copy.deepcopy(t);mut['y'][168:]+=999
    check('calendar fit train only',np.array_equal(model['coef'],fit_temporal(mut)['coef']))
    a,st=temporal_scores(t,model,168,400);mut=copy.deepcopy(t);mut['y'][400:]+=999;b,st2=temporal_scores(mut,model,168,400)
    check('causal future mutation invariance',all(np.array_equal(a[k],b[k]) for k in a) and st==st2)
    mut=copy.deepcopy(t);mut['plan_available_at'][350]=350;reject('time covariate unavailable',lambda:temporal_scores(mut,model,336,400))
    reject('invalid stream boundary',lambda:temporal_scores(t,model,100,120))
    a,st=temporal_scores(t,model,168,430);b,st=temporal_scores(t,model,430,504,st);whole,ws=temporal_scores(t,model,168,504)
    check('CUSUM chunk exact replay',all(np.array_equal(np.r_[a[k],b[k]],whole[k]) for k in a) and st==ws)
    hits=[0,1,1,1,0,0,0,1,1,1,1,1];policy=apply(hits)
    check('alert persistence recovery cooldown hand',policy['firings']==[2,11] and not policy['active'][6])
    p1=apply(hits[:8]);p2=apply(hits[8:],8,json.loads(json.dumps(p1['state'])))
    check('alert chunk/state exact replay',p1['firings']+p2['firings']==policy['firings'] and p2['state']==policy['state'] and p1['active']+p2['active']==policy['active'])
    check('single spike does not fire with persistence two',not apply([0,1,0])['firings'])
    reject('out of order alert state rejected',lambda:apply([1],10,p1['state']))
    reject('invalid alert policy rejected',lambda:apply([1],persist=0))
    for kind in KINDS:
        m=fit(x,kind);path=directory/(kind+'.json');save_bundle(path,m,1.,'test');loaded,q=load_bundle(path,'test')
        check(kind+' JSON exact score roundtrip',np.array_equal(score(m,d['x'][620:]),score(loaded,d['x'][620:])) and q==1.)
    reject('bundle data hash mismatch',lambda:load_bundle(path,'other'))
    blob=json.loads(path.read_text());blob['payload']['threshold']+=1;bad=directory/'bad.json';bad.write_text(json.dumps(blob));reject('bundle digest mismatch',lambda:load_bundle(bad))
    blob['payload']['model']['std'][0]=0;blob['digest']=digest(blob['payload']);bad.write_text(json.dumps(blob));reject('bundle invalid scale despite recomputed digest',lambda:load_bundle(bad))
    blob=json.loads(path.read_text());blob['payload']['model']['reference']=[[1,2]];blob['digest']=digest(blob['payload']);bad.write_text(json.dumps(blob));reject('bundle shape mismatch',lambda:load_bundle(bad))
    return dict(checks=len(names),passed=len(names),names=names)
