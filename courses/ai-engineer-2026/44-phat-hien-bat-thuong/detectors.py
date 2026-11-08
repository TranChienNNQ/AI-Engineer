import math
import numpy as np
from data import calendar,validate_temporal
KINDS=['zmax','madmax','knn','pca_residual','mahalanobis']


def matrix(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<1 or not np.isfinite(x).all():raise ValueError('finite nonempty matrix with two features required')
    return x


def robust(a,axis=0):
    a=np.asarray(a,dtype=float)
    if not a.size or not np.isfinite(a).all():raise ValueError('finite reference required')
    med=np.median(a,axis=axis);mad=np.median(np.abs(a-med),axis=axis);scale=mad/.6745
    if np.any(scale<=1e-10):raise ValueError('zero/near-zero MAD: define a domain fallback instead of dividing')
    return med,scale


def fit(x,kind):
    x=matrix(x)
    if kind not in KINDS or len(x)<6:raise ValueError('detector kind/reference size')
    mean=x.mean(0);std=x.std(0)
    if np.any(std<=1e-10):raise ValueError('zero/near-zero feature scale')
    z=(x-mean)/std;center,madscale=robust(x);cov=z.T@z/len(x)
    _,vectors=np.linalg.eigh(cov);projection=vectors[:,-1:];inverse=np.linalg.inv(cov+.001*np.eye(2))
    return dict(kind=kind,mean=mean,std=std,median=center,madscale=madscale,
        reference=z,projection=projection,inverse=inverse,k=5)


def score(model,x):
    x=matrix(x);z=(x-model['mean'])/model['std'];kind=model['kind']
    if kind=='zmax':s=np.max(np.abs(z),axis=1)
    elif kind=='madmax':s=np.max(np.abs((x-model['median'])/model['madscale']),axis=1)
    elif kind=='knn':
        distance=np.sqrt(np.sum((z[:,None,:]-model['reference'][None,:,:])**2,axis=2))
        s=np.sort(distance,axis=1)[:,:model['k']].mean(1)
    elif kind=='pca_residual':
        residual=z-z@model['projection']@model['projection'].T;s=np.sum(residual**2,axis=1)
    elif kind=='mahalanobis':s=np.einsum('ni,ij,nj->n',z,model['inverse'],z)
    else:raise ValueError('unknown score')
    if not np.isfinite(s).all():raise ValueError('nonfinite score')
    return s


def threshold(scores,alpha=.05):
    s=np.asarray(scores,dtype=float)
    if s.ndim!=1 or not s.size or not np.isfinite(s).all() or not 0<alpha<1:raise ValueError('calibration scores/alpha')
    rank=math.ceil((len(s)+1)*(1-alpha))
    if rank>len(s):raise ValueError('too few reference scores for requested rank')
    return float(np.sort(s)[rank-1]),rank


def cusum(residual,state=(0.,0.),k=.5):
    r=np.asarray(residual,dtype=float);g=np.asarray(state,dtype=float)
    if r.ndim!=1 or not np.isfinite(r).all() or g.shape!=(2,) or not np.isfinite(g).all() or np.any(g<0) or not np.isfinite(k) or k<=0:raise ValueError('CUSUM residual/state/k')
    plus,minus=g;values=[]
    for value in r:
        plus=max(0.,plus+value-k);minus=max(0.,minus-value-k);values.append(max(plus,minus))
    return np.asarray(values),[float(plus),float(minus)]


def fit_temporal(d,end=168):
    validate_temporal(d)
    if type(end) is not int or end<48 or end>len(d['y']):raise ValueError('training cutoff')
    ticks=d['t'][:end]
    if np.any(d['plan_available_at'][:end]>ticks-1):raise ValueError('unavailable training covariate')
    x=calendar(ticks,d['plan'][:end]);coef=np.linalg.lstsq(x,d['y'][:end],rcond=None)[0]
    center,res_scale=robust(d['y'][:end]-x@coef);raw_center,raw_scale=robust(d['y'][:end])
    seasonal_center,seasonal_scale=robust(d['y'][24:end]-d['y'][:end-24])
    return dict(coef=coef,res_center=float(center),res_scale=float(res_scale),raw_center=float(raw_center),raw_scale=float(raw_scale),seasonal_center=float(seasonal_center),seasonal_scale=float(seasonal_scale),train_end=end)


def temporal_scores(d,m,start,end,state=(0.,0.)):
    validate_temporal(d)
    if type(start) is not int or type(end) is not int or start<max(24,m['train_end']) or end<=start or end>len(d['y']):raise ValueError('stream bounds')
    ix=np.arange(start,end)
    if np.any(d['plan_available_at'][ix]>ix-1):raise ValueError('plan unavailable before observation')
    expected=calendar(ix,d['plan'][ix])@m['coef'];residual=(d['y'][ix]-expected-m['res_center'])/m['res_scale']
    cumulative,newstate=cusum(residual,state)
    return dict(global_mad=np.abs((d['y'][ix]-m['raw_center'])/m['raw_scale']),
        seasonal_naive=np.abs((d['y'][ix]-d['y'][ix-24]-m['seasonal_center'])/m['seasonal_scale']),
        calendar_residual=np.abs(residual),cusum=cumulative),newstate
