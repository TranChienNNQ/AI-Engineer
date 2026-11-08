"""Synthetic observations; labels/hidden means are for evaluation/illustration only."""
import hashlib
import numpy as np
TAB_SPEC=dict(version='correlated-v1',features=['request_rate','latency'],train_end=300,calibration_end=500,validation_end=620,total=800)
TIME_SPEC=dict(version='hourly-v1',train_end=168,calibration_end=336,total=504,period=24)


def tabular():
    rng=np.random.default_rng(44);u=rng.normal(size=800);v=.15*rng.normal(size=800)
    label=np.zeros(800,dtype=int);kind=np.full(800,'normal',dtype='<U8')
    for start,end,count in [(500,620,12),(620,800,18)]:
        indices=np.sort(rng.choice(np.arange(start,end),count,replace=False))
        for j,i in enumerate(indices):
            label[i]=1;sign=(-1.)**(j//2)
            if j%2==0:v[i]=sign*(1.0+.1*rng.random());kind[i]='off'
            else:u[i]=sign*(3.2+.4*rng.random());kind[i]='along'
    x=np.column_stack([100+20*u,50+10*(u+v)])
    return dict(x=x,label=label,kind=kind)


def calendar(t,plan):
    t=np.asarray(t);plan=np.asarray(plan)
    if t.ndim!=1 or plan.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(plan).all():raise ValueError('calendar contract')
    return np.column_stack([np.ones(len(t)),np.sin(2*np.pi*t/24),np.cos(2*np.pi*t/24),plan])


def temporal():
    t=np.arange(504);rng=np.random.default_rng(440);noise=np.zeros(504)
    for i in t:noise[i]=(.4*noise[i-1] if i else 0)+rng.normal(scale=.8)
    plan=(((t%168)>=24)&((t%168)<30)).astype(float)
    expected=calendar(t,plan)@np.array([100.,10.,4.,12.])
    effect=np.zeros(504);effect[354]=20;effect[384:392]=8;effect[426:432]=-10;effect[468:484]=2.5
    y=expected+noise+effect
    return dict(t=t,y=y,plan=plan,plan_available_at=t-48,label=(effect!=0).astype(int),effect=effect,expected=expected)


def validate_temporal(d):
    t,y=np.asarray(d['t']),np.asarray(d['y'])
    if t.ndim!=1 or not np.array_equal(t,np.arange(len(t))) or y.shape!=t.shape or not np.isfinite(y).all():raise ValueError('complete finite hourly grid required')
    if np.asarray(d['plan']).shape!=t.shape or np.asarray(d['plan_available_at']).shape!=t.shape:raise ValueError('covariate shape')


def fingerprint(data,keys):
    h=hashlib.sha256()
    for key in keys:h.update(key.encode());h.update(np.asarray(data[key]).tobytes())
    return h.hexdigest()
