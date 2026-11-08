import numpy as np
from data import SPEC


def row(data,origin,horizon=7,known=True,history=None,asof=None):
    if type(origin) is not int or origin<27 or type(horizon) is not int or not 1<=horizon<=7 or origin+horizon>=len(data['t']):raise ValueError('origin/horizon')
    history=np.asarray(data['y'][:origin+1] if history is None else history,dtype=float)
    if history.shape!=(origin+1,) or not np.isfinite(history).all():raise ValueError('exact finite observed/generated prefix required')
    anchor=history[-1];lags=history[-28:]-anchor
    if not known:return lags.copy()
    asof=origin if asof is None else asof;indices=np.r_[origin,np.arange(origin+1,origin+horizon+1)]
    if np.any(data['plan_available_at'][indices]>asof):raise ValueError('future predictor was not available at forecast origin')
    plan=np.asarray(data['plan'][indices])
    if not np.isfinite(plan).all():raise ValueError('unknown/nonfinite planned covariate')
    calendar=np.column_stack([np.sin(2*np.pi*indices/7),np.cos(2*np.pi*indices/7),plan]).ravel()
    return np.r_[lags,calendar]


def supervised(data,end,horizon=7,known=True):
    if type(end) is not int or end<35 or end>len(data['y']):raise ValueError('training cutoff')
    ticks=list(range(27,end-horizon));x=np.stack([row(data,t,horizon,known) for t in ticks])
    y=np.stack([data['y'][t+1:t+horizon+1]-data['y'][t] for t in ticks])
    return x,y,np.array(ticks,dtype=np.int64)


def normalize_fit(x,y):
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('finite supervised matrices required')
    return dict(x_mean=x.mean(axis=0),x_std=np.maximum(x.std(axis=0),1.),y_mean=y.mean(axis=0),y_std=np.maximum(y.std(axis=0),1.))
