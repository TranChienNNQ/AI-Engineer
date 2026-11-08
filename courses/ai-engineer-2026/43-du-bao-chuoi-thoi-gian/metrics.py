import math
import numpy as np


def seasonal_scale(train,period=7):
    a=np.asarray(train,dtype=float)
    if a.ndim!=1 or type(period) is not int or period<1 or len(a)<=period or not np.isfinite(a).all():raise ValueError('scale history')
    scale=float(np.mean(np.abs(a[period:]-a[:-period])))
    if scale<=0:raise ValueError('MASE scale is zero')
    return scale


def point(actual,prediction,scale):
    y,p=np.asarray(actual,dtype=float),np.asarray(prediction,dtype=float)
    if y.shape!=p.shape or y.size==0 or not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(scale) or scale<=0:raise ValueError('point metric contract')
    error=p-y;mae=float(np.mean(np.abs(error)));volume=float(np.sum(np.abs(y)))
    return dict(mae=mae,rmse=float(np.sqrt(np.mean(error**2))),bias=float(np.mean(error)),mase=mae/scale,
        wape=None if volume==0 else float(np.sum(np.abs(error))/volume))


def residual_quantiles(errors,alpha=.2):
    e=np.asarray(errors,dtype=float)
    if e.ndim!=2 or min(e.shape)<1 or not np.isfinite(e).all() or not 0<alpha<1:raise ValueError('calibration matrix/alpha')
    n=len(e);rank=math.ceil((n+1)*(1-alpha))
    if rank>n:raise ValueError('too few calibration origins for this finite order-statistic level')
    return np.sort(np.abs(e),axis=0)[rank-1],rank


def interval(actual,lower,upper,alpha=.2):
    y,lo,hi=[np.asarray(v,dtype=float) for v in [actual,lower,upper]]
    if y.shape!=lo.shape or y.shape!=hi.shape or y.size==0 or not 0<alpha<1 or not all(np.isfinite(a).all() for a in [y,lo,hi]) or np.any(lo>hi):raise ValueError('interval contract')
    score=(hi-lo)+2/alpha*np.maximum(lo-y,0)+2/alpha*np.maximum(y-hi,0)
    return dict(coverage=float(np.mean((lo<=y)&(y<=hi))),mean_width=float(np.mean(hi-lo)),interval_score=float(np.mean(score)))
