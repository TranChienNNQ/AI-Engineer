import numpy as np


def labels(y):
    y=np.asarray(y)
    if y.ndim!=1 or not y.size or not np.isin(y,[0,1]).all():raise ValueError('nonempty binary labels')
    return y.astype(bool)


def confusion(y,pred):
    y=labels(y);pred=labels(pred)
    if y.shape!=pred.shape:raise ValueError('label shape')
    tp=int(np.sum(y&pred));fp=int(np.sum(~y&pred));fn=int(np.sum(y&~pred));tn=int(np.sum(~y&~pred))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        fpr=fp/(fp+tn) if fp+tn else None)


def average_precision(y,scores):
    y=labels(y);s=np.asarray(scores,dtype=float)
    if s.shape!=y.shape or not np.isfinite(s).all():raise ValueError('score shape/finite')
    if not y.sum():return None
    total=0.;seen=0;tp=0
    for value in np.unique(s)[::-1]:
        group=s==value;positives=int(y[group].sum());seen+=int(group.sum());tp+=positives
        total+=(positives/int(y.sum()))*(tp/seen)
    return float(total)


def segments(y):
    y=labels(y);edges=np.diff(np.r_[False,y,False].astype(int))
    return [(int(a),int(b)) for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0])]


def events(y,firings,start=0):
    y=labels(y);raw=np.asarray(firings)
    if raw.ndim!=1 or (len(raw)>0 and not np.issubdtype(raw.dtype,np.integer)) or len(set(raw.tolist()))!=len(raw) or np.any(raw<start) or np.any(raw>=start+len(y)):raise ValueError('unique integer firing timestamps within interval required')
    fires=sorted(raw.tolist());ranges=[(a+start,b+start) for a,b in segments(y)];delays=[]
    for a,b in ranges:
        matched=[f for f in fires if a<=f<b]
        if matched:delays.append(matched[0]-a)
    return dict(events=len(ranges),detected=len(delays),missed=len(ranges)-len(delays),event_recall=len(delays)/len(ranges) if ranges else None,
        detected_delays=delays,mean_detected_delay=float(np.mean(delays)) if delays else None,
        firings=len(fires),false_firings=sum(not any(a<=f<b for a,b in ranges) for f in fires))
