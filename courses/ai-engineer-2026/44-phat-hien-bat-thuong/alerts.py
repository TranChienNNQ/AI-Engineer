"""Causal persistence/recovery/cooldown policy, separate from the scoring model."""
import copy
import numpy as np


def initial(start):return dict(high_run=0,low_run=0,active=False,next_allowed=start,last_tick=start-1)


def apply(hits,start=0,state=None,persist=2,clear=3,cooldown=4):
    hits=np.asarray(hits)
    if hits.ndim!=1 or not np.isin(hits,[0,1]).all() or type(start) is not int or start<0 or any(type(x) is not int or x<1 for x in [persist,clear]) or type(cooldown) is not int or cooldown<0:raise ValueError('alert input/policy')
    s=initial(start) if state is None else copy.deepcopy(state)
    if set(s)!=set(initial(start)) or type(s['active']) is not bool or any(type(s[k]) is not int for k in ['high_run','low_run','next_allowed','last_tick']) or s['last_tick']!=start-1 or min(s['high_run'],s['low_run'],s['next_allowed'])<0:raise ValueError('alert state/continuity')
    fires=[];active=[]
    for tick,hit in enumerate(hits,start):
        if hit:s['high_run']+=1;s['low_run']=0
        else:s['low_run']+=1;s['high_run']=0
        if s['active']:
            if s['low_run']>=clear:s['active']=False;s['next_allowed']=tick+cooldown+1
        elif hit and s['high_run']>=persist and tick>=s['next_allowed']:
            s['active']=True;fires.append(tick)
        s['last_tick']=tick;active.append(s['active'])
    return dict(firings=fires,active=active,state=s)
