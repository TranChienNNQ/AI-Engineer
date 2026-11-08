import numpy as np
from metrics import point,residual_quantiles,interval


def hand():
    history=np.array([10.,12.,14.,16.]);actual=np.array([0.,10.,20.]);pred=np.array([2.,8.,25.]);q,rank=residual_quantiles(np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]],dtype=float),.2)
    return dict(baselines=dict(history=history.tolist(),naive=[16.,16.,16.],mean=[13.,13.,13.],drift=[18.,20.,22.],
            seasonal_history=[10,20,30,40,50,60,70,11,21,31,41,51,61,71],seasonal_next=[11,21,31,41,51,61,71]),
        recursion=dict(y0=10.,ar=.8,intercept=1.,forecast=[9.,8.2,7.56],note='AR demo 1+.8*y; future actuals never fed in'),
        lag_row=dict(origin=6,lookback=3,features=[5,6,7],target=[8,9],last_target_index=8),
        metrics=point(actual,pred,2.),calibration=dict(errors=list(range(1,10)),alpha=.2,rank=rank,q=float(q[0]),interval=[20-float(q[0]),20+float(q[0])]),
        interval_score=interval([5.],[0.],[4.],.2),
        windows=dict(initial_end=224,lookback=28,horizon=7,rows=190,last_origin=216,last_target=223))
