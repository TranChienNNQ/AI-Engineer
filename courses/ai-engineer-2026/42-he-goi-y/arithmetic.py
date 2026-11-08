import math
import numpy as np
from metrics import ranking_metrics,reciprocal_fusion,diversify


def hand():
    p=np.array([1.,0.]);qi=np.array([1.,0.]);qj=np.array([0.,1.]);delta=float(p@(qi-qj));g=1/(1+math.exp(delta))
    gp=-g*(qi-qj);gi=-g*p;gj=g*p;lr=.1
    return dict(mf=dict(user=[.8,.2],items=[[1.,0.],[0.,1.]],scores=[.8,.2],rating_with_bias=4.4),
        bpr=dict(delta=delta,loss=math.log1p(math.exp(-delta)),gradient_p=gp.tolist(),gradient_i=gi.tolist(),gradient_j=gj.tolist(),
                 after_p=(p-lr*gp).tolist(),after_i=(qi-lr*gi).tolist(),after_j=(qj-lr*gj).tolist(),learning_rate=lr),
        metrics=ranking_metrics(['A','B','C','D','E'],['B','E','F'],5),
        content=dict(history_topics=[0,0,1],profile=[2/3,1/3],scores=[2/3,1/3]),
        ips=dict(rewards=[1,0],propensities=[.5,.25],sample_mean=1.,note='mean(r/p) for two hypothetical matched logged actions; illustrative contribution, not a deployed OPE estimator'),
        sampled=dict(full_rank=20,full_candidates=100,full_hit5=0,sampled_rank=2,sampled_candidates=6,sampled_hit5=1,
                     note='a possible selected subset of negatives, not an unbiased estimate or expectation'))
