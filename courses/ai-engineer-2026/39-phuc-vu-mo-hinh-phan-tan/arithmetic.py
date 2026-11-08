"""Hypothetical ledgers and forward-only timing models, not hardware benchmarks."""
import math
import numpy as np
from model import weights,forward,tp_parts,payload_bytes


def pipeline(stage_times,microbatches):
    if not stage_times or any(not math.isfinite(t) or t<=0 for t in stage_times) or type(microbatches) is not int or microbatches<1:raise ValueError('positive times/count')
    finish=[]
    for i in range(microbatches):
        row=[]
        for j,t in enumerate(stage_times):row.append(max(finish[i-1][j] if i else 0,row[j-1] if j else 0)+t)
        finish.append(row)
    makespan=finish[-1][-1];work=microbatches*sum(stage_times)
    return dict(finish=finish,makespan=makespan,first_latency=sum(stage_times),stage_slot_utilization=work/(len(stage_times)*makespan))


def ring_allreduce(ranks,bytes_per_rank,bandwidth_bytes_s,alpha_s):
    if type(ranks) is not int or ranks<2 or type(bytes_per_rank) is not int or bytes_per_rank<1 or not math.isfinite(bandwidth_bytes_s) or bandwidth_bytes_s<=0 or not math.isfinite(alpha_s) or alpha_s<0:raise ValueError('ring inputs')
    return 2*(ranks-1)*alpha_s+2*(ranks-1)/ranks*bytes_per_rank/bandwidth_bytes_s


def hand():
    w=weights();x=np.array([[1,2,0,-1]],dtype=np.float32);parts=tp_parts(w);h=np.maximum(x@w['w1']+w['b1'],0)
    partial=[(np.maximum(x@p['w1']+p['b1'],0)@p['w2']).tolist()[0] for p in parts]
    return dict(hidden=h.tolist()[0],partials=partial,output=forward(x,w).tolist()[0],weights_bytes=payload_bytes(w),
                pipeline_balanced=pipeline([2.,2.,2.],4),pipeline_uneven=pipeline([2.,4.,2.],4),
                ring=dict(ranks=4,payload_bytes=16*1024**2,bandwidth_bytes_s=8*1024**3,alpha_s=10e-6,
                          estimated_seconds=ring_allreduce(4,16*1024**2,8*1024**3,10e-6)),
                kv_transfer=dict(bytes=512*1024**2,bandwidth_bytes_s=8*1024**3,lower_bound_seconds=.0625),
                mesh=dict(dp=2,tp=4,pp=2,total_ranks=16,model_groups=2,ranks_per_group=8),
                memory=dict(total_weight_gib=28,sharded_fraction=.9,tp=4,replicated_gib=2.8,sharded_per_rank_gib=6.3,
                            weight_per_rank_gib=9.1,other_per_rank_gib=3,rank_budget_gib=16,remaining_gib=3.9),
                scope='all times and large-memory numbers are hypothetical; no topology/allocator/GPU capacity claim')
