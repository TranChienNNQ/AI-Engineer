import math
import numpy as np
from model import weights,forward,tp_parts,pp_parts,payload_bytes,validate
from arithmetic import hand,pipeline,ring_allreduce


def run():
    names=[]
    def check(name,ok):
        if not bool(ok):raise AssertionError(name)
        names.append(name)
    def reject(name,fn):
        try:fn()
        except ValueError:check(name,True);return
        raise AssertionError(name+' accepted')
    h=hand();check('hidden hand',h['hidden']==[1,1,0,3]);check('TP partial hand',h['partials']==[[0,3],[3,-6]]);check('hand output',h['output']==[3.5,-3.5])
    w=weights();rng=np.random.default_rng(39)
    for n in [1,2,7,64]:
        x=rng.normal(size=(n,4)).astype(np.float32);ref=forward(x,w)
        # Independent scalar loops for each hidden/output unit.
        oracle=[]
        for row in x.tolist():
            hidden=[max(0,sum(row[k]*float(w['w1'][k,j]) for k in range(4))+float(w['b1'][j])) for j in range(4)]
            oracle.append([sum(hidden[k]*float(w['w2'][k,j]) for k in range(4))+float(w['b2'][j]) for j in range(2)])
        check(f'scalar MLP oracle B{n}',np.allclose(ref,oracle,atol=2e-6))
        parts=tp_parts(w);partial=[np.maximum(x@p['w1']+p['b1'],0)@p['w2'] for p in parts]
        check(f'TP sum plus one bias B{n}',np.allclose(sum(partial)+w['b2'],ref,atol=2e-6))
        first,second=pp_parts(w);actual=np.maximum(x@first['w1']+first['b1'],0)@second['w2']+second['b2']
        check(f'PP layer split B{n}',np.allclose(actual,ref,atol=2e-6))
    check('DP payload two copies',payload_bytes(w)==120)
    check('TP local payload56 each',list(map(payload_bytes,tp_parts(w)))==[56,56])
    check('PP payload80 40',list(map(payload_bytes,pp_parts(w)))==[80,40])
    check('balanced forward pipeline',h['pipeline_balanced']['makespan']==12 and h['pipeline_balanced']['stage_slot_utilization']==2/3)
    check('unbalanced forward pipeline',h['pipeline_uneven']['makespan']==20 and abs(h['pipeline_uneven']['stage_slot_utilization']-8/15)<1e-12)
    check('ring model 2.9896875ms',abs(h['ring']['estimated_seconds']-.0029896875)<1e-12)
    check('KV transfer lower bound',h['kv_transfer']['bytes']/h['kv_transfer']['bandwidth_bytes_s']==.0625)
    check('mesh ranks',h['mesh']['dp']*h['mesh']['tp']*h['mesh']['pp']==16)
    check('replicated weights not divided',abs(.9*28/4+.1*28-9.1)<1e-12)
    reject('invalid input dim',lambda:validate([[1,2]]));reject('empty model batch',lambda:validate(np.zeros((0,4))))
    reject('oversized model batch',lambda:validate(np.zeros((65,4))));reject('nonfinite model input',lambda:validate([[0,0,0,float('inf')]]))
    reject('invalid stage time',lambda:pipeline([2,0],4));reject('invalid microbatch count',lambda:pipeline([2],0))
    reject('invalid ring size',lambda:ring_allreduce(1,8,8,0));reject('invalid bandwidth',lambda:ring_allreduce(2,8,0,0))
    return dict(checks=len(names),passed=len(names),names=names)
